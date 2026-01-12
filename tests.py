# run_pbkdf2_opencl.py
from __future__ import annotations

import hashlib
import struct
import time
from pathlib import Path

import numpy as np
import pyopencl as cl


KERNEL_FILE = "pbkdf2_hmac_sha256_u32.cl"

PW_WORDS = 16           # 16*4 = 64 bytes por senha (trunca/pad)
OUT_WORDS = 8           # SHA256 = 32 bytes = 8 u32
MSG_WORDS = 17          # salt + block_index (>= 4 bytes) + folga

LOCAL_SIZE = 64


def pick_device(prefer_gpu: bool = True) -> cl.Device:
    plats = cl.get_platforms()
    if not plats:
        raise RuntimeError("Nenhuma plataforma OpenCL encontrada.")

    best = None
    for p in plats:
        devs = p.get_devices()
        if not devs:
            continue
        if prefer_gpu:
            gpus = [d for d in devs if d.type & cl.device_type.GPU]
            if gpus:
                best = gpus[0]
                break
        best = devs[0]

    if best is None:
        raise RuntimeError("Nenhum device OpenCL encontrado.")
    return best


def pack_be_u32_fixed(data: bytes, num_words: int) -> np.ndarray:
    b = data[: 4 * num_words].ljust(4 * num_words, b"\x00")
    return np.frombuffer(b, dtype=">u4").astype(np.uint32, copy=False)


def unpack_be_u32_words(words_u32: np.ndarray) -> bytes:
    # words_u32: uint32 nativo -> bytes big-endian
    return b"".join(int(w).to_bytes(4, "big") for w in words_u32.tolist())


def make_password_batch(passwords: list[bytes], n_total: int) -> tuple[np.ndarray, np.ndarray]:
    if not passwords:
        raise ValueError("passwords vazio.")

    n = int(n_total)
    pw_words = np.zeros((n, PW_WORDS), dtype=np.uint32)
    pw_len = np.zeros(n, dtype=np.uint16)

    m = len(passwords)
    for i in range(n):
        pw = passwords[i % m]
        pw_len[i] = min(len(pw), PW_WORDS * 4)
        pw_words[i, :] = pack_be_u32_fixed(pw, PW_WORDS)

    return pw_words.reshape(-1), pw_len


def build_program(ctx: cl.Context, kernel_path: str) -> cl.Program:
    src = Path(kernel_path).read_text(encoding="utf-8")
    return cl.Program(ctx, src).build(options=["-cl-std=CL1.2"])


def fmt_gib(x: float) -> str:
    return f"{x / (1024**3):.3f} GiB"


def benchmark(
    prg: cl.Program,
    queue: cl.CommandQueue,
    pw_words_flat: np.ndarray,
    pw_len: np.ndarray,
    msg_words: np.ndarray,
    msg_len: int,
    iterations: int,
    dklen: int,
    out_stride_words: int,
    local_size: int = LOCAL_SIZE,
    warmup: int = 2,
    runs: int = 5,
) -> tuple[np.ndarray, dict]:
    ctx = queue.context
    mf = cl.mem_flags

    n = int(pw_len.size)
    if n % local_size != 0:
        n = n - (n % local_size)
        pw_len = pw_len[:n]
        pw_words_flat = pw_words_flat[: n * PW_WORDS]

    out_words = np.empty((n, OUT_WORDS), dtype=np.uint32)

    d_pw_words = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=pw_words_flat)
    d_pw_len = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=pw_len)
    d_msg_words = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=msg_words)
    d_out_words = cl.Buffer(ctx, mf.WRITE_ONLY, size=out_words.nbytes)

    gsize = (n,)
    lsize = (local_size,)

    def launch() -> cl.Event:
        return prg.pbkdf2_hmac_sha256_u32(
            queue, gsize, lsize,
            d_pw_words, d_pw_len, np.uint32(PW_WORDS),
            d_msg_words, np.uint32(msg_len),
            np.uint32(iterations), np.uint32(dklen),
            d_out_words, np.uint32(out_stride_words),
        )

    # warmup (compilação JIT + caches)
    for _ in range(int(warmup)):
        ev = launch()
        ev.wait()

    # mede kernel-only via profiling event + mede end-to-end também
    kernel_ms = []
    e2e_ms = []

    for _ in range(int(runs)):
        t0 = time.perf_counter()
        ev = launch()
        ev.wait()
        t1 = time.perf_counter()

        # kernel-only (profiling)
        # start/end em nanos
        k_ms = (ev.profile.end - ev.profile.start) * 1e-6
        kernel_ms.append(k_ms)
        e2e_ms.append((t1 - t0) * 1e3)

    cl.enqueue_copy(queue, out_words, d_out_words).wait()

    kernel_ms = np.array(kernel_ms, dtype=np.float64)
    e2e_ms = np.array(e2e_ms, dtype=np.float64)

    stats = {
        "N": n,
        "iterations": int(iterations),
        "dklen": int(dklen),
        "local_size": int(local_size),
        "kernel_ms_min": float(kernel_ms.min()),
        "kernel_ms_med": float(np.median(kernel_ms)),
        "kernel_ms_avg": float(kernel_ms.mean()),
        "kernel_ms_max": float(kernel_ms.max()),
        "e2e_ms_min": float(e2e_ms.min()),
        "e2e_ms_med": float(np.median(e2e_ms)),
        "e2e_ms_avg": float(e2e_ms.mean()),
        "e2e_ms_max": float(e2e_ms.max()),
    }

    # throughput usando tempo do kernel (mais “honesto”)
    k_s = stats["kernel_ms_med"] / 1e3
    stats["throughput_deriv_s_med_kernel"] = float(n / k_s) if k_s > 0 else float("inf")

    # aproximado bytes trafegados (host não incluso): pw + len + out
    bytes_in = n * (PW_WORDS * 4 + 2) + (MSG_WORDS * 4)
    bytes_out = n * (OUT_WORDS * 4)
    stats["io_bytes_est"] = int(bytes_in + bytes_out)
    stats["io_est_gib"] = stats["io_bytes_est"] / (1024**3)

    return out_words, stats


def main():
    # parâmetros principais (ajuste aqui)
    salt = b"ab333333333cd"
    iterations = 999
    dklen = 32
    n = 5_000_000  # mantenha múltiplo de LOCAL_SIZE pra evitar truncar

    # Senhas para benchmark real do kernel: repete poucas entradas
    passwords = [
        b"333333333",
        b"joao",
        b"nuo3uhuoi0nhi0password",
    ]

    dev = pick_device(prefer_gpu=True)
    ctx = cl.Context([dev])
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

    print("OpenCL device:")
    print(f"  Name: {dev.name}")
    print(f"  Vendor: {dev.vendor}")
    print(f"  Driver: {dev.driver_version}")
    print(f"  Compute Units: {dev.max_compute_units}")
    print(f"  Max WG size: {dev.max_work_group_size}")
    print()

    prg = build_program(ctx, KERNEL_FILE)

    # PBKDF2 block index (1) para dkLen=32
    msg = salt + struct.pack(">I", 1)
    msg_len = len(msg)
    msg_words = pack_be_u32_fixed(msg, MSG_WORDS)

    t_pack0 = time.perf_counter()
    pw_words_flat, pw_len = make_password_batch(passwords, n)
    t_pack1 = time.perf_counter()

    print("Batch:")
    print(f"  N: {pw_len.size:,}")
    print(f"  PW_WORDS: {PW_WORDS} ({PW_WORDS*4} bytes/entry)")
    print(f"  msg_len: {msg_len} bytes | iterations: {iterations} | dkLen: {dklen}")
    print(f"  Host pack time: {(t_pack1 - t_pack0)*1e3:.2f} ms")
    print()

    out_words, stats = benchmark(
        prg=prg,
        queue=queue,
        pw_words_flat=pw_words_flat,
        pw_len=pw_len,
        msg_words=msg_words,
        msg_len=msg_len,
        iterations=iterations,
        dklen=dklen,
        out_stride_words=OUT_WORDS,
        local_size=LOCAL_SIZE,
        warmup=2,
        runs=7,
    )

    # valida 1 amostra
    ref = hashlib.pbkdf2_hmac("sha256", passwords[0], salt, iterations, dklen=dklen)
    gpu = unpack_be_u32_words(out_words[0])[:dklen]

    print("Benchmark (median):")
    print(f"  Kernel time: {stats['kernel_ms_med']:.3f} ms  (min {stats['kernel_ms_min']:.3f} / max {stats['kernel_ms_max']:.3f})")
    print(f"  End-to-end:  {stats['e2e_ms_med']:.3f} ms  (min {stats['e2e_ms_min']:.3f} / max {stats['e2e_ms_max']:.3f})")
    print(f"  Throughput (kernel): {stats['throughput_deriv_s_med_kernel']:,.1f} deriv/s")
    print(f"  I/O estimate: {fmt_gib(stats['io_bytes_est'])}")
    print()

    print("Validation:")
    print("  Match:", gpu == ref)
    print("  Ref:", ref.hex())
    print("  GPU:", gpu.hex())


if __name__ == "__main__":
    main()
