# SHA-256 / PBKDF2-HMAC-SHA256 Benchmark Pipeline for GPU Password Cracking 🚀

This repository contains my  personal tests with high-performance benchmark implementation of **PBKDF2-HMAC-SHA256** (and its SHA-256 inner primitives) built around a **pipeline optimized for GPU password-cracking workloads**—specifically the common scenario where you iterate through a wordlist while keeping most of the input structure constant.

In practice, **SHA-256 is often lighter than SHA-512** for portable implementations and tends to map well to GPU architectures due to its **32-bit datapath**, instruction throughput, and register pressure characteristics. The kernel is engineered to minimize redundant work per candidate and maximize arithmetic density.

## Core idea: host-side prefix precomputation (“fixed salt / fixed prefix”)

The main optimization is treating the input as:

**prefix || password**

Where the **prefix** (e.g., a fixed salt, constant header, or known message prefix) is **precomputed on the host** and injected into the kernel in a form that reduces per-candidate overhead. In wordlist-style cracking, this can yield major speedups because the GPU work becomes mostly “rotate password bytes + finalize”.

Depending on the exact prefix structure and the baseline you compare against, this technique can improve throughput dramatically (often quoted as “up to ~3×” in favorable layouts).

---

## Implementation details / modifications 📝

* **Fixed `dkLen` / `pkLen` (32 bytes)**
  The benchmark targets a fixed derived-key length (32 bytes) to keep the code path deterministic and remove branching that would otherwise harm occupancy.

* **Fixed prefix/salt precomputed on host**
  The host prepares the constant portion of the HMAC input (ipad/opad path included where applicable), so each GPU thread only patches the password region and completes the compression.

* **HMAC precomputation outside the iteration loop**
  PBKDF2 performs many repeated HMAC operations. When the “static” part of the message is constant, you can precompute the HMAC state so the kernel does less work per iteration.

* **Custom PBKDF2 loop structure (aggressive unrolling + reuse)**
  The SHA-256 rounds are unrolled to reduce loop overhead and enable instruction scheduling. The implementation also focuses on **intermediate reuse**, especially around boolean functions (e.g., MAJ), to reduce recomputation and lower the dynamic instruction count.

* **32-bit packing (`uint32`) instead of byte-wise (`uchar`)**
  The data path is optimized for **word-based loads/stores**. Packing message material into `uint32` improves memory coalescing, reduces byte shuffles, and speeds up the inner loop (especially when the padding/length fields are predictable).

> **Security / usage note ⚠️**
> This is a **benchmark + study implementation** meant for controlled environments and authorized recovery work. Do **not** treat it as a drop-in production KDF library.

---

## Performance snapshot: Custom kernel vs Hashcat (PBKDF2-HMAC-SHA256)

The table below summarizes measured throughput for PBKDF2-HMAC-SHA256 at different iteration counts. “deriv/s” here means full PBKDF2 derivations per second (equivalent to H/s in this context). Hashcat figures should be treated as **benchmark-mode** results on optimized kernels; your exact numbers will vary with driver version, clocks, and kernel choice.

| Iterations | Custom Speed (deriv/s) | Custom Time |                                Hashcat Speed (H/s) | Hashcat Time | Notes                                                                                                                |
| ---------: | ---------------------: | ----------: | -------------------------------------------------: | -----------: | -------------------------------------------------------------------------------------------------------------------- |
|         64 |            9,757,510.6 |         N/A | ~3,308,400 *(low-iter estimate / comparable mode)* |          N/A | Custom shows ~3× speedup at low iteration counts where precompute dominates. **Match: True** (reference verified).   |
|        999 |            3,722,074.4 |     0.013 s |                                          3,308,400 |          N/A | Similar throughput; custom benefits from reduced setup overhead per candidate. **Match: True** (reference verified). |

Reference hash validation example (custom output matches known-good reference):
`45904325a0cbd53058c729bac47d91db47c2a6dc5492a20dbce75ddddad88fee`

**Interpretation:**
This style of optimization is strongest in **low-to-mid iteration regimes** and in workflows where a **large fraction of the input is constant across candidates** (wordlist rotation, known prefix formats, fixed salt layouts). General-purpose tools (e.g., Hashcat) remain extremely competitive across many modes because they must support broad formats and dynamic inputs.

---

## Real-world usage of PBKDF2 and iteration counts 🌐 (configuration varies)

PBKDF2 is still deployed across many products and ecosystems, but **iteration counts are not static**—they can change with versions, policies, and user settings. Treat the numbers below as **typical/representative**, and verify against current vendor documentation or the specific artifact you are analyzing.

| System / Product                           | Typical KDF                    |          Typical iteration count (varies) |
| ------------------------------------------ | ------------------------------ | ----------------------------------------: |
| Password managers (some configurations)    | PBKDF2-HMAC-SHA256             |        hundreds of thousands to ~millions |
| Bitcoin wallet ecosystems (some formats)   | PBKDF2-HMAC-SHA512             |          a few thousand (format-specific) |
| OS / disk encryption stacks (some configs) | Iterated SHA-256 / PBKDF2-like |       can be very high (policy-dependent) |
| Web frameworks (e.g., Django defaults)     | PBKDF2-SHA256                  | often ~hundreds of thousands to ~millions |

**Why it matters:** higher iteration counts raise brute-force cost, but GPU-centric optimization still matters when the attacker/defender workload involves massive candidate throughput or when you’re benchmarking recovery pipelines.










Bruno da Silva
2026



