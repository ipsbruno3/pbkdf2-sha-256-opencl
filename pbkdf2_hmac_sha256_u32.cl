typedef uint   u32;
typedef ulong  u64;
typedef ushort u16;

#define ROTR32(x,n)  (((x) >> (n)) | ((x) << (32u - (n))))
#define CH(x,y,z)    (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x,y,z)   (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define BSIG0(x)     (ROTR32((x), 2u) ^ ROTR32((x),13u) ^ ROTR32((x),22u))
#define BSIG1(x)     (ROTR32((x), 6u) ^ ROTR32((x),11u) ^ ROTR32((x),25u))
#define SSIG0(x)     (ROTR32((x), 7u) ^ ROTR32((x),18u) ^ ((x) >> 3u))
#define SSIG1(x)     (ROTR32((x),17u) ^ ROTR32((x),19u) ^ ((x) >> 10u))

__constant u32 K256[64] = {
  0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
  0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
  0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
  0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
  0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
  0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
  0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
  0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
};

static inline u32 mask_keep_be_bytes(u32 n) {
  if (n == 0u) return 0x00000000u;
  if (n == 1u) return 0xFF000000u;
  if (n == 2u) return 0xFFFF0000u;
  if (n == 3u) return 0xFFFFFF00u;
  return 0xFFFFFFFFu;
}

static inline void set_be_byte(u32 w[], u32 byte_pos, u32 v) {
  u32 wi = (byte_pos >> 2);
  u32 bi = (byte_pos & 3u);
  u32 sh = 24u - (bi << 3);
  u32 m  = 0xFFu << sh;
  w[wi] = (w[wi] & ~m) | ((v & 0xFFu) << sh);
}

static inline void zero_words(u32 *x, u32 n) {
  for (u32 i = 0u; i < n; i++) x[i] = 0u;
}

static inline void trim_be_words(u32 *x, u32 len_bytes, u32 max_words) {
  u32 full = (len_bytes >> 2);
  u32 rem  = (len_bytes & 3u);

  if (full >= max_words) return;

  if (rem) {
    x[full] &= mask_keep_be_bytes(rem);
    full += 1u;
  }

  #pragma unroll
  for (u32 i = full; i < max_words; i++) x[i] = 0u;
}

static inline void sha256_init(u32 st[8]) {
  st[0]=0x6a09e667u; st[1]=0xbb67ae85u; st[2]=0x3c6ef372u; st[3]=0xa54ff53au;
  st[4]=0x510e527fu; st[5]=0x9b05688cu; st[6]=0x1f83d9abu; st[7]=0x5be0cd19u;
}

static inline u32 Round(
    u32 a, u32 b, u32 c, u32 *d,
    u32 e, u32 f, u32 g, u32 *h,
    u32 x, u32 K,
    u32 bc_prev
){
    u32 ab_and = a & b;

    u32 t1 = *h + BSIG1(e) + CH(e,f,g) + K + x;
    u32 t2 = BSIG0(a) + (ab_and ^ (a & c) ^ bc_prev);

    *d += t1;
    *h  = t1 + t2;

    return ab_and;
}

static inline void sha256_transform(u32 H[8], const u32 block[16]) {
    u32 w[64];

    #pragma unroll
    for (int i = 0; i < 16; i++) w[i] = block[i];
    #pragma unroll
    for (int i = 16; i < 64; i++) w[i] = SSIG1(w[i-2]) + w[i-7] + SSIG0(w[i-15]) + w[i-16];

    u32 a0 = H[0], a1 = H[1], a2 = H[2], a3 = H[3];
    u32 a4 = H[4], a5 = H[5], a6 = H[6], a7 = H[7];

    u32 bc = a1 & a2;

#define R0(a,b,c,d,e,f,g,h,i) bc = Round((a),(b),(c),&(d),(e),(f),(g),&(h), block[(i)], K256[(i)], bc)
#define Rw(a,b,c,d,e,f,g,h,i) bc = Round((a),(b),(c),&(d),(e),(f),(g),&(h), w[(i)],     K256[(i)], bc)

    R0(a0,a1,a2,a3,a4,a5,a6,a7, 0);
    R0(a7,a0,a1,a2,a3,a4,a5,a6, 1);
    R0(a6,a7,a0,a1,a2,a3,a4,a5, 2);
    R0(a5,a6,a7,a0,a1,a2,a3,a4, 3);
    R0(a4,a5,a6,a7,a0,a1,a2,a3, 4);
    R0(a3,a4,a5,a6,a7,a0,a1,a2, 5);
    R0(a2,a3,a4,a5,a6,a7,a0,a1, 6);
    R0(a1,a2,a3,a4,a5,a6,a7,a0, 7);

    R0(a0,a1,a2,a3,a4,a5,a6,a7, 8);
    R0(a7,a0,a1,a2,a3,a4,a5,a6, 9);
    bc = Round(a6,a7,a0,&a1, a2,a3,a4,&a5, w[10], K256[10], bc);
    R0(a5,a6,a7,a0,a1,a2,a3,a4,11);
    R0(a4,a5,a6,a7,a0,a1,a2,a3,12);
    R0(a3,a4,a5,a6,a7,a0,a1,a2,13);
    R0(a2,a3,a4,a5,a6,a7,a0,a1,14);
    R0(a1,a2,a3,a4,a5,a6,a7,a0,15);

    Rw(a0,a1,a2,a3,a4,a5,a6,a7,16);
    Rw(a7,a0,a1,a2,a3,a4,a5,a6,17);
    Rw(a6,a7,a0,a1,a2,a3,a4,a5,18);
    Rw(a5,a6,a7,a0,a1,a2,a3,a4,19);
    Rw(a4,a5,a6,a7,a0,a1,a2,a3,20);
    Rw(a3,a4,a5,a6,a7,a0,a1,a2,21);
    Rw(a2,a3,a4,a5,a6,a7,a0,a1,22);
    Rw(a1,a2,a3,a4,a5,a6,a7,a0,23);

    Rw(a0,a1,a2,a3,a4,a5,a6,a7,24);
    Rw(a7,a0,a1,a2,a3,a4,a5,a6,25);
    Rw(a6,a7,a0,a1,a2,a3,a4,a5,26);
    Rw(a5,a6,a7,a0,a1,a2,a3,a4,27);
    Rw(a4,a5,a6,a7,a0,a1,a2,a3,28);
    Rw(a3,a4,a5,a6,a7,a0,a1,a2,29);
    Rw(a2,a3,a4,a5,a6,a7,a0,a1,30);
    Rw(a1,a2,a3,a4,a5,a6,a7,a0,31);

    Rw(a0,a1,a2,a3,a4,a5,a6,a7,32);
    Rw(a7,a0,a1,a2,a3,a4,a5,a6,33);
    Rw(a6,a7,a0,a1,a2,a3,a4,a5,34);
    Rw(a5,a6,a7,a0,a1,a2,a3,a4,35);
    Rw(a4,a5,a6,a7,a0,a1,a2,a3,36);
    Rw(a3,a4,a5,a6,a7,a0,a1,a2,37);
    Rw(a2,a3,a4,a5,a6,a7,a0,a1,38);
    Rw(a1,a2,a3,a4,a5,a6,a7,a0,39);

    Rw(a0,a1,a2,a3,a4,a5,a6,a7,40);
    Rw(a7,a0,a1,a2,a3,a4,a5,a6,41);
    Rw(a6,a7,a0,a1,a2,a3,a4,a5,42);
    Rw(a5,a6,a7,a0,a1,a2,a3,a4,43);
    Rw(a4,a5,a6,a7,a0,a1,a2,a3,44);
    Rw(a3,a4,a5,a6,a7,a0,a1,a2,45);
    Rw(a2,a3,a4,a5,a6,a7,a0,a1,46);
    Rw(a1,a2,a3,a4,a5,a6,a7,a0,47);

    Rw(a0,a1,a2,a3,a4,a5,a6,a7,48);
    Rw(a7,a0,a1,a2,a3,a4,a5,a6,49);
    Rw(a6,a7,a0,a1,a2,a3,a4,a5,50);
    Rw(a5,a6,a7,a0,a1,a2,a3,a4,51);
    Rw(a4,a5,a6,a7,a0,a1,a2,a3,52);
    Rw(a3,a4,a5,a6,a7,a0,a1,a2,53);
    Rw(a2,a3,a4,a5,a6,a7,a0,a1,54);
    Rw(a1,a2,a3,a4,a5,a6,a7,a0,55);

    Rw(a0,a1,a2,a3,a4,a5,a6,a7,56);
    Rw(a7,a0,a1,a2,a3,a4,a5,a6,57);
    Rw(a6,a7,a0,a1,a2,a3,a4,a5,58);
    Rw(a5,a6,a7,a0,a1,a2,a3,a4,59);
    Rw(a4,a5,a6,a7,a0,a1,a2,a3,60);
    Rw(a3,a4,a5,a6,a7,a0,a1,a2,61);
    Rw(a2,a3,a4,a5,a6,a7,a0,a1,62);
    Rw(a1,a2,a3,a4,a5,a6,a7,a0,63);

#undef R0
#undef Rw

    H[0] += a0; H[1] += a1; H[2] += a2; H[3] += a3;
    H[4] += a4; H[5] += a5; H[6] += a6; H[7] += a7;
}

static inline u32 build_after_prefix64(u32 b1[16], u32 b2[16], __global const u32 *msg, u32 msg_len) {
  zero_words(b1, 16u);
  zero_words(b2, 16u);

  u32 nwords = (msg_len + 3u) >> 2;
  for (u32 i = 0u; i < nwords; i++) {
    u32 v = msg[i];
    // You need remove this branches before run in brute force mode
    if (i < 16u) b1[i] = v;
    else         b2[i - 16u] = v;
  }

  u32 rem = (msg_len & 3u);
  if (rem) {
    u32 idx = (msg_len >> 2);
    if (idx < 16u) b1[idx] &= mask_keep_be_bytes(rem);
    else           b2[idx - 16u] &= mask_keep_be_bytes(rem);
  }

  if (msg_len < 64u) set_be_byte(b1, msg_len, 0x80u);
  else               set_be_byte(b2, msg_len - 64u, 0x80u);

  u64 bitlen = ((u64)(64u + msg_len)) * 8ull;

  u32 mod64 = (msg_len & 63u);
  u32 blocks_for_msg = (msg_len + 63u) >> 6;
  u32 extra = (mod64 <= 55u) ? 0u : 1u;
  u32 rem_blocks = blocks_for_msg + extra;

  if (rem_blocks == 1u) {
    b1[14] = (u32)(bitlen >> 32);
    b1[15] = (u32)(bitlen);
  } else {
    b2[14] = (u32)(bitlen >> 32);
    b2[15] = (u32)(bitlen);
  }

  return rem_blocks;
}

static inline void sha256_copy8(u32 dst[8], const u32 src[8]) {
  #pragma unroll
  for (int i = 0; i < 8; i++) dst[i] = src[i];
}

static inline void sha256_from_prestate_msg(const u32 pre[8], __global const u32 *msg, u32 msg_len, u32 out8[8]) {
  u32 st[8];
  sha256_copy8(st, pre);

  u32 b1[16], b2[16];
  u32 nb = build_after_prefix64(b1, b2, msg, msg_len);

  sha256_transform(st, b1);
  if (nb == 2u) sha256_transform(st, b2);

  #pragma unroll
  for (int i = 0; i < 8; i++) out8[i] = st[i];
}

static inline void sha256_from_prestate_msg32(const u32 pre[8], const u32 msg8[8], u32 out8[8]) {
  u32 st[8];
  sha256_copy8(st, pre);

  u32 blk[16];
  #pragma unroll
  for (int i = 0; i < 8; i++) blk[i] = msg8[i];

  blk[8]  = 0x80000000u;
  blk[9]  = 0u;
  blk[10] = 0u;
  blk[11] = 0u;
  blk[12] = 0u;
  blk[13] = 0u;
  blk[14] = 0u;
  blk[15] = 768u;

  sha256_transform(st, blk);

  #pragma unroll
  for (int i = 0; i < 8; i++) out8[i] = st[i];
}

static inline void hmac_prepare_prestate(const u32 k0_16[16], u32 inner_pre[8], u32 outer_pre[8]) {
  u32 ipad[16], opad[16];

  #pragma unroll
  for (int i = 0; i < 16; i++) {
    ipad[i] = k0_16[i] ^ 0x36363636u;
    opad[i] = k0_16[i] ^ 0x5c5c5c5cu;
  }

  sha256_init(inner_pre);
  sha256_transform(inner_pre, ipad);

  sha256_init(outer_pre);
  sha256_transform(outer_pre, opad);
}

static inline void hmac_sha256_prepared_var(
  const u32 inner_pre[8], const u32 outer_pre[8],
  __global const u32 *msg, u32 msg_len, u32 out8[8]
) {
  u32 inner8[8];
  sha256_from_prestate_msg(inner_pre, msg, msg_len, inner8);
  sha256_from_prestate_msg32(outer_pre, inner8, out8);
}

static inline void hmac_sha256_prepared_32(
  const u32 inner_pre[8], const u32 outer_pre[8],
  const u32 msg8[8], u32 out8[8]
) {
  u32 inner8[8];
  sha256_from_prestate_msg32(inner_pre, msg8, inner8);
  sha256_from_prestate_msg32(outer_pre, inner8, out8);
}

static inline void pbkdf2_hmac_sha256_block_u32(
  const u32 pw16[16], u32 pwlen_bytes,
  __global const u32 *msg, u32 msglen_bytes,
  u32 iterations,
  u32 out8[8]
){
  u32 k0[16];

  #pragma unroll 16
  for (int i = 0; i < 16; i++) k0[i] = pw16[i];

  trim_be_words(k0, pwlen_bytes, 16u);

  u32 inner_pre[8], outer_pre[8];
  hmac_prepare_prestate(k0, inner_pre, outer_pre);

  u32 u8buf[8];
  hmac_sha256_prepared_var(inner_pre, outer_pre, msg, msglen_bytes, u8buf);

  #pragma unroll
  for (int i = 0; i < 8; i++) out8[i] = u8buf[i];
  // You need remove this branches before run in brute force mode
  if (iterations == 0u) iterations = 1u;

  for (u32 it = 1u; it < iterations; it++) {
    hmac_sha256_prepared_32(inner_pre, outer_pre, u8buf, u8buf);
    #pragma unroll
    for (int i = 0; i < 8; i++) out8[i] ^= u8buf[i];
  }
}

static inline void store_digest_nbytes_u32(__global u32 *dst, const u32 dig8[8], u32 nbytes) {
  // You need remove this branches before run in brute force mode
  if (nbytes >= 32u) {
    #pragma unroll
    for (int i = 0; i < 8; i++) dst[i] = dig8[i];
    return;
  }

  u32 fullw = (nbytes >> 2);
  u32 rem   = (nbytes & 3u);

  for (u32 i = 0u; i < fullw; i++) dst[i] = dig8[i];
  if (rem) dst[fullw] = dig8[fullw] & mask_keep_be_bytes(rem);
}

__kernel void pbkdf2_hmac_sha256_u32(
  __global const u32 *pw_buf,
  __global const u16 *pw_len,
  const u32 pw_stride_words,
  __global const u32 *msg_buf,
  const u32 msg_len,
  const u32 iterations,
  const u32 dkLen,
  __global u32 *out,
  const u32 out_stride_words
){
  const u32 gid = (u32)get_global_id(0);

  u32 pw_bytes = (u32)pw_len[gid];

  u32 pw16[16];
  u64 pw_off = (u64)gid * (u64)pw_stride_words;

  #pragma unroll
  for (int i = 0; i < 16; i++) pw16[i] = pw_buf[pw_off + (u64)i];

  __global u32 *dst = out + ((u64)gid * (u64)out_stride_words);

  u32 dk8[8];
  pbkdf2_hmac_sha256_block_u32(pw16, pw_bytes, msg_buf, msg_len, iterations, dk8);

  store_digest_nbytes_u32(dst, dk8, dkLen);
}
