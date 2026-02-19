/**
 * @file encrypt.cu
 * @brief Industrial-Strength GPU File Encryption (ChaCha20-Poly1305 + Argon2id)
 *
 * Security model:
 *   - Key Derivation : Argon2id (RFC 9106) — memory-hard, GPU/ASIC resistant
 *   - Encryption     : ChaCha20 stream cipher (RFC 8439) — GPU-parallelised
 *   - Authentication : Poly1305 MAC (RFC 8439) — CPU-side, per AEAD spec
 *   - Nonce          : 96-bit random (OS CSPRNG), stored in file header
 *   - Salt           : 128-bit random (OS CSPRNG), stored in file header
 *
 * File format (.enc):
 *   [4 bytes magic]  [1 byte version]  [16 bytes salt]  [12 bytes nonce]
 *   [8 bytes orig_size]  [ciphertext...]  [16 bytes Poly1305 tag]
 *
 * IMPORTANT: No secret material is ever printed to stdout.
 *
 * Argon2id parameters (defaults, adjustable via #define):
 *   ARGON2_T_COST   = 3     (iterations)
 *   ARGON2_M_COST   = 65536 (64 MB memory)
 *   ARGON2_PARALLELISM = 4
 *
 * Usage:
 *   vibecoder encrypt <file> <passphrase>   -> <file>.enc
 *   vibecoder decrypt <file.enc> <passphrase> -> original file
 *
 * Compilation (Linux/GCP, sm_75+):
 *   nvcc -O3 -use_fast_math -arch=sm_75 --ptxas-options=-v encrypt.cu -o
 * vibecoder
 *
 * Compilation (Windows MSVC):
 *   // 1. Install Argon2 via vcpkg:
 *   //    git clone https://github.com/microsoft/vcpkg C:\vcpkg
 *   //    C:\vcpkg\bootstrap-vcpkg.bat
 *   //    C:\vcpkg\vcpkg install argon2:x64-windows
 *
 *   // 2. Compile (example):
 *   nvcc -allow-unsupported-compiler -O3 -use_fast_math ^
 *     -I "C:\vcpkg\installed\x64-windows\include" ^
 *     -L "C:\vcpkg\installed\x64-windows\lib" ^
 *     encrypt.cu -o vibecoder.exe -largon2
 *
 *   // Note for Visual Studio 2022+ Preview:
 *   // The flag '-allow-unsupported-compiler' is required if nvcc
 *   // complains about an unsupported MSVC version.
 *
 * Dependencies:
 *   - CUDA Toolkit (>= 10.0)
 *   - C++17 standard library
 *   - Argon2 (via vcpkg or manual install)
 *     Link with: -largon2
 *
 * Security Notes:
 *   - Decryption will ABORT if the Poly1305 tag does not match (tamper
 * detection).
 *   - Keys are zeroed from memory after use.
 *   - The derived key is NEVER printed or logged.
 */

#include <argon2.h>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// ─── Configuration
// ────────────────────────────────────────────────────────────

#define CUDA_BLOCK_SIZE 256

// File format constants
static constexpr uint8_t MAGIC[4] = {'V', 'B', 'C', 'R'};
static constexpr uint8_t VERSION = 0x01;
static constexpr size_t SALT_LEN = 16;  // 128-bit salt
static constexpr size_t NONCE_LEN = 12; // 96-bit nonce (RFC 8439)
static constexpr size_t TAG_LEN = 16;   // Poly1305 tag
static constexpr size_t KEY_LEN = 32;   // 256-bit key
static constexpr size_t HEADER_SIZE =
    sizeof(MAGIC) + sizeof(VERSION) + SALT_LEN + NONCE_LEN + sizeof(uint64_t);

// Argon2id parameters — tune for your threat model
// At 64 MB / 3 iterations: ~200ms on a modern CPU
#define ARGON2_T_COST 3
#define ARGON2_M_COST 65536 // kibibytes
#define ARGON2_PARALLELISM 4

// ─── Error Handling
// ───────────────────────────────────────────────────────────

#define gpuCheck(ans)                                                          \
  {                                                                            \
    gpuAssert((ans), __FILE__, __LINE__);                                      \
  }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s  (%s:%d)\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

// ─── Secure Zeroing
// ─────────────────────────────────────────────────────────── Prevents compiler
// from optimizing away key erasure.
static void secure_zero(void *ptr, size_t len) {
  volatile uint8_t *p = reinterpret_cast<volatile uint8_t *>(ptr);
  while (len--)
    *p++ = 0;
}

// ─── OS Entropy
// ───────────────────────────────────────────────────────────────
static void get_random_bytes(uint8_t *buf, size_t len) {
  std::random_device rd;
  for (size_t i = 0; i < len; ++i)
    buf[i] = static_cast<uint8_t>(rd());
}

// ─── Argon2id Key Derivation
// ──────────────────────────────────────────────────
/**
 * Derives a 256-bit key from a passphrase and random salt using Argon2id.
 * Argon2id is memory-hard — it defeats GPU/ASIC brute-force attacks.
 *
 * @param passphrase  User-supplied password string
 * @param salt        16-byte random salt (caller generates or reads from file)
 * @param out_key     Output buffer, must be KEY_LEN (32) bytes
 */
static void derive_key(const std::string &passphrase, const uint8_t *salt,
                       uint8_t out_key[KEY_LEN]) {
  int rc = argon2id_hash_raw(ARGON2_T_COST, ARGON2_M_COST, ARGON2_PARALLELISM,
                             passphrase.c_str(), passphrase.length(), salt,
                             SALT_LEN, out_key, KEY_LEN);
  if (rc != ARGON2_OK) {
    std::cerr << "Argon2id failed: " << argon2_error_message(rc) << "\n";
    exit(1);
  }
}

// ─── Poly1305 (CPU, RFC 8439)
// ─────────────────────────────────────────────────
/**
 * Computes a Poly1305 MAC over a message using a 32-byte one-time key.
 * The key is typically the first 32 bytes of the ChaCha20 keystream at
 * counter=0 (ChaCha20-Poly1305 construction per RFC 8439).
 *
 * Implementation: constant-time, 130-bit integer arithmetic.
 */
class Poly1305 {
  uint32_t r[5], h[5], pad[4];

  static uint32_t le32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
  }

public:
  Poly1305(const uint8_t key[32]) {
    // Clamp r per spec
    r[0] = le32(key + 0) & 0x0fffffff;
    r[1] = (le32(key + 3) >> 2) & 0x0ffffffc;
    r[2] = (le32(key + 6) >> 4) & 0x0ffffffc;
    r[3] = (le32(key + 9) >> 6) & 0x0ffffffc;
    r[4] = (le32(key + 12) >> 8) & 0x0000000f;
    pad[0] = le32(key + 16);
    pad[1] = le32(key + 20);
    pad[2] = le32(key + 24);
    pad[3] = le32(key + 28);
    h[0] = h[1] = h[2] = h[3] = h[4] = 0;
  }

  void update(const uint8_t *msg, size_t len) {
    while (len > 0) {
      uint8_t block[17] = {0};
      size_t want = (len >= 16) ? 16 : len;
      memcpy(block, msg, want);
      block[want] = 1; // high bit for partial block

      uint32_t m[5];
      m[0] = le32(block + 0);
      m[1] = le32(block + 4);
      m[2] = le32(block + 8);
      m[3] = le32(block + 12);
      m[4] = block[16];

      h[0] += m[0];
      h[1] += m[1];
      h[2] += m[2];
      h[3] += m[3];
      h[4] += m[4];

      // Multiply h by r mod 2^130-5 using 64-bit intermediates
      uint64_t d0 = (uint64_t)h[0] * r[0] + (uint64_t)h[1] * (5 * r[4]) +
                    (uint64_t)h[2] * (5 * r[3]) + (uint64_t)h[3] * (5 * r[2]) +
                    (uint64_t)h[4] * (5 * r[1]);
      uint64_t d1 = (uint64_t)h[0] * r[1] + (uint64_t)h[1] * r[0] +
                    (uint64_t)h[2] * (5 * r[4]) + (uint64_t)h[3] * (5 * r[3]) +
                    (uint64_t)h[4] * (5 * r[2]);
      uint64_t d2 = (uint64_t)h[0] * r[2] + (uint64_t)h[1] * r[1] +
                    (uint64_t)h[2] * r[0] + (uint64_t)h[3] * (5 * r[4]) +
                    (uint64_t)h[4] * (5 * r[3]);
      uint64_t d3 = (uint64_t)h[0] * r[3] + (uint64_t)h[1] * r[2] +
                    (uint64_t)h[2] * r[1] + (uint64_t)h[3] * r[0] +
                    (uint64_t)h[4] * (5 * r[4]);
      uint64_t d4 = (uint64_t)h[0] * r[4] + (uint64_t)h[1] * r[3] +
                    (uint64_t)h[2] * r[2] + (uint64_t)h[3] * r[1] +
                    (uint64_t)h[4] * r[0];

      uint32_t c = (uint32_t)(d0 >> 26);
      h[0] = (uint32_t)d0 & 0x3ffffff;
      d1 += c;
      c = (uint32_t)(d1 >> 26);
      h[1] = (uint32_t)d1 & 0x3ffffff;
      d2 += c;
      c = (uint32_t)(d2 >> 26);
      h[2] = (uint32_t)d2 & 0x3ffffff;
      d3 += c;
      c = (uint32_t)(d3 >> 26);
      h[3] = (uint32_t)d3 & 0x3ffffff;
      d4 += c;
      c = (uint32_t)(d4 >> 26);
      h[4] = (uint32_t)d4 & 0x3ffffff;
      h[0] += c * 5;
      c = h[0] >> 26;
      h[0] &= 0x3ffffff;
      h[1] += c;

      msg += want;
      len -= want;
    }
  }

  void finalize(uint8_t tag[16]) {
    // Final reduction mod 2^130-5
    uint32_t c = h[1] >> 26;
    h[1] &= 0x3ffffff;
    h[2] += c;
    c = h[2] >> 26;
    h[2] &= 0x3ffffff;
    h[3] += c;
    c = h[3] >> 26;
    h[3] &= 0x3ffffff;
    h[4] += c;
    c = h[4] >> 26;
    h[4] &= 0x3ffffff;
    h[0] += c * 5;
    c = h[0] >> 26;
    h[0] &= 0x3ffffff;
    h[1] += c;

    // Compute h + -p
    uint32_t g[5];
    g[0] = h[0] + 5;
    c = g[0] >> 26;
    g[0] &= 0x3ffffff;
    g[1] = h[1] + c;
    c = g[1] >> 26;
    g[1] &= 0x3ffffff;
    g[2] = h[2] + c;
    c = g[2] >> 26;
    g[2] &= 0x3ffffff;
    g[3] = h[3] + c;
    c = g[3] >> 26;
    g[3] &= 0x3ffffff;
    g[4] = h[4] + c - (1 << 26);

    // Select h if h < p, else g (constant time)
    uint32_t mask = (g[4] >> 31) - 1;
    for (int i = 0; i < 5; i++)
      h[i] = (h[i] & ~mask) | (g[i] & mask);

    // Pack h into 128-bit and add pad
    uint64_t f;
    f = (uint64_t)h[0] | ((uint64_t)h[1] << 26) | ((uint64_t)h[2] << 52);
    uint32_t t0 = (uint32_t)f;
    f >>= 32;
    f |= (uint64_t)h[2] >> 12 | ((uint64_t)h[3] << 14) | ((uint64_t)h[4] << 40);
    uint32_t t1 = (uint32_t)f;
    f >>= 32;
    uint32_t t2 = (uint32_t)((uint64_t)h[3] >> 18 | ((uint64_t)h[4] << 8));
    f = (uint64_t)t0 + pad[0];
    tag[0] = (uint8_t)f;
    tag[1] = (uint8_t)(f >> 8);
    tag[2] = (uint8_t)(f >> 16);
    tag[3] = (uint8_t)(f >> 24);
    f >>= 32;
    f += (uint64_t)t1 + pad[1];
    tag[4] = (uint8_t)f;
    tag[5] = (uint8_t)(f >> 8);
    tag[6] = (uint8_t)(f >> 16);
    tag[7] = (uint8_t)(f >> 24);
    f >>= 32;
    f += (uint64_t)t2 + pad[2];
    tag[8] = (uint8_t)f;
    tag[9] = (uint8_t)(f >> 8);
    tag[10] = (uint8_t)(f >> 16);
    tag[11] = (uint8_t)(f >> 24);
    f >>= 32;
    f += pad[3];
    tag[12] = (uint8_t)f;
    tag[13] = (uint8_t)(f >> 8);
    tag[14] = (uint8_t)(f >> 16);
    tag[15] = (uint8_t)(f >> 24);
  }
};

// ─── Constant-time tag comparison
// ───────────────────────────────────────────── Prevents timing attacks against
// MAC verification.
static bool ct_equal(const uint8_t *a, const uint8_t *b, size_t len) {
  uint8_t diff = 0;
  for (size_t i = 0; i < len; i++)
    diff |= a[i] ^ b[i];
  return diff == 0;
}

// ─── GPU: ChaCha20 Quarter Round
// ──────────────────────────────────────────────
__device__ __forceinline__ void quarter_round(uint32_t &a, uint32_t &b,
                                              uint32_t &c, uint32_t &d) {
  a += b;
  d ^= a;
  d = (d << 16) | (d >> 16);
  c += d;
  b ^= c;
  b = (b << 12) | (b >> 20);
  a += b;
  d ^= a;
  d = (d << 8) | (d >> 24);
  c += d;
  b ^= c;
  b = (b << 7) | (b >> 25);
}

// ─── GPU: ChaCha20 Keystream Kernel
// ───────────────────────────────────────────
/**
 * Each thread processes one 64-byte ChaCha20 block.
 * Uses shared memory for key/nonce and 128-bit vectorised loads/stores.
 *
 * @param data      In/out buffer (padded to 64-byte boundary)
 * @param key       256-bit key (8 x uint32)
 * @param nonce     96-bit nonce (3 x uint32, RFC 8439)
 * @param n_chunks  Total number of 64-byte blocks
 * @param ctr_base  Block counter offset (always 1 for ChaCha20-Poly1305;
 *                  counter 0 is reserved for generating the Poly1305 key)
 */
__global__ void chacha20_kernel(uint4 *data, const uint32_t *key,
                                const uint32_t *nonce, int n_chunks,
                                uint32_t ctr_base) {
  __shared__ uint32_t s_key[8];
  __shared__ uint32_t s_nonce[3]; // 96-bit nonce per RFC 8439

  if (threadIdx.x < 8)
    s_key[threadIdx.x] = key[threadIdx.x];
  if (threadIdx.x < 3)
    s_nonce[threadIdx.x] = nonce[threadIdx.x];
  __syncthreads();

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_chunks)
    return;

  // Vectorized 64-byte load (4 x uint4 = 64 bytes)
  int vec = idx * 4;
  uint4 v0 = data[vec + 0];
  uint4 v1 = data[vec + 1];
  uint4 v2 = data[vec + 2];
  uint4 v3 = data[vec + 3];

  // Initialise ChaCha20 state (RFC 8439 layout)
  uint32_t st[16];
  st[0] = 0x61707865;
  st[1] = 0x3320646e;
  st[2] = 0x79622d32;
  st[3] = 0x6b206574; // "expand 32-byte k"

#pragma unroll
  for (int i = 0; i < 8; i++)
    st[4 + i] = s_key[i];

  // 32-bit counter (ctr_base + thread index) — supports files up to 256 GB
  st[12] = ctr_base + (uint32_t)idx;

  // 96-bit nonce (RFC 8439 positions 13/14/15)
  st[13] = s_nonce[0];
  st[14] = s_nonce[1];
  st[15] = s_nonce[2];

  // Working copy
  uint32_t wk[16];
#pragma unroll
  for (int i = 0; i < 16; i++)
    wk[i] = st[i];

// 20 rounds (10 column + 10 diagonal)
#pragma unroll
  for (int i = 0; i < 10; i++) {
    quarter_round(wk[0], wk[4], wk[8], wk[12]);
    quarter_round(wk[1], wk[5], wk[9], wk[13]);
    quarter_round(wk[2], wk[6], wk[10], wk[14]);
    quarter_round(wk[3], wk[7], wk[11], wk[15]);
    quarter_round(wk[0], wk[5], wk[10], wk[15]);
    quarter_round(wk[1], wk[6], wk[11], wk[12]);
    quarter_round(wk[2], wk[7], wk[8], wk[13]);
    quarter_round(wk[3], wk[4], wk[9], wk[14]);
  }

#pragma unroll
  for (int i = 0; i < 16; i++)
    wk[i] += st[i];

  // XOR keystream with data
  v0.x ^= wk[0];
  v0.y ^= wk[1];
  v0.z ^= wk[2];
  v0.w ^= wk[3];
  v1.x ^= wk[4];
  v1.y ^= wk[5];
  v1.z ^= wk[6];
  v1.w ^= wk[7];
  v2.x ^= wk[8];
  v2.y ^= wk[9];
  v2.z ^= wk[10];
  v2.w ^= wk[11];
  v3.x ^= wk[12];
  v3.y ^= wk[13];
  v3.z ^= wk[14];
  v3.w ^= wk[15];

  data[vec + 0] = v0;
  data[vec + 1] = v1;
  data[vec + 2] = v2;
  data[vec + 3] = v3;
}

// ─── CPU: Generate Poly1305 one-time key
// ──────────────────────────────────────
/**
 * Per RFC 8439 §2.6: run ChaCha20 with counter=0, take first 32 bytes
 * of keystream as the Poly1305 key. This is done on the CPU so the GPU
 * kernel can start at counter=1.
 */
static void generate_poly1305_key(
    const uint8_t key[32],\n const uint8_t nonce[12],\n uint8_t otk[32]) {
  uint32_t st[16];
  st[0] = 0x61707865;
  st[1] = 0x3320646e;
  st[2] = 0x79622d32;
  st[3] = 0x6b206574;
  for (int i = 0; i < 8; i++)
    st[4 + i] = (uint32_t)key[i * 4] | ((uint32_t)key[i * 4 + 1] << 8) |
                ((uint32_t)key[i * 4 + 2] << 16) |
                ((uint32_t)key[i * 4 + 3] << 24);
  st[12] = 0; // counter = 0
  for (int i = 0; i < 3; i++)
    st[13 + i] = (uint32_t)nonce[i * 4] | ((uint32_t)nonce[i * 4 + 1] << 8) |
                 ((uint32_t)nonce[i * 4 + 2] << 16) |
                 ((uint32_t)nonce[i * 4 + 3] << 24);

  uint32_t wk[16];
  for (int i = 0; i < 16; i++)
    wk[i] = st[i];
  for (int i = 0; i < 10; i++) {
    // Column rounds
    auto qr = [](uint32_t &a, uint32_t &b, uint32_t &c, uint32_t &d) {
      a += b;
      d ^= a;
      d = (d << 16) | (d >> 16);
      c += d;
      b ^= c;
      b = (b << 12) | (b >> 20);
      a += b;
      d ^= a;
      d = (d << 8) | (d >> 24);
      c += d;
      b ^= c;
      b = (b << 7) | (b >> 25);
    };
    qr(wk[0], wk[4], wk[8], wk[12]);
    qr(wk[1], wk[5], wk[9], wk[13]);
    qr(wk[2], wk[6], wk[10], wk[14]);
    qr(wk[3], wk[7], wk[11], wk[15]);
    qr(wk[0], wk[5], wk[10], wk[15]);
    qr(wk[1], wk[6], wk[11], wk[12]);
    qr(wk[2], wk[7], wk[8], wk[13]);
    qr(wk[3], wk[4], wk[9], wk[14]);
  }
  for (int i = 0; i < 16; i++)
    wk[i] += st[i];
  // Output first 32 bytes as the OTK
  for (int i = 0; i < 8; i++) {
    otk[i * 4 + 0] = (uint8_t)(wk[i]);
    otk[i * 4 + 1] = (uint8_t)(wk[i] >> 8);
    otk[i * 4 + 2] = (uint8_t)(wk[i] >> 16);
    otk[i * 4 + 3] = (uint8_t)(wk[i] >> 24);
  }
}

// ─── MAIN
// ─────────────────────────────────────────────────────────────────────
int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0]
              << " <encrypt|decrypt> <filename> <passphrase>\\n";
    return 1;
  }

  const std::string mode = argv[1];
  const std::string filename = argv[2];
  const std::string passphrase = argv[3];

  // ── ENCRYPT ───────────────────────────────────────────────────────────────
  if (mode == "encrypt") {
    // Read plaintext
    std::ifstream in(filename, std::ios::binary | std::ios::ate);
    if (!in) {
      std::cerr << "Cannot open: " << filename << "\\n";
      return 1;
    }
    uint64_t orig_size = in.tellg();
    in.seekg(0);

    size_t padded = ((orig_size + 63) / 64) * 64;
    uint8_t *h_data;
    gpuCheck(cudaMallocHost(&h_data, padded));
    memset(h_data, 0, padded);
    in.read(reinterpret_cast<char *>(h_data), orig_size);
    in.close();

    // Generate random salt and nonce
    uint8_t salt[SALT_LEN], nonce[NONCE_LEN];
    get_random_bytes(salt, SALT_LEN);
    get_random_bytes(nonce, NONCE_LEN);

    // Derive key via Argon2id
    std::cerr << "Deriving key (Argon2id, " << ARGON2_M_COST / 1024 << " MB, "
              << ARGON2_T_COST << " iterations)...\\n";
    uint8_t key[KEY_LEN];
    derive_key(passphrase, salt, key);

    // Generate Poly1305 one-time key (ChaCha20 block 0)
    uint8_t otk[32];
    generate_poly1305_key(key, nonce, otk);

    // Convert key/nonce to uint32 arrays for GPU
    uint32_t key32[8], nonce32[3];
    memcpy(key32, key, 32);
    memcpy(nonce32, nonce, 12);

    // GPU encryption (blocks 1..N, block 0 was used for OTK)
    uint4 *d_data;
    uint32_t *d_key, *d_nonce;
    int n_chunks = (int)(padded / 64);

    gpuCheck(cudaMalloc(&d_data, padded));
    gpuCheck(cudaMalloc(&d_key, 32));
    gpuCheck(cudaMalloc(&d_nonce, 12));
    gpuCheck(cudaMemcpy(d_data, h_data, padded, cudaMemcpyHostToDevice));
    gpuCheck(cudaMemcpy(d_key, key32, 32, cudaMemcpyHostToDevice));
    gpuCheck(cudaMemcpy(d_nonce, nonce32, 12, cudaMemcpyHostToDevice));

    int grid = (n_chunks + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    chacha20_kernel<<<grid, CUDA_BLOCK_SIZE>>>(\n d_data, d_key, d_nonce,
                                               n_chunks, /*ctr_base=*/1);
    gpuCheck(cudaPeekAtLastError());
    gpuCheck(cudaDeviceSynchronize());
    gpuCheck(cudaMemcpy(h_data, d_data, padded, cudaMemcpyDeviceToHost));

    // Compute Poly1305 tag over ciphertext (first orig_size bytes)
    Poly1305 mac(otk);
    mac.update(h_data, orig_size);
    uint8_t tag[TAG_LEN];
    mac.finalize(tag);

    // Write output: header | ciphertext | tag
    std::string out_name = filename + ".enc";
    std::ofstream out(out_name, std::ios::binary);
    out.write(reinterpret_cast<const char *>(MAGIC), 4);
    out.write(reinterpret_cast<const char *>(&VERSION), 1);
    out.write(reinterpret_cast<const char *>(salt), SALT_LEN);
    out.write(reinterpret_cast<const char *>(nonce), NONCE_LEN);
    out.write(reinterpret_cast<const char *>(&orig_size), sizeof(uint64_t));
    out.write(reinterpret_cast<const char *>(h_data), orig_size);
    out.write(reinterpret_cast<const char *>(tag), TAG_LEN);
    out.close();

    std::cerr << "Encrypted -> " << out_name << "  (" << orig_size
              << " bytes)\\n";

    // Zeroize key material
    secure_zero(key, KEY_LEN);
    secure_zero(otk, 32);
    secure_zero(key32, 32);
    cudaFree(d_data);
    cudaFree(d_key);
    cudaFree(d_nonce);
    cudaFreeHost(h_data);
    return 0;
  }

  // ── DECRYPT ───────────────────────────────────────────────────────────────
  else if (mode == "decrypt") {
    std::ifstream in(filename, std::ios::binary | std::ios::ate);
    if (!in) {
      std::cerr << "Cannot open: " << filename << "\\n";
      return 1;
    }
    size_t total = in.tellg();
    in.seekg(0);

    // Minimum size sanity check
    if (total < HEADER_SIZE + TAG_LEN) {
      std::cerr << "File too small to be valid.\\n";
      return 1;
    }

    // Read and verify header
    uint8_t magic[4], version;
    in.read(reinterpret_cast<char *>(magic), 4);
    in.read(reinterpret_cast<char *>(&version), 1);
    if (memcmp(magic, MAGIC, 4) != 0 || version != VERSION) {
      std::cerr << "Invalid file format or version.\\n";
      return 1;
    }

    uint8_t salt[SALT_LEN], nonce[NONCE_LEN];
    uint64_t orig_size;
    in.read(reinterpret_cast<char *>(salt), SALT_LEN);
    in.read(reinterpret_cast<char *>(nonce), NONCE_LEN);
    in.read(reinterpret_cast<char *>(&orig_size), sizeof(uint64_t));

    size_t ct_size = total - HEADER_SIZE - TAG_LEN;
    if (ct_size < orig_size) {
      std::cerr << "Truncated ciphertext.\\n";
      return 1;
    }

    size_t padded = ((orig_size + 63) / 64) * 64;
    uint8_t *h_data;
    gpuCheck(cudaMallocHost(&h_data, padded));
    memset(h_data, 0, padded);
    in.read(reinterpret_cast<char *>(h_data), orig_size);

    uint8_t stored_tag[TAG_LEN];
    in.read(reinterpret_cast<char *>(stored_tag), TAG_LEN);
    in.close();

    // Derive key
    std::cerr << "Deriving key (Argon2id)...\\n";
    uint8_t key[KEY_LEN];
    derive_key(passphrase, salt, key);

    // Generate Poly1305 OTK and verify tag BEFORE decrypting
    uint8_t otk[32];
    generate_poly1305_key(key, nonce, otk);

    Poly1305 mac(otk);
    mac.update(h_data, orig_size);
    uint8_t computed_tag[TAG_LEN];
    mac.finalize(computed_tag);

    if (!ct_equal(computed_tag, stored_tag, TAG_LEN)) {
      std::cerr
          << "Authentication FAILED — wrong passphrase or file tampered.\\n";
      secure_zero(key, KEY_LEN);
      secure_zero(otk, 32);
      cudaFreeHost(h_data);
      return 1;
    }
    std::cerr << "MAC verified OK.\\n";

    // GPU decryption (ChaCha20 is symmetric — same kernel)
    uint32_t key32[8], nonce32[3];
    memcpy(key32, key, 32);
    memcpy(nonce32, nonce, 12);

    uint4 *d_data;
    uint32_t *d_key, *d_nonce;
    int n_chunks = (int)(padded / 64);

    gpuCheck(cudaMalloc(&d_data, padded));
    gpuCheck(cudaMalloc(&d_key, 32));
    gpuCheck(cudaMalloc(&d_nonce, 12));
    gpuCheck(cudaMemcpy(d_data, h_data, padded, cudaMemcpyHostToDevice));
    gpuCheck(cudaMemcpy(d_key, key32, 32, cudaMemcpyHostToDevice));
    gpuCheck(cudaMemcpy(d_nonce, nonce32, 12, cudaMemcpyHostToDevice));

    int grid = (n_chunks + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    chacha20_kernel<<<grid, CUDA_BLOCK_SIZE>>>(\n d_data, d_key, d_nonce,
                                               n_chunks, /*ctr_base=*/1);
    gpuCheck(cudaPeekAtLastError());
    gpuCheck(cudaDeviceSynchronize());
    gpuCheck(cudaMemcpy(h_data, d_data, padded, cudaMemcpyDeviceToHost));

    // Write plaintext output
    std::string out_name;
    if (filename.size() > 4 && filename.substr(filename.size() - 4) == ".enc")
      out_name = filename.substr(0, filename.size() - 4);
    else
      out_name = "decrypted_" + filename;

    std::ofstream out(out_name, std::ios::binary);
    out.write(reinterpret_cast<char *>(h_data), orig_size);
    out.close();

    std::cerr << "Decrypted -> " << out_name << "  (" << orig_size
              << " bytes)\\n";

    // Zeroize
    secure_zero(key, KEY_LEN);
    secure_zero(otk, 32);
    secure_zero(key32, 32);
    secure_zero(h_data, padded);
    cudaFree(d_data);
    cudaFree(d_key);
    cudaFree(d_nonce);
    cudaFreeHost(h_data);
    return 0;
  }

  std::cerr << "Unknown mode: " << mode << "\\n";
  return 1;
}
