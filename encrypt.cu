/**
 * @file encrypt.cu
 * @brief High-performance GPU File Encryption Tool (ChaCha20-Poly1305
 * equivalent performance, without Poly1305)
 *
 * Designed for NVIDIA GPUs using CUDA.
 * Implements:
 * 1. Key Derivation: SHA-256 (CPU-side)
 * 2. Encryption: ChaCha20 Stream Cipher (GPU-side, highly parallel)
 * 3. Entropy: OS-provided CSPRNG (std::random_device) for unique Nonces per
 * file.
 * 4. Optimization: Inline PTX Assembly for bitwise rotations.
 *
 * Usage:
 *   encryption.exe encrypt <file> <passphrase> -> Creates <file>.enc
 *   encryption.exe decrypt <file.enc> <passphrase> -> Restores original file
 *
 * Compilation:
 *   nvcc -allow-unsupported-compiler -ccbin "C:\\Program Files\\Microsoft
 * Visual
 * Studio\\18\\Community\\VC\\Tools\\MSVC\\14.50.35717\\bin\\Hostx64\\x64"
 * encrypt.cu -o encryption.exe (Ensure you adjust the MSVC path if using a
 * different version)
 */

#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// --- Configuration ---
#define CUDA_BLOCK_SIZE 256

// --- Error Handling ---
// Wraps CUDA calls to check for errors (e.g., OOM, Invalid Configuration)
#define gpuErrchk(ans)                                                         \
  {                                                                            \
    gpuAssert((ans), __FILE__, __LINE__);                                      \
  }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

// --- OS Entropy Helper ---
/**
 * @brief Generates cryptographically secure random bytes using the OS CSPRNG.
 *        Used for generating the 64-bit Nonce.
 */
bool get_os_random_bytes(uint8_t *buffer, size_t size) {
  try {
    // std::random_device is non-deterministic and maps to /dev/urandom (Linux)
    // or CryptGenRandom (Windows)
    std::random_device rd;
    for (size_t i = 0; i < size; ++i) {
      // rd() returns unsigned int; we take the lower 8 bits.
      buffer[i] = static_cast<uint8_t>(rd());
    }
    return true;
  } catch (...) {
    return false;
  }
}

// ==========================================
// HOST: SHA-256 Implementation
// ==========================================
/**
 * @class SHA256
 * @brief A bare-bones, self-contained SHA-256 implementation for Key
 * Derivation. Used to hash the user's passphrase into a 32-byte (256-bit)
 * encryption key.
 */
class SHA256 {
  uint32_t state[8];
  uint8_t buffer[64];
  uint64_t count;

  // Helper: Bitwise Rotation for SHA-256 compression function
  inline uint32_t jrotate(uint32_t x, uint32_t c) {
    return (x >> c) | (x << (32 - c));
  }

public:
  SHA256() { reset(); }

  void reset() {
    // Initial Hash Values (First 32 bits of the fractional parts of the square
    // roots of the first 8 primes)
    state[0] = 0x6a09e667;
    state[1] = 0xbb67ae85;
    state[2] = 0x3c6ef372;
    state[3] = 0xa54ff53a;
    state[4] = 0x510e527f;
    state[5] = 0x9b05688c;
    state[6] = 0x1f83d9ab;
    state[7] = 0x5be0cd19;
    count = 0;
  }

  void update(const uint8_t *data, size_t len) {
    for (size_t i = 0; i < len; ++i) {
      buffer[count % 64] = data[i];
      count++;
      if (count % 64 == 0)
        transform(buffer);
    }
  }

  void finalize(uint8_t hash[32]) {
    // Padding: Append '1' bit, then zeros, then length of message in bits.
    uint64_t bitlen = count * 8;
    buffer[count % 64] = 0x80;
    size_t current_len = count % 64 + 1;
    if (current_len > 56) {
      // If not enough space for length (8 bytes), pad with zeros, transform,
      // then pad again.
      memset(buffer + current_len, 0, 64 - current_len);
      transform(buffer);
      memset(buffer, 0, 56);
    } else {
      memset(buffer + current_len, 0, 56 - current_len);
    }

    // Append length (Big Endian)
    for (int i = 0; i < 8; ++i)
      buffer[63 - i] = (bitlen >> (i * 8)) & 0xFF;
    transform(buffer);

    // Output Hash (Big Endian)
    for (int i = 0; i < 8; ++i) {
      hash[i * 4] = (state[i] >> 24) & 0xFF;
      hash[i * 4 + 1] = (state[i] >> 16) & 0xFF;
      hash[i * 4 + 2] = (state[i] >> 8) & 0xFF;
      hash[i * 4 + 3] = state[i] & 0xFF;
    }
  }

private:
  void transform(const uint8_t *data) {
    // SHA-256 Compression Function
    uint32_t w[64], a, b, c, d, e, f, g, h;

    // Prepare Message Schedule
    for (int i = 0; i < 16; ++i)
      w[i] = (data[i * 4] << 24) | (data[i * 4 + 1] << 16) |
             (data[i * 4 + 2] << 8) | data[i * 4 + 3];
    for (int i = 16; i < 64; ++i) {
      uint32_t s0 =
          jrotate(w[i - 15], 7) ^ jrotate(w[i - 15], 18) ^ (w[i - 15] >> 3);
      uint32_t s1 =
          jrotate(w[i - 2], 17) ^ jrotate(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    // Initialize Working Variables
    a = state[0];
    b = state[1];
    c = state[2];
    d = state[3];
    e = state[4];
    f = state[5];
    g = state[6];
    h = state[7];

    // SHA-256 Constants
    const uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
        0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
        0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
        0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
        0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

    // Main Loop
    for (int i = 0; i < 64; ++i) {
      uint32_t S1 = jrotate(e, 6) ^ jrotate(e, 11) ^ jrotate(e, 25);
      uint32_t ch = (e & f) ^ (~e & g);
      uint32_t temp1 = h + S1 + ch + k[i] + w[i];
      uint32_t S0 = jrotate(a, 2) ^ jrotate(a, 13) ^ jrotate(a, 22);
      uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      uint32_t temp2 = S0 + maj;

      h = g;
      g = f;
      f = e;
      e = d + temp1;
      d = c;
      c = b;
      b = a;
      a = temp1 + temp2;
    }

    // Add computed values to state
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
  }
};

// ==========================================
// DEVICE: ChaCha20 Implementation w/ PTX ASM
// ==========================================

struct ChaChaKey {
  uint32_t k[8];     // 256-bit Key
  uint32_t nonce[2]; // 64-bit Nonce
};

/**
 * @brief Inline PTX Assembly for 32-bit Rotate Left.
 *        Using "shf.l.wrap.b32" (funnel shift) for single-instruction rotation.
 *
 * @param x Value to rotate
 * @param n Bits to rotate by
 * @return Rotated value
 */
__device__ __forceinline__ uint32_t rotl32(uint32_t x, uint32_t n) {
  uint32_t res;
  // PTX: shf.l.wrap.b32 dest, source, source, shift_amount
  // This performs (x << n) | (x >> (32-n)) in one hardware cycle.
  asm("shf.l.wrap.b32 %0, %1, %1, %2;" : "=r"(res) : "r"(x), "r"(n));
  return res;
}

// Quarter Round Macro for ChaCha20
#define QR(a, b, c, d)                                                         \
  a += b;                                                                      \
  d ^= a;                                                                      \
  d = rotl32(d, 16);                                                           \
  c += d;                                                                      \
  b ^= c;                                                                      \
  b = rotl32(b, 12);                                                           \
  a += b;                                                                      \
  d ^= a;                                                                      \
  d = rotl32(d, 8);                                                            \
  c += d;                                                                      \
  b ^= c;                                                                      \
  b = rotl32(b, 7);

/**
 * @brief Generates one 64-byte block of ChaCha20 keystream.
 *
 * @param key Struct containing Key and Nonce
 * @param counter The 64-bit block counter for CTR mode
 * @param keystream Output buffer (64 bytes)
 */
__device__ void chacha20_block(const ChaChaKey &key, uint64_t counter,
                               uint8_t keystream[64]) {
  uint32_t x[16];

  // 1. Initialize State with Constants ("expand 32-byte k")
  x[0] = 0x61707865;
  x[1] = 0x3320646e;
  x[2] = 0x79622d32;
  x[3] = 0x6b206574;

// 2. Load Key
#pragma unroll
  for (int i = 0; i < 8; i++)
    x[4 + i] = key.k[i];

  // 3. Load Counter (64-bit)
  x[12] = (uint32_t)(counter & 0xFFFFFFFF);
  x[13] = (uint32_t)(counter >> 32);

  // 4. Load Nonce (64-bit)
  x[14] = key.nonce[0];
  x[15] = key.nonce[1];

  // Copy state to allow addition at the end
  uint32_t orig[16];
#pragma unroll
  for (int i = 0; i < 16; i++)
    orig[i] = x[i];

  // 5. Run 20 Rounds (10 iterations of double-rounds)
  for (int i = 0; i < 10; i++) {
    // Odd round (Column rounds)
    QR(x[0], x[4], x[8], x[12]);
    QR(x[1], x[5], x[9], x[13]);
    QR(x[2], x[6], x[10], x[14]);
    QR(x[3], x[7], x[11], x[15]);

    // Even round (Diagonal rounds)
    QR(x[0], x[5], x[10], x[15]);
    QR(x[1], x[6], x[11], x[12]);
    QR(x[2], x[7], x[8], x[13]);
    QR(x[3], x[4], x[9], x[14]);
  }

// 6. Add original state to scrambled state
#pragma unroll
  for (int i = 0; i < 16; i++)
    x[i] += orig[i];

// 7. Serialize to Little Endian bytes
#pragma unroll
  for (int i = 0; i < 16; i++) {
    keystream[i * 4 + 0] = (x[i] >> 0) & 0xFF;
    keystream[i * 4 + 1] = (x[i] >> 8) & 0xFF;
    keystream[i * 4 + 2] = (x[i] >> 16) & 0xFF;
    keystream[i * 4 + 3] = (x[i] >> 24) & 0xFF;
  }
}

/**
 * @brief CUDA Kernel to encrypt/decrypt data using ChaCha20.
 *        Uses Grid-Stride Loop pattern to handle data of any size.
 */
__global__ void chacha20_kernel(uint8_t *data, size_t n, ChaChaKey key) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  // Calculate total number of 64-byte blocks
  size_t num_blocks = (n + 63) / 64;
  uint8_t keystream[64];

  // Grid-Stride Loop: Each thread processes multiple blocks if n > grid_size
  for (size_t blk_idx = idx; blk_idx < num_blocks; blk_idx += stride) {
    // Generate the keystream for this specific block index (Counter = blk_idx)
    chacha20_block(key, (uint64_t)blk_idx, keystream);

    // Output Range
    size_t start_byte = blk_idx * 64;
    size_t end_byte = start_byte + 64;
    if (end_byte > n)
      end_byte = n; // Handle partial last block

    // XOR plaintext with keystream to get ciphertext (or vice-versa)
    for (size_t i = 0; i < (end_byte - start_byte); ++i) {
      data[start_byte + i] ^= keystream[i];
    }
  }
}

// ==========================================
// MAIN
// ==========================================
int main(int argc, char *argv[]) {
  // 1. Argument Parsing
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0]
              << " <encrypt/decrypt> <filename> <passphrase>" << std::endl;
    return 1;
  }
  std::string mode = argv[1];
  std::string filename = argv[2];
  std::string passphrase = argv[3];

  // 2. Key Derivation (Passphrase -> 256-bit Key)
  uint8_t raw_key[32];
  SHA256 sha;
  sha.update((const uint8_t *)passphrase.c_str(), passphrase.length());
  sha.finalize(raw_key);

  std::cout << "Derived Key (Hex): ";
  for (int i = 0; i < 32; i++)
    std::cout << std::hex << std::setw(2) << std::setfill('0')
              << (int)raw_key[i];
  std::cout << std::dec << std::endl;

  // 3. Prepare Buffer & Nonce
  uint8_t nonce_bytes[8] = {0};
  uint8_t *h_data = nullptr; // Host Pinned Memory
  size_t n = 0;
  std::string out_filename;

  if (mode == "encrypt") {
    out_filename = filename + ".enc";

    // Generate Secure Random Nonce for this file
    if (!get_os_random_bytes(nonce_bytes, 8)) {
      std::cerr << "Error: Failed to get entropy.\n";
      return 1;
    }

    // Read File
    std::ifstream infile(filename, std::ios::binary | std::ios::ate);
    if (!infile) {
      std::cerr << "Error opening file.\n";
      return 1;
    }
    n = infile.tellg();
    infile.seekg(0, std::ios::beg);

    // Allocate Pinned Memory (Faster Transfer)
    gpuErrchk(cudaMallocHost(&h_data, n));
    if (!h_data) {
      std::cerr << "Pinned alloc failed.\n";
      return 1;
    }
    infile.read((char *)h_data, n);
    infile.close();

    // Write Header (Nonce) immediately to output file
    std::ofstream outfile(out_filename, std::ios::binary);
    outfile.write((char *)nonce_bytes, 8);
    outfile.close();

  } else if (mode == "decrypt") {
    // Determine Output Filename
    if (filename.length() > 4 &&
        filename.substr(filename.length() - 4) == ".enc")
      out_filename = filename.substr(0, filename.length() - 4);
    else
      out_filename = "decrypted_" + filename;

    // Open File
    std::ifstream infile(filename, std::ios::binary | std::ios::ate);
    if (!infile) {
      std::cerr << "Error opening file.\n";
      return 1;
    }
    size_t total_size = infile.tellg();

    // Validate File Size (Must contain at least 8 bytes for Nonce)
    if (total_size < 8) {
      std::cerr << "Error: File too small (missing header).\n";
      return 1;
    }

    n = total_size - 8;
    infile.seekg(0, std::ios::beg);

    // Read Nonce from Header
    infile.read((char *)nonce_bytes, 8);

    // Read Encrypted Payload
    gpuErrchk(cudaMallocHost(&h_data, n));
    infile.read((char *)h_data, n);
    infile.close();
  } else {
    std::cerr << "Invalid mode. Use 'encrypt' or 'decrypt'.\n";
    return 1;
  }

  std::cout << "Nonce (Hex): ";
  for (int i = 0; i < 8; i++)
    std::cout << std::hex << std::setw(2) << std::setfill('0')
              << (int)nonce_bytes[i];
  std::cout << std::dec << std::endl;
  std::cout << mode << "ing " << n << " bytes..." << std::endl;

  // 4. Setup GPU
  ChaChaKey host_key;
  memcpy(host_key.k, raw_key, 32);
  memcpy(host_key.nonce, nonce_bytes, 8);

  uint8_t *d_data;
  gpuErrchk(cudaMalloc(&d_data, n));
  gpuErrchk(cudaMemcpy(d_data, h_data, n, cudaMemcpyHostToDevice));

  // Calculate Grid Size
  int blockSize = 256;
  // Each thread processes at least one 64-byte block.
  // Calculate how many blocks the grid needs to cover.
  int gridSize = (int)((n + blockSize * 64 - 1) / (blockSize * 64));

  // Cap grid size to avoid launch overhead (Grid-Stride loop handles the rest)
  if (gridSize > 65535)
    gridSize = 65535;
  if (gridSize == 0)
    gridSize = 1;

  // 5. Encrypt/Decrypt on GPU
  chacha20_kernel<<<gridSize, blockSize>>>(d_data, n, host_key);

  // Error Check
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  // Retrieve Data
  gpuErrchk(cudaMemcpy(h_data, d_data, n, cudaMemcpyDeviceToHost));
  gpuErrchk(cudaFree(d_data));

  // 6. Append Result to Output File
  std::ofstream outfile(out_filename, std::ios::binary | std::ios::app);
  outfile.write((char *)h_data, n);
  outfile.close();

  // Cleanup
  gpuErrchk(cudaFreeHost(h_data));
  std::cout << "Success -> " << out_filename << std::endl;

  return 0;
}
