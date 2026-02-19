/**
 * @file encrypt.cu
 * @brief High-performance GPU File Encryption Tool (ChaCha20)
 *
 * Designed for NVIDIA GPUs using CUDA.
 * Implements:
 * 1. Key Derivation: SHA-256 (CPU-side)
 * 2. Encryption: ChaCha20 Stream Cipher (GPU-side, highly parallel)
 * 3. Entropy: OS-provided CSPRNG (std::random_device)
 * 4. Optimization:
 *    - Shared Memory for Key/Nonce
 *    - 128-bit Vectorized Loads/Stores (uint4)
 *    - Inline PTX Assembly for rotations (via device function)
 *
 * Usage:
 *   encryption.exe encrypt <file> <passphrase> -> Creates <file>.enc
 *   encryption.exe decrypt <file.enc> <passphrase> -> Restores original file
 *
 * Compiling for GCP / Linux (optimized):
 *   make  # Using the included Makefile which detects GPU architecture
 *
 * Manual Compilation (Linux/GCP):
 *   nvcc -O3 -use_fast_math -arch=sm_75 --ptxas-options=-v -lineinfo encrypt.cu
 * -o vibecoder
 *
 * Manual Compilation (Windows MSVC):
 *   nvcc -O3 -use_fast_math -ccbin "C:\Path\To\MSVC\bin\Hostx64\x64" encrypt.cu
 * -o encryption.exe
 */

#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// --- Configuration ---
#define CUDA_BLOCK_SIZE 256

// --- Error Handling ---
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
bool get_os_random_bytes(uint8_t *buffer, size_t size) {
  try {
    std::random_device rd;
    for (size_t i = 0; i < size; ++i) {
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
class SHA256 {
  uint32_t state[8];
  uint8_t buffer[64];
  uint64_t count;

  inline uint32_t jrotate(uint32_t x, uint32_t c) {
    return (x >> c) | (x << (32 - c));
  }

public:
  SHA256() { reset(); }

  void reset() {
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
    uint64_t bitlen = count * 8;
    buffer[count % 64] = 0x80;
    size_t current_len = count % 64 + 1;
    if (current_len > 56) {
      memset(buffer + current_len, 0, 64 - current_len);
      transform(buffer);
      memset(buffer, 0, 56);
    } else {
      memset(buffer + current_len, 0, 56 - current_len);
    }

    for (int i = 0; i < 8; ++i)
      buffer[63 - i] = (bitlen >> (i * 8)) & 0xFF;
    transform(buffer);

    for (int i = 0; i < 8; ++i) {
      hash[i * 4] = (state[i] >> 24) & 0xFF;
      hash[i * 4 + 1] = (state[i] >> 16) & 0xFF;
      hash[i * 4 + 2] = (state[i] >> 8) & 0xFF;
      hash[i * 4 + 3] = state[i] & 0xFF;
    }
  }

private:
  void transform(const uint8_t *data) {
    uint32_t w[64], a, b, c, d, e, f, g, h;
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
    a = state[0];
    b = state[1];
    c = state[2];
    d = state[3];
    e = state[4];
    f = state[5];
    g = state[6];
    h = state[7];

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
// DEVICE: ChaCha20 Implementation
// ==========================================

// Rotate function
__device__ __forceinline__ void quarter_round(uint32_t &a, uint32_t &b,
                                              uint32_t &c, uint32_t &d) {
  a += b;
  d ^= a;
  // Manual bitwise rotation (compiler will optimize to PTX shift instructions)
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

/**
 * @brief Optimized encryption kernel using Shared Memory and Vectorized Loads
 * Each thread handles ONE 64-byte block.
 */
__global__ void encrypt_kernel_optimized(uint4 *data, const uint32_t *key,
                                         const uint32_t *nonce, int n_chunks) {
  // 1. Shared Memory for Constants
  // Pulling Key/Nonce from Shared Memory reduces Global Memory pressure.
  __shared__ uint32_t s_key[8];
  __shared__ uint32_t s_nonce[2]; // Using 64-bit Nonce to match format

  // Cooperative Load into Shared Memory
  if (threadIdx.x < 8)
    s_key[threadIdx.x] = key[threadIdx.x];
  if (threadIdx.x < 2)
    s_nonce[threadIdx.x] = nonce[threadIdx.x];
  __syncthreads();

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n_chunks) {
    // 2. Vectorized Load
    // Load 64 bytes (1 block) as 4 x uint4 vectors
    // data pointer is cast to uint4*, so data[idx*4] points to this thread's
    // block start
    int vec_start = idx * 4;

    uint4 v0 = data[vec_start + 0];
    uint4 v1 = data[vec_start + 1];
    uint4 v2 = data[vec_start + 2];
    uint4 v3 = data[vec_start + 3];

    // 3. Prepare State
    uint32_t state[16];
    state[0] = 0x61707865;
    state[1] = 0x3320646e;
    state[2] = 0x79622d32;
    state[3] = 0x6b206574;

#pragma unroll
    for (int i = 0; i < 8; i++)
      state[4 + i] = s_key[i];

    // Counter (64-bit) based on block index `idx`
    state[12] = (uint32_t)(idx & 0xFFFFFFFF);
    state[13] = (uint32_t)((uint64_t)idx >> 32);

    state[14] = s_nonce[0];
    state[15] = s_nonce[1];

    // Copy working state
    uint32_t work[16];
#pragma unroll
    for (int i = 0; i < 16; i++)
      work[i] = state[i];

    // 4. Transform (20 Rounds)
    for (int i = 0; i < 10; i++) {
      quarter_round(work[0], work[4], work[8], work[12]);
      quarter_round(work[1], work[5], work[9], work[13]);
      quarter_round(work[2], work[6], work[10], work[14]);
      quarter_round(work[3], work[7], work[11], work[15]);

      quarter_round(work[0], work[5], work[10], work[15]);
      quarter_round(work[1], work[6], work[11], work[12]);
      quarter_round(work[2], work[7], work[8], work[13]);
      quarter_round(work[3], work[4], work[9], work[14]);
    }

// 5. Add original state
#pragma unroll
    for (int i = 0; i < 16; i++)
      work[i] += state[i];

    // 6. XOR with Loaded Data (Vectorized)
    // Manual XOR with registers directly
    v0.x ^= work[0];
    v0.y ^= work[1];
    v0.z ^= work[2];
    v0.w ^= work[3];
    v1.x ^= work[4];
    v1.y ^= work[5];
    v1.z ^= work[6];
    v1.w ^= work[7];
    v2.x ^= work[8];
    v2.y ^= work[9];
    v2.z ^= work[10];
    v2.w ^= work[11];
    v3.x ^= work[12];
    v3.y ^= work[13];
    v3.z ^= work[14];
    v3.w ^= work[15];

    // 7. Vectorized Store
    data[vec_start + 0] = v0;
    data[vec_start + 1] = v1;
    data[vec_start + 2] = v2;
    data[vec_start + 3] = v3;
  }
}

// ==========================================
// MAIN
// ==========================================
int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0]
              << " <encrypt/decrypt> <filename> <passphrase>" << std::endl;
    return 1;
  }
  std::string mode = argv[1];
  std::string filename = argv[2];
  std::string passphrase = argv[3];

  // Key Derivation
  uint8_t raw_key[32];
  SHA256 sha;
  sha.update((const uint8_t *)passphrase.c_str(), passphrase.length());
  sha.finalize(raw_key);

  std::cout << "Derived Key (Hex): ";
  for (int i = 0; i < 32; i++)
    std::cout << std::hex << std::setw(2) << std::setfill('0')
              << (int)raw_key[i];
  std::cout << std::dec << std::endl;

  // Buffer Setup
  uint8_t nonce_bytes[8] = {0};
  uint8_t *h_data = nullptr; // Host Pinned Memory
  size_t n = 0;
  std::string out_filename;

  if (mode == "encrypt") {
    out_filename = filename + ".enc";
    if (!get_os_random_bytes(nonce_bytes, 8))
      return 1;

    std::ifstream infile(filename, std::ios::binary | std::ios::ate);
    if (!infile)
      return 1;
    n = infile.tellg();
    infile.seekg(0, std::ios::beg);

    // Pad to multiple of 64 bytes for safe vectorized processing
    size_t padded_n = (n + 63) / 64 * 64;

    gpuErrchk(cudaMallocHost((void **)&h_data, padded_n));
    memset(h_data, 0, padded_n); // Zero init padding
    infile.read((char *)h_data, n);
    infile.close();

    std::ofstream outfile(out_filename, std::ios::binary);
    outfile.write((char *)nonce_bytes, 8);
    outfile.close();

  } else if (mode == "decrypt") {
    if (filename.length() > 4 &&
        filename.substr(filename.length() - 4) == ".enc")
      out_filename = filename.substr(0, filename.length() - 4);
    else
      out_filename = "decrypted_" + filename;

    std::ifstream infile(filename, std::ios::binary | std::ios::ate);
    if (!infile)
      return 1;
    size_t total_size = infile.tellg();
    if (total_size < 8)
      return 1;

    n = total_size - 8;
    infile.seekg(0, std::ios::beg);
    infile.read((char *)nonce_bytes, 8);

    size_t padded_n = (n + 63) / 64 * 64;
    gpuErrchk(cudaMallocHost((void **)&h_data, padded_n));
    memset(h_data, 0, padded_n);
    infile.read((char *)h_data, n);
    infile.close();
  } else {
    return 1;
  }

  std::cout << "Nonce (Hex): ";
  for (int i = 0; i < 8; i++)
    std::cout << std::hex << std::setw(2) << std::setfill('0')
              << (int)nonce_bytes[i];
  std::cout << std::dec << std::endl;
  std::cout << mode << "ing " << n << " bytes..." << std::endl;

  size_t padded_n = (n + 63) / 64 * 64;
  int num_blocks = (int)(padded_n / 64);

  // GPU Setup
  uint4 *d_data;
  uint32_t *d_key, *d_nonce;

  // Convert 8-byte key/nonce into uint32 arrays
  uint32_t key32[8];
  uint32_t nonce32[2];
  memcpy(key32, raw_key, 32);
  memcpy(nonce32, nonce_bytes, 8);

  gpuErrchk(cudaMalloc((void **)&d_data, padded_n)); // Aligned allocation
  gpuErrchk(cudaMalloc((void **)&d_key, 32));
  gpuErrchk(cudaMalloc((void **)&d_nonce, 8));

  gpuErrchk(cudaMemcpy(d_data, h_data, padded_n, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_key, key32, 32, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_nonce, nonce32, 8, cudaMemcpyHostToDevice));

  // Launch
  int blockSize = CUDA_BLOCK_SIZE;
  int gridSize = (num_blocks + blockSize - 1) / blockSize;

  std::cout << "Launching Kernel: " << num_blocks
            << " blocks, Grid: " << gridSize << std::endl;

  encrypt_kernel_optimized<<<gridSize, blockSize>>>(d_data, d_key, d_nonce,
                                                    num_blocks);

  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  gpuErrchk(cudaMemcpy(h_data, d_data, padded_n, cudaMemcpyDeviceToHost));

  // Output
  std::ofstream outfile(out_filename, std::ios::binary | std::ios::app);
  outfile.write((char *)h_data, n); // Write original size n (ignore padding)
  outfile.close();

  // Cleanup
  cudaFree(d_data);
  cudaFree(d_key);
  cudaFree(d_nonce);
  cudaFreeHost(h_data);

  std::cout << "Success -> " << out_filename << std::endl;
  return 0;
}
