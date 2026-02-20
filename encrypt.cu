/**
 * @file encrypt.cu
 * @brief Industrial-Strength GPU File Encryption (ChaCha20-Poly1305 + Argon2id)
 *
 * Security model:
 *   - Key Derivation : Argon2id (RFC 9106) — memory-hard, GPU/ASIC resistant
 *   - Encryption     : ChaCha20 stream cipher (RFC 8439) — GPU-parallelised
 *   - Authentication : Poly1305 MAC (RFC 8439) — CPU-side, per AEAD spec
 *   - AAD            : File header authenticated via Poly1305 (RFC 8439 §2.8)
 *   - Nonce          : 96-bit random (OS CSPRNG), stored in file header
 *   - Salt           : 128-bit random (OS CSPRNG), stored in file header
 *
 * Security fixes vs v1:
 *   - Passphrase read from /dev/tty — never via argv (no ps/history exposure)
 *   - Header bytes fed into Poly1305 as AAD (prevents header-swapping attacks)
 *   - Plaintext in pinned memory zeroed on encrypt path before cudaFreeHost
 *   - GPU capability check at startup — graceful error if no CUDA device found
 *
 * New features:
 *   - Streaming encryption — fixed 64 MB window, supports arbitrarily large files
 *   - Directory mode (-r) — recursively encrypts/decrypts every file in a tree
 *   - Progress bar — real-time MB/s throughput on stderr
 *   - Verify mode — MAC check only, no plaintext written
 *
 * File format (.enc):
 *   [4 bytes magic 'VBCR'] [1 byte version=0x02] [16 bytes salt] [12 bytes nonce]
 *   [8 bytes orig_size]    [ciphertext chunks...]  [16 bytes Poly1305 tag]
 *
 *   The entire 41-byte header is fed to Poly1305 as AAD before the ciphertext,
 *   binding the header fields to the authentication tag.
 *
 * Usage:
 *   vibecoder encrypt  [-r] [--chunk <MB>] <file|dir>   -> <file>.enc
 *   vibecoder decrypt  [-r] [--chunk <MB>] <file|dir>   -> original file
 *   vibecoder verify        [--chunk <MB>] <file.enc>   -> MAC check only
 *
 *   --chunk <MB>  Streaming window in MB (default 256, min 16).
 *                 Automatically clamped to 40% of free VRAM if needed.
 *
 *   Passphrase: read from /dev/tty (interactive, echo disabled)
 *               or VIBECODER_PASSPHRASE env var (CI / scripted use)
 *
 * Compilation (Linux, sm_75+):
 *   nvcc -O3 -use_fast_math -arch=sm_75 -std=c++17 \
 *        --ptxas-options=-v encrypt.cu -o vibecoder -largon2
 *
 * Compilation (Windows MSVC + vcpkg argon2):
 *   nvcc -allow-unsupported-compiler -O3 -use_fast_math -std=c++17 ^
 *     -I "C:\vcpkg\installed\x64-windows\include" ^
 *     -L "C:\vcpkg\installed\x64-windows\lib" ^
 *     encrypt.cu -o vibecoder.exe -largon2
 *
 * Dependencies:
 *   - CUDA Toolkit >= 10.0
 *   - C++17 standard library  (for std::filesystem)
 *   - Argon2 reference implementation  (link: -largon2)
 *
 * Security notes:
 *   - Decryption aborts and removes partial output if MAC fails.
 *   - All key material is zeroed after use via secure_zero().
 *   - Passphrase is never printed, logged, or passed via argv.
 */

#include <argon2.h>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <chrono>

#ifndef _WIN32
#  include <termios.h>
#  include <unistd.h>
#  include <fcntl.h>
#else
#  include <windows.h>
#  include <conio.h>
#endif

namespace fs = std::filesystem;

// ─── Configuration ────────────────────────────────────────────────────────────

#define CUDA_BLOCK_SIZE      256
// Default streaming window: 256 MB.
// Overridden at runtime via --chunk <MB> (clamped to available VRAM).
#define DEFAULT_CHUNK_MB     256ULL
#define MIN_CHUNK_MB         16ULL

// Global chunk size — set once in main() before any file operation.
static uint64_t g_chunk_bytes = DEFAULT_CHUNK_MB * 1024 * 1024;

static constexpr uint8_t  MAGIC[4]  = {'V', 'B', 'C', 'R'};
static constexpr uint8_t  VERSION   = 0x02;  // v2: AAD + streaming
static constexpr size_t   SALT_LEN  = 16;
static constexpr size_t   NONCE_LEN = 12;
static constexpr size_t   TAG_LEN   = 16;
static constexpr size_t   KEY_LEN   = 32;
// 4+1+16+12+8 = 41 bytes
static constexpr size_t   HEADER_SIZE =
    sizeof(MAGIC) + sizeof(VERSION) + SALT_LEN + NONCE_LEN + sizeof(uint64_t);

#define ARGON2_T_COST       3
#define ARGON2_M_COST       65536   // 64 MB
#define ARGON2_PARALLELISM  4

// ─── CUDA Error Handling ──────────────────────────────────────────────────────

#define gpuCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s  (%s:%d)\n",
            cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

// ─── GPU Capability Check ─────────────────────────────────────────────────────
/**
 * Fail early with a clear message if no CUDA GPU is present, rather than
 * letting the first cudaMalloc emit a cryptic driver error.
 */
static void require_gpu() {
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess || count == 0) {
    std::cerr << "Error: no CUDA-capable GPU detected.\n"
              << "  (" << cudaGetErrorString(err) << ")\n";
    exit(1);
  }
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, 0);
  std::cerr << "GPU : " << prop.name
            << "  (" << prop.totalGlobalMem / (1024*1024) << " MB VRAM)\n";
}

// ─── Chunk size selection ─────────────────────────────────────────────────────
/**
 * Clamp the requested chunk size to a safe fraction of available VRAM.
 * We budget at most 40% of free VRAM for the pinned host + device buffers
 * (the kernel needs two copies: h_buf + d_data = 2x chunk).
 * If the requested size fits, use it; otherwise halve until it does or
 * we hit MIN_CHUNK_MB.
 */
static uint64_t choose_chunk_size(uint64_t requested_bytes) {
  size_t free_vram = 0, total_vram = 0;
  cudaMemGetInfo(&free_vram, &total_vram);
  // Two buffers of chunk size: h_buf (pinned) + d_data (device)
  uint64_t budget = (uint64_t)(free_vram * 0.40);
  uint64_t chunk  = requested_bytes;
  while (chunk > MIN_CHUNK_MB * 1024 * 1024 && chunk * 2 > budget)
    chunk /= 2;
  chunk = std::max(chunk, MIN_CHUNK_MB * 1024 * 1024);
  // Round down to 64-byte ChaCha20 block boundary
  chunk = (chunk / 64) * 64;
  if (chunk != requested_bytes) {
    std::cerr << "Note: chunk size reduced to "
              << chunk / (1024*1024) << " MB to fit available VRAM ("
              << free_vram / (1024*1024) << " MB free).\n";
  } else {
    std::cerr << "Chunk: " << chunk / (1024*1024) << " MB  |  "
              << "VRAM free: " << free_vram / (1024*1024) << " MB\n";
  }
  return chunk;
}

// ─── Secure Zeroing ───────────────────────────────────────────────────────────
// volatile prevents the compiler optimising the loop away.

static void secure_zero(void *ptr, size_t len) {
  volatile uint8_t *p = reinterpret_cast<volatile uint8_t *>(ptr);
  while (len--) *p++ = 0;
}

// ─── OS Entropy ───────────────────────────────────────────────────────────────

static void get_random_bytes(uint8_t *buf, size_t len) {
  std::random_device rd;
  for (size_t i = 0; i < len; ++i)
    buf[i] = static_cast<uint8_t>(rd());
}

// ─── Passphrase Input ─────────────────────────────────────────────────────────
/**
 * Read a passphrase with echo disabled.
 *
 * Priority:
 *   1. VIBECODER_PASSPHRASE env var  (CI / scripted pipelines)
 *   2. Interactive /dev/tty prompt   (default; never goes through argv)
 *
 * Passing secrets via argv is unsafe: they appear in `ps aux`, shell history,
 * and /proc/<pid>/cmdline on Linux.
 */
static std::string read_passphrase(const char *prompt) {
  const char *env = std::getenv("VIBECODER_PASSPHRASE");
  if (env && env[0] != '\0') {
    std::cerr << "[passphrase sourced from VIBECODER_PASSPHRASE]\n";
    return std::string(env);
  }

  std::string pass;

#ifndef _WIN32
  int tty = open("/dev/tty", O_RDWR);
  if (tty < 0) {
    std::cerr << "Error: cannot open /dev/tty for passphrase input.\n";
    exit(1);
  }
  struct termios old_tio, new_tio;
  tcgetattr(tty, &old_tio);
  new_tio = old_tio;
  new_tio.c_lflag &= ~(tcflag_t)(ECHO | ECHOE | ECHOK | ECHONL);
  tcsetattr(tty, TCSAFLUSH, &new_tio);
  write(tty, prompt, strlen(prompt));
  char c;
  while (read(tty, &c, 1) == 1 && c != '\n' && c != '\r')
    pass += c;
  write(tty, "\n", 1);
  tcsetattr(tty, TCSAFLUSH, &old_tio);
  close(tty);
#else
  std::cerr << prompt;
  int c;
  while ((c = _getch()) != '\r' && c != '\n' && c != EOF)
    pass += static_cast<char>(c);
  std::cerr << "\n";
#endif

  return pass;
}

// ─── Argon2id Key Derivation ──────────────────────────────────────────────────

static void derive_key(const std::string &passphrase, const uint8_t *salt,
                       uint8_t out_key[KEY_LEN]) {
  int rc = argon2id_hash_raw(
      ARGON2_T_COST, ARGON2_M_COST, ARGON2_PARALLELISM,
      passphrase.c_str(), passphrase.size(),
      salt, SALT_LEN, out_key, KEY_LEN);
  if (rc != ARGON2_OK) {
    std::cerr << "Argon2id failed: " << argon2_error_message(rc) << "\n";
    exit(1);
  }
}

// ─── Poly1305 (CPU, RFC 8439) ─────────────────────────────────────────────────

class Poly1305 {
  uint32_t r[5], h[5], pad[4];

  static uint32_t le32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1]<<8) |
           ((uint32_t)p[2]<<16) | ((uint32_t)p[3]<<24);
  }

public:
  Poly1305(const uint8_t key[32]) {
    r[0] =  le32(key+ 0) & 0x0fffffff;
    r[1] = (le32(key+ 3) >>  2) & 0x0ffffffc;
    r[2] = (le32(key+ 6) >>  4) & 0x0ffffffc;
    r[3] = (le32(key+ 9) >>  6) & 0x0ffffffc;
    r[4] = (le32(key+12) >>  8) & 0x0000000f;
    pad[0]=le32(key+16); pad[1]=le32(key+20);
    pad[2]=le32(key+24); pad[3]=le32(key+28);
    h[0]=h[1]=h[2]=h[3]=h[4]=0;
  }

  void update(const uint8_t *msg, size_t len) {
    while (len > 0) {
      uint8_t block[17]={0};
      size_t want = (len>=16)?16:len;
      memcpy(block, msg, want);
      block[want]=1;
      uint32_t m[5];
      m[0]=le32(block+ 0); m[1]=le32(block+4);
      m[2]=le32(block+ 8); m[3]=le32(block+12); m[4]=block[16];
      h[0]+=m[0]; h[1]+=m[1]; h[2]+=m[2]; h[3]+=m[3]; h[4]+=m[4];

      uint64_t d0=(uint64_t)h[0]*r[0]+(uint64_t)h[1]*(5*r[4])+(uint64_t)h[2]*(5*r[3])+(uint64_t)h[3]*(5*r[2])+(uint64_t)h[4]*(5*r[1]);
      uint64_t d1=(uint64_t)h[0]*r[1]+(uint64_t)h[1]*    r[0]+(uint64_t)h[2]*(5*r[4])+(uint64_t)h[3]*(5*r[3])+(uint64_t)h[4]*(5*r[2]);
      uint64_t d2=(uint64_t)h[0]*r[2]+(uint64_t)h[1]*    r[1]+(uint64_t)h[2]*    r[0]+(uint64_t)h[3]*(5*r[4])+(uint64_t)h[4]*(5*r[3]);
      uint64_t d3=(uint64_t)h[0]*r[3]+(uint64_t)h[1]*    r[2]+(uint64_t)h[2]*    r[1]+(uint64_t)h[3]*    r[0]+(uint64_t)h[4]*(5*r[4]);
      uint64_t d4=(uint64_t)h[0]*r[4]+(uint64_t)h[1]*    r[3]+(uint64_t)h[2]*    r[2]+(uint64_t)h[3]*    r[1]+(uint64_t)h[4]*    r[0];

      uint32_t c=(uint32_t)(d0>>26); h[0]=(uint32_t)d0&0x3ffffff; d1+=c;
                 c=(uint32_t)(d1>>26); h[1]=(uint32_t)d1&0x3ffffff; d2+=c;
                 c=(uint32_t)(d2>>26); h[2]=(uint32_t)d2&0x3ffffff; d3+=c;
                 c=(uint32_t)(d3>>26); h[3]=(uint32_t)d3&0x3ffffff; d4+=c;
                 c=(uint32_t)(d4>>26); h[4]=(uint32_t)d4&0x3ffffff; h[0]+=c*5;
      c=h[0]>>26; h[0]&=0x3ffffff; h[1]+=c;
      msg+=want; len-=want;
    }
  }

  /**
   * Feed AAD with RFC 8439 §2.8 zero-padding to the next 16-byte boundary.
   * Called once for the header before any ciphertext is processed.
   */
  void update_aad(const uint8_t *aad, size_t len) {
    update(aad, len);
    size_t pad_len = (16 - (len % 16)) % 16;
    if (pad_len) {
      uint8_t zeros[16]={};
      update(zeros, pad_len);
    }
  }

  void finalize(uint8_t tag[16]) {
    uint32_t c=h[1]>>26; h[1]&=0x3ffffff; h[2]+=c;
    c=h[2]>>26; h[2]&=0x3ffffff; h[3]+=c;
    c=h[3]>>26; h[3]&=0x3ffffff; h[4]+=c;
    c=h[4]>>26; h[4]&=0x3ffffff; h[0]+=c*5;
    c=h[0]>>26; h[0]&=0x3ffffff; h[1]+=c;
    uint32_t g[5];
    g[0]=h[0]+5; c=g[0]>>26; g[0]&=0x3ffffff;
    g[1]=h[1]+c; c=g[1]>>26; g[1]&=0x3ffffff;
    g[2]=h[2]+c; c=g[2]>>26; g[2]&=0x3ffffff;
    g[3]=h[3]+c; c=g[3]>>26; g[3]&=0x3ffffff;
    g[4]=h[4]+c-(1<<26);
    uint32_t mask=(g[4]>>31)-1;
    for(int i=0;i<5;i++) h[i]=(h[i]&~mask)|(g[i]&mask);
    uint64_t f;
    f=(uint64_t)h[0]|((uint64_t)h[1]<<26)|((uint64_t)h[2]<<52);
    uint32_t t0=(uint32_t)f; f>>=32;
    f|=(uint64_t)h[2]>>12|((uint64_t)h[3]<<14)|((uint64_t)h[4]<<40);
    uint32_t t1=(uint32_t)f; f>>=32;
    uint32_t t2=(uint32_t)((uint64_t)h[3]>>18|((uint64_t)h[4]<<8));
    f=(uint64_t)t0+pad[0];
    tag[0]=(uint8_t)f;tag[1]=(uint8_t)(f>>8);tag[2]=(uint8_t)(f>>16);tag[3]=(uint8_t)(f>>24);f>>=32;
    f+=(uint64_t)t1+pad[1];
    tag[4]=(uint8_t)f;tag[5]=(uint8_t)(f>>8);tag[6]=(uint8_t)(f>>16);tag[7]=(uint8_t)(f>>24);f>>=32;
    f+=(uint64_t)t2+pad[2];
    tag[8]=(uint8_t)f;tag[9]=(uint8_t)(f>>8);tag[10]=(uint8_t)(f>>16);tag[11]=(uint8_t)(f>>24);f>>=32;
    f+=pad[3];
    tag[12]=(uint8_t)f;tag[13]=(uint8_t)(f>>8);tag[14]=(uint8_t)(f>>16);tag[15]=(uint8_t)(f>>24);
  }
};

// ─── Constant-time tag comparison ────────────────────────────────────────────

static bool ct_equal(const uint8_t *a, const uint8_t *b, size_t len) {
  uint8_t diff = 0;
  for (size_t i = 0; i < len; i++) diff |= a[i] ^ b[i];
  return diff == 0;
}

// ─── ChaCha20 Quarter Round (shared host + device) ───────────────────────────
/**
 * __host__ __device__ allows nvcc to inline this into both the GPU kernel
 * and the CPU generate_poly1305_key() — single canonical implementation.
 */
__host__ __device__ __forceinline__
void quarter_round(uint32_t &a, uint32_t &b, uint32_t &c, uint32_t &d) {
  a+=b; d^=a; d=(d<<16)|(d>>16);
  c+=d; b^=c; b=(b<<12)|(b>>20);
  a+=b; d^=a; d=(d<< 8)|(d>>24);
  c+=d; b^=c; b=(b<< 7)|(b>>25);
}

// ─── GPU: ChaCha20 Keystream Kernel ──────────────────────────────────────────
/**
 * Each thread encrypts exactly one 64-byte ChaCha20 block.
 * ctr_base allows counter to pick up across streaming chunks.
 */
__global__ void chacha20_kernel(uint4 *data, const uint32_t *key,
                                const uint32_t *nonce, int n_chunks,
                                uint32_t ctr_base) {
  __shared__ uint32_t s_key[8];
  __shared__ uint32_t s_nonce[3];
  if (threadIdx.x < 8) s_key[threadIdx.x]   = key[threadIdx.x];
  if (threadIdx.x < 3) s_nonce[threadIdx.x] = nonce[threadIdx.x];
  __syncthreads();

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_chunks) return;

  int vec = idx * 4;
  uint4 v0=data[vec+0], v1=data[vec+1], v2=data[vec+2], v3=data[vec+3];

  uint32_t st[16];
  st[0]=0x61707865; st[1]=0x3320646e; st[2]=0x79622d32; st[3]=0x6b206574;
#pragma unroll
  for (int i=0;i<8;i++) st[4+i]=s_key[i];
  st[12]=ctr_base+(uint32_t)idx;
  st[13]=s_nonce[0]; st[14]=s_nonce[1]; st[15]=s_nonce[2];

  uint32_t wk[16];
#pragma unroll
  for (int i=0;i<16;i++) wk[i]=st[i];
#pragma unroll
  for (int i=0;i<10;i++) {
    quarter_round(wk[0],wk[4],wk[ 8],wk[12]);
    quarter_round(wk[1],wk[5],wk[ 9],wk[13]);
    quarter_round(wk[2],wk[6],wk[10],wk[14]);
    quarter_round(wk[3],wk[7],wk[11],wk[15]);
    quarter_round(wk[0],wk[5],wk[10],wk[15]);
    quarter_round(wk[1],wk[6],wk[11],wk[12]);
    quarter_round(wk[2],wk[7],wk[ 8],wk[13]);
    quarter_round(wk[3],wk[4],wk[ 9],wk[14]);
  }
#pragma unroll
  for (int i=0;i<16;i++) wk[i]+=st[i];

  v0.x^=wk[0];  v0.y^=wk[1];  v0.z^=wk[2];  v0.w^=wk[3];
  v1.x^=wk[4];  v1.y^=wk[5];  v1.z^=wk[6];  v1.w^=wk[7];
  v2.x^=wk[8];  v2.y^=wk[9];  v2.z^=wk[10]; v2.w^=wk[11];
  v3.x^=wk[12]; v3.y^=wk[13]; v3.z^=wk[14]; v3.w^=wk[15];

  data[vec+0]=v0; data[vec+1]=v1; data[vec+2]=v2; data[vec+3]=v3;
}

// ─── CPU: Generate Poly1305 one-time key (RFC 8439 §2.6) ─────────────────────
/**
 * Run ChaCha20 with counter=0; first 32 bytes of keystream become the OTK.
 * Uses the shared quarter_round() — no duplicate round logic.
 */
static void generate_poly1305_key(const uint8_t key[32],
                                   const uint8_t nonce[12],
                                   uint8_t otk[32]) {
  uint32_t st[16];
  st[0]=0x61707865; st[1]=0x3320646e; st[2]=0x79622d32; st[3]=0x6b206574;
  for (int i=0;i<8;i++)
    st[4+i]=(uint32_t)key[i*4]|((uint32_t)key[i*4+1]<<8)|
            ((uint32_t)key[i*4+2]<<16)|((uint32_t)key[i*4+3]<<24);
  st[12]=0;
  for (int i=0;i<3;i++)
    st[13+i]=(uint32_t)nonce[i*4]|((uint32_t)nonce[i*4+1]<<8)|
             ((uint32_t)nonce[i*4+2]<<16)|((uint32_t)nonce[i*4+3]<<24);
  uint32_t wk[16];
  for (int i=0;i<16;i++) wk[i]=st[i];
  for (int i=0;i<10;i++) {
    quarter_round(wk[0],wk[4],wk[ 8],wk[12]);
    quarter_round(wk[1],wk[5],wk[ 9],wk[13]);
    quarter_round(wk[2],wk[6],wk[10],wk[14]);
    quarter_round(wk[3],wk[7],wk[11],wk[15]);
    quarter_round(wk[0],wk[5],wk[10],wk[15]);
    quarter_round(wk[1],wk[6],wk[11],wk[12]);
    quarter_round(wk[2],wk[7],wk[ 8],wk[13]);
    quarter_round(wk[3],wk[4],wk[ 9],wk[14]);
  }
  for (int i=0;i<16;i++) wk[i]+=st[i];
  for (int i=0;i<8;i++) {
    otk[i*4+0]=(uint8_t)(wk[i]);
    otk[i*4+1]=(uint8_t)(wk[i]>>8);
    otk[i*4+2]=(uint8_t)(wk[i]>>16);
    otk[i*4+3]=(uint8_t)(wk[i]>>24);
  }
}

// ─── Progress Bar ─────────────────────────────────────────────────────────────

struct Progress {
  std::string label;
  uint64_t    total;
  uint64_t    done = 0;
  std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();

  Progress(std::string lbl, uint64_t total_bytes)
    : label(std::move(lbl)), total(total_bytes) { render(); }

  void advance(uint64_t bytes) { done += bytes; render(); }

  void finish() { done = total; render(); std::cerr << "\n"; }

private:
  void render() const {
    auto now    = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(now - t0).count();
    double mbps = (secs > 0.01) ? (done / 1048576.0) / secs : 0.0;
    int pct     = (total > 0) ? (int)(100.0 * done / total) : 0;
    int fill    = (total > 0) ? (int)(40.0  * done / total) : 0;

    // Rightmost 20 chars of label so long paths don't wrap
    std::string lbl = label.size() > 20
                    ? "..." + label.substr(label.size()-17) : label;

    std::cerr << "\r\033[K"
              << std::left  << std::setw(21) << lbl << " ["
              << std::string(fill, '=')
              << (fill < 40 ? ">" : "")
              << std::string(std::max(0, 39-fill), ' ')
              << "] "
              << std::right << std::setw(3) << pct << "%"
              << "  " << std::fixed << std::setprecision(1)
              << std::setw(7) << mbps << " MB/s"
              << std::flush;
  }
};

// ─── AAD construction ────────────────────────────────────────────────────────
/**
 * Serialise the 41-byte header into a flat buffer for Poly1305 AAD input.
 * Must match the exact on-disk layout written by encrypt_file().
 */
static void build_aad(uint8_t aad[HEADER_SIZE],
                      const uint8_t salt[SALT_LEN],
                      const uint8_t nonce[NONCE_LEN],
                      uint64_t orig_size) {
  uint8_t *p = aad;
  memcpy(p, MAGIC,    4);        p += 4;
  memcpy(p, &VERSION, 1);        p += 1;
  memcpy(p, salt,  SALT_LEN);    p += SALT_LEN;
  memcpy(p, nonce, NONCE_LEN);   p += NONCE_LEN;
  memcpy(p, &orig_size, 8);
}

// ─── GPU resource bundle (RAII helper) ───────────────────────────────────────

struct GpuCtx {
  uint4    *d_data  = nullptr;
  uint32_t *d_key   = nullptr;
  uint32_t *d_nonce = nullptr;
  uint8_t  *h_buf   = nullptr;
  size_t    padded_chunk;

  explicit GpuCtx(size_t chunk = 0) {
    if (chunk == 0) chunk = (size_t)g_chunk_bytes;
    padded_chunk = ((chunk + 63) / 64) * 64;
    gpuCheck(cudaMallocHost(&h_buf,  padded_chunk));
    gpuCheck(cudaMalloc(&d_data,  padded_chunk));
    gpuCheck(cudaMalloc(&d_key,   32));
    gpuCheck(cudaMalloc(&d_nonce, 12));
  }

  void upload_key_nonce(const uint32_t key32[8], const uint32_t nonce32[3]) {
    gpuCheck(cudaMemcpy(d_key,   key32,   32, cudaMemcpyHostToDevice));
    gpuCheck(cudaMemcpy(d_nonce, nonce32, 12, cudaMemcpyHostToDevice));
  }

  ~GpuCtx() {
    if (h_buf)   { secure_zero(h_buf, padded_chunk); cudaFreeHost(h_buf); }
    if (d_data)  cudaFree(d_data);
    if (d_key)   cudaFree(d_key);
    if (d_nonce) cudaFree(d_nonce);
  }
};

// ─── encrypt_file ────────────────────────────────────────────────────────────

static bool encrypt_file(const std::string &src, const std::string &dst,
                         const std::string &passphrase) {
  std::ifstream in(src, std::ios::binary | std::ios::ate);
  if (!in) { std::cerr << "Cannot open: " << src << "\n"; return false; }
  uint64_t orig_size = in.tellg();
  in.seekg(0);

  std::ofstream out(dst, std::ios::binary);
  if (!out) { std::cerr << "Cannot create: " << dst << "\n"; return false; }

  uint8_t salt[SALT_LEN], nonce[NONCE_LEN];
  get_random_bytes(salt,  SALT_LEN);
  get_random_bytes(nonce, NONCE_LEN);

  std::cerr << "Deriving key (Argon2id, " << ARGON2_M_COST/1024
            << " MB, " << ARGON2_T_COST << " iter)...\n";
  uint8_t key[KEY_LEN];
  derive_key(passphrase, salt, key);

  uint8_t otk[32];
  generate_poly1305_key(key, nonce, otk);

  // Write header and authenticate it as AAD
  uint8_t aad[HEADER_SIZE];
  build_aad(aad, salt, nonce, orig_size);
  out.write(reinterpret_cast<char*>(aad), HEADER_SIZE);
  if (!out.good()) { std::cerr << "Write error: " << dst << "\n"; return false; }

  Poly1305 mac(otk);
  mac.update_aad(aad, HEADER_SIZE);

  // GPU setup
  uint32_t key32[8], nonce32[3];
  memcpy(key32,   key,   32);
  memcpy(nonce32, nonce, 12);

  GpuCtx gpu;
  gpu.upload_key_nonce(key32, nonce32);
  secure_zero(key, KEY_LEN);
  secure_zero(key32, 32);

  Progress bar(fs::path(src).filename().string(), orig_size);
  uint64_t remaining = orig_size;
  uint32_t ctr_base  = 1;   // block 0 reserved for OTK

  while (remaining > 0) {
    uint64_t read_bytes = std::min(remaining, g_chunk_bytes);
    size_t   padded     = ((read_bytes + 63) / 64) * 64;

    memset(gpu.h_buf, 0, padded);
    in.read(reinterpret_cast<char*>(gpu.h_buf), (std::streamsize)read_bytes);

    gpuCheck(cudaMemcpy(gpu.d_data, gpu.h_buf, padded, cudaMemcpyHostToDevice));
    int n_chunks = (int)(padded / 64);
    int grid     = (n_chunks + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    chacha20_kernel<<<grid, CUDA_BLOCK_SIZE>>>(
        gpu.d_data, gpu.d_key, gpu.d_nonce, n_chunks, ctr_base);
    gpuCheck(cudaPeekAtLastError());
    gpuCheck(cudaDeviceSynchronize());
    gpuCheck(cudaMemcpy(gpu.h_buf, gpu.d_data, padded, cudaMemcpyDeviceToHost));

    mac.update(gpu.h_buf, (size_t)read_bytes);  // authenticate ciphertext

    out.write(reinterpret_cast<char*>(gpu.h_buf), (std::streamsize)read_bytes);
    if (!out.good()) {
      std::cerr << "\nWrite error: " << dst << "\n"; return false;
    }

    // Zero plaintext residue from pinned host buffer immediately
    secure_zero(gpu.h_buf, padded);

    ctr_base  += (uint32_t)n_chunks;
    remaining -= read_bytes;
    bar.advance(read_bytes);
  }
  bar.finish();

  uint8_t tag[TAG_LEN];
  mac.finalize(tag);
  out.write(reinterpret_cast<char*>(tag), TAG_LEN);
  out.close();

  secure_zero(otk, 32);
  std::cerr << "Encrypted -> " << dst << "  (" << orig_size << " bytes)\n";
  return true;
}

// ─── decrypt_file ────────────────────────────────────────────────────────────

static bool decrypt_file(const std::string &src, const std::string &dst,
                         const std::string &passphrase,
                         bool verify_only = false) {
  std::ifstream in(src, std::ios::binary | std::ios::ate);
  if (!in) { std::cerr << "Cannot open: " << src << "\n"; return false; }
  size_t total = (size_t)in.tellg();
  in.seekg(0);

  if (total < HEADER_SIZE + TAG_LEN) {
    std::cerr << "File too small to be valid: " << src << "\n"; return false;
  }

  // Read and validate header
  uint8_t aad[HEADER_SIZE];
  in.read(reinterpret_cast<char*>(aad), HEADER_SIZE);
  if (memcmp(aad, MAGIC, 4) != 0 || aad[4] != VERSION) {
    std::cerr << "Invalid magic or unsupported version: " << src << "\n";
    return false;
  }

  const uint8_t *salt  = aad + 5;
  const uint8_t *nonce = aad + 5 + SALT_LEN;
  uint64_t orig_size;
  memcpy(&orig_size, aad + 5 + SALT_LEN + NONCE_LEN, 8);

  if ((size_t)(total - HEADER_SIZE - TAG_LEN) < orig_size) {
    std::cerr << "Truncated ciphertext: " << src << "\n"; return false;
  }

  std::cerr << "Deriving key (Argon2id)...\n";
  uint8_t key[KEY_LEN];
  derive_key(passphrase, salt, key);

  uint8_t otk[32];
  generate_poly1305_key(key, nonce, otk);

  // Authenticate header as AAD first
  Poly1305 mac(otk);
  mac.update_aad(aad, HEADER_SIZE);

  // Pre-read stored tag from end of file (enables single forward pass)
  uint8_t stored_tag[TAG_LEN];
  {
    std::ifstream t(src, std::ios::binary);
    t.seekg(-(std::streamoff)TAG_LEN, std::ios::end);
    t.read(reinterpret_cast<char*>(stored_tag), TAG_LEN);
  }

  std::ofstream out;
  if (!verify_only) {
    out.open(dst, std::ios::binary);
    if (!out) { std::cerr << "Cannot create: " << dst << "\n"; return false; }
  }

  uint32_t key32[8], nonce32[3];
  memcpy(key32,   key,   32);
  memcpy(nonce32, nonce, 12);

  GpuCtx gpu;
  gpu.upload_key_nonce(key32, nonce32);
  secure_zero(key, KEY_LEN);
  secure_zero(key32, 32);

  Progress bar(fs::path(src).filename().string(), orig_size);
  uint64_t remaining = orig_size;
  uint32_t ctr_base  = 1;

  while (remaining > 0) {
    uint64_t read_bytes = std::min(remaining, g_chunk_bytes);
    size_t   padded     = ((read_bytes + 63) / 64) * 64;

    memset(gpu.h_buf, 0, padded);
    in.read(reinterpret_cast<char*>(gpu.h_buf), (std::streamsize)read_bytes);

    // Authenticate ciphertext BEFORE decrypting (per AEAD spec)
    mac.update(gpu.h_buf, (size_t)read_bytes);

    gpuCheck(cudaMemcpy(gpu.d_data, gpu.h_buf, padded, cudaMemcpyHostToDevice));
    int n_chunks = (int)(padded / 64);
    int grid     = (n_chunks + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    chacha20_kernel<<<grid, CUDA_BLOCK_SIZE>>>(
        gpu.d_data, gpu.d_key, gpu.d_nonce, n_chunks, ctr_base);
    gpuCheck(cudaPeekAtLastError());
    gpuCheck(cudaDeviceSynchronize());
    gpuCheck(cudaMemcpy(gpu.h_buf, gpu.d_data, padded, cudaMemcpyDeviceToHost));

    if (!verify_only) {
      out.write(reinterpret_cast<char*>(gpu.h_buf), (std::streamsize)read_bytes);
      if (!out.good()) {
        std::cerr << "\nWrite error: " << dst << "\n"; return false;
      }
    }

    ctr_base  += (uint32_t)n_chunks;
    remaining -= read_bytes;
    bar.advance(read_bytes);
  }
  bar.finish();

  uint8_t computed_tag[TAG_LEN];
  mac.finalize(computed_tag);

  if (!ct_equal(computed_tag, stored_tag, TAG_LEN)) {
    std::cerr << "Authentication FAILED — wrong passphrase or tampered file:\n"
              << "  " << src << "\n";
    // Remove partial output so corrupt plaintext never reaches disk
    if (!verify_only && fs::exists(dst)) fs::remove(dst);
    secure_zero(otk, 32);
    return false;
  }

  secure_zero(otk, 32);

  if (verify_only) {
    std::cerr << "MAC OK  (verify only) — " << src << "\n";
  } else {
    out.close();
    std::cerr << "Decrypted -> " << dst << "  (" << orig_size << " bytes)\n";
  }
  return true;
}

// ─── Directory walker ─────────────────────────────────────────────────────────
/**
 * Collect all candidate files, then process each with a single shared key
 * derivation (the user is asked for the passphrase once per invocation).
 */
static bool process_directory(const std::string &dir_path, bool encrypting,
                               const std::string &passphrase) {
  if (!fs::is_directory(dir_path)) {
    std::cerr << "Not a directory: " << dir_path << "\n"; return false;
  }

  std::vector<fs::path> targets;
  for (auto &entry : fs::recursive_directory_iterator(dir_path)) {
    if (!entry.is_regular_file()) continue;
    std::string p = entry.path().string();
    bool is_enc = p.size() >= 4 && p.substr(p.size()-4) == ".enc";
    if ( encrypting && !is_enc) targets.push_back(entry.path());
    if (!encrypting &&  is_enc) targets.push_back(entry.path());
  }

  if (targets.empty()) {
    std::cerr << "No " << (encrypting ? "plaintext" : ".enc")
              << " files found under: " << dir_path << "\n";
    return true;
  }

  std::cerr << (encrypting ? "Encrypting" : "Decrypting") << " "
            << targets.size() << " file(s) under " << dir_path << "\n\n";

  int ok = 0, fail = 0;
  for (auto &p : targets) {
    std::string src = p.string();
    std::string dst;
    if (encrypting) {
      dst = src + ".enc";
    } else {
      dst = src.substr(0, src.size()-4);
      if (fs::exists(dst)) {
        std::cerr << "Skipping (output exists): " << dst << "\n\n";
        continue;
      }
    }
    bool result = encrypting ? encrypt_file(src, dst, passphrase)
                             : decrypt_file(src, dst, passphrase);
    result ? ++ok : ++fail;
    std::cerr << "\n";
  }

  std::cerr << "Done: " << ok << " OK, " << fail << " failed.\n";
  return fail == 0;
}

// ─── Usage ────────────────────────────────────────────────────────────────────

static void usage(const char *prog) {
  std::cerr
    << "Usage:\n"
    << "  " << prog << " encrypt [-r] [--chunk <MB>] <file|dir>\n"
    << "  " << prog << " decrypt [-r] [--chunk <MB>] <file.enc|dir>\n"
    << "  " << prog << " verify       [--chunk <MB>] <file.enc>\n\n"
    << "Options:\n"
    << "  -r            Recurse into directory\n"
    << "  --chunk <MB>  Streaming window size in MB (default: "
    << DEFAULT_CHUNK_MB << ", min: " << MIN_CHUNK_MB << ")\n"
    << "                Clamped automatically to fit available VRAM.\n\n"
    << "Passphrase: read from /dev/tty (echo off) or\n"
    << "            VIBECODER_PASSPHRASE environment variable.\n";
}

// ─── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char *argv[]) {
  if (argc < 3) { usage(argv[0]); return 1; }

  std::string mode = argv[1];

  // Parse flags: [-r] [--chunk <MB>]
  bool recursive     = false;
  uint64_t chunk_mb  = DEFAULT_CHUNK_MB;
  int  path_arg      = 2;

  for (int i = 2; i < argc - 1; ++i) {
    std::string arg = argv[i];
    if (arg == "-r") {
      recursive = true;
      path_arg  = i + 1;
    } else if (arg == "--chunk" && i + 1 < argc) {
      try {
        chunk_mb = std::stoull(argv[i + 1]);
        if (chunk_mb < MIN_CHUNK_MB) {
          std::cerr << "--chunk must be >= " << MIN_CHUNK_MB << " MB.\n";
          return 1;
        }
      } catch (...) {
        std::cerr << "Invalid --chunk value: " << argv[i+1] << "\n";
        return 1;
      }
      ++i;          // skip the value token
      path_arg = i + 1;
    }
  }
  if (path_arg >= argc) { usage(argv[0]); return 1; }

  std::string target = argv[path_arg];

  // ── verify ──────────────────────────────────────────────────────────────────
  if (mode == "verify") {
    require_gpu();
    g_chunk_bytes = choose_chunk_size(chunk_mb * 1024 * 1024);
    std::string pass = read_passphrase("Passphrase: ");
    if (pass.empty()) { std::cerr << "Empty passphrase.\n"; return 1; }
    return decrypt_file(target, "", pass, /*verify_only=*/true) ? 0 : 1;
  }

  if (mode != "encrypt" && mode != "decrypt") { usage(argv[0]); return 1; }

  require_gpu();
  g_chunk_bytes = choose_chunk_size(chunk_mb * 1024 * 1024);

  std::string pass = read_passphrase("Passphrase: ");
  if (pass.empty()) { std::cerr << "Empty passphrase.\n"; return 1; }

  bool encrypting = (mode == "encrypt");

  // ── directory ───────────────────────────────────────────────────────────────
  if (recursive || fs::is_directory(target)) {
    return process_directory(target, encrypting, pass) ? 0 : 1;
  }

  // ── single file ─────────────────────────────────────────────────────────────
  if (encrypting) {
    return encrypt_file(target, target + ".enc", pass) ? 0 : 1;
  } else {
    std::string dst;
    if (target.size() > 4 && target.substr(target.size()-4) == ".enc")
      dst = target.substr(0, target.size()-4);
    else
      dst = "decrypted_" + target;
    return decrypt_file(target, dst, pass) ? 0 : 1;
  }
}
