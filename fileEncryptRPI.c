/*
 * VIBEVAULT v12.5: Final Rescue & Logic Alignment
 * VERSION: v12.5-H (Hardened)
 *
 * COMMAND SUMMARY:
 *   Encryption:  ./vibevault enc <file/dir> [--key <key_dir>] [--rescue]
 *   Decryption:  ./vibevault dec <file/dir> [--key <key_dir>] [--rescue]
 *   Key Info:    ./vibevault --keyinfo <key_dir>
 *   Check File:  ./vibevault --check <file>
 *   Rescue Mode: Append --rescue to enc/dec commands.
 *                (Uses static emergency salt. Required for data recovery.)
 *
 * COMPILE: gcc -O3 -mcpu=native -pthread encrypt2.c -o vibevault -lcrypto
 */

#define VERSION "v12.5-H"

#define _GNU_SOURCE
#include <arm_neon.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <openssl/evp.h>
#include <openssl/hmac.h>
#include <openssl/kdf.h>
#include <openssl/sha.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syslog.h>
#include <time.h>
#include <unistd.h>
#include <termios.h>
#include <libgen.h>

// --- CONSTANTS ---
#define MAGIC_BYTE_0 0x41
#define MAGIC_BYTE_1 0x42
#define MAGIC_BYTE_2 0x49
#define MAGIC_BYTE_3 0x56
#define KEY_SIZE 32
#define HMAC_SIZE 32
#define SALT_SIZE 16
#define ITERATIONS 600000
#define NAME_SIZE 256
#define HEADER_SIZE (48 + NAME_SIZE)
#define CHUNK_SIZE (16 * 1024 * 1024)

#define SECURE_LOCK(p, l) mlock(p, l)
#define SECURE_ZERO(p, l) explicit_bzero(p, l)

// --- RANDOMNESS ---
int get_random_bytes(uint8_t *buf, size_t len) {
  int fd = open("/dev/urandom", O_RDONLY);
  if (fd < 0) {
    perror("[!] Critical: Failed to access entropy source");
    return -1;
  }
  size_t total = 0;
  while (total < len) {
    ssize_t r = read(fd, buf + total, len - total);
    if (r < 0) {
      if (errno == EINTR) continue;
      close(fd);
      return -1;
    }
    if (r == 0) break;
    total += r;
  }
  close(fd);
  return (total == len) ? 0 : -1;
}

// --- STRUCTURES ---
typedef struct {
  unsigned long total_files;
  unsigned long skipped_files;
  unsigned long rejected_files;
  unsigned long auth_failures;
  unsigned long failed_files;
  unsigned long total_bytes;
  unsigned long processed_bytes;
  pthread_mutex_t lock;
} GlobalStats;

typedef struct {
  uint8_t aes_raw[KEY_SIZE];
  uint8_t hmac_raw[HMAC_SIZE];
  uint8x16_t rkeys[15];
} KeySet;

typedef struct {
  uint8_t *data;
  size_t len;
  uint8x16_t *round_keys;
  uint8_t nonce[8];
  uint64_t block_offset;
} Task;

typedef struct {
  pthread_t *threads;
  Task **current_tasks;
  pthread_mutex_t mutex;
  pthread_cond_t cond;
  pthread_cond_t done_cond;
  int num_workers;
  int stop;
  int active_workers;
} ThreadPool;

// --- GLOBALS (DEFINED FIRST) ---
GlobalStats stats = {.lock = PTHREAD_MUTEX_INITIALIZER};
ThreadPool pool;

void read_passphrase(const char *prompt, char *buf, size_t size) {
  struct termios old_t, new_t;
  printf("%s", prompt);
  fflush(stdout);
  if (tcgetattr(STDIN_FILENO, &old_t) != 0) {
    perror("tcgetattr");
    exit(1);
  }
  new_t = old_t;
  new_t.c_lflag &= ~ECHO;
  if (tcsetattr(STDIN_FILENO, TCSAFLUSH, &new_t) != 0) {
    perror("tcsetattr");
    exit(1);
  }
  if (fgets(buf, size, stdin) == NULL) buf[0] = '\0';
  buf[strcspn(buf, "\n")] = 0;
  tcsetattr(STDIN_FILENO, TCSAFLUSH, &old_t);
  printf("\n");
}

void print_progress() {
  pthread_mutex_lock(&stats.lock);
  if (stats.total_bytes == 0) {
    pthread_mutex_unlock(&stats.lock);
    return;
  }
  double pct = (double)stats.processed_bytes * 100.0 / stats.total_bytes;
  int bar_len = 30;
  int pos = bar_len * pct / 100.0;
  printf("\r[");
  for (int i = 0; i < bar_len; ++i) {
    if (i < pos)
      printf("=");
    else if (i == pos)
      printf(">");
    else
      printf(" ");
  }
  printf("] %.1f%%", pct);
  fflush(stdout);
  pthread_mutex_unlock(&stats.lock);
}

unsigned long get_total_size(const char *dir) {
  unsigned long size = 0;
  DIR *d = opendir(dir);
  if (!d) {
    struct stat st;
    if (lstat(dir, &st) == 0 && S_ISREG(st.st_mode))
      return st.st_size;
    return 0;
  }
  struct dirent *e;
  while ((e = readdir(d))) {
    if (e->d_name[0] == '.')
      continue;
    char p[4096];
    snprintf(p, 4096, "%s/%s", dir, e->d_name);
    struct stat s;
    if (lstat(p, &s) == 0) {
      if (S_ISDIR(s.st_mode))
        size += get_total_size(p);
      else if (S_ISREG(s.st_mode))
        size += s.st_size;
    }
  }
  closedir(d);
  return size;
}

static const uint8_t sbox[256] = {
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b,
    0xfe, 0xd7, 0xab, 0x76, 0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0,
    0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0, 0xb7, 0xfd, 0x93, 0x26,
    0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2,
    0xeb, 0x27, 0xb2, 0x75, 0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0,
    0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84, 0x53, 0xd1, 0x00, 0xed,
    0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f,
    0x50, 0x3c, 0x9f, 0xa8, 0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5,
    0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2, 0xcd, 0x0c, 0x13, 0xec,
    0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14,
    0xde, 0x5e, 0x0b, 0xdb, 0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c,
    0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79, 0xe7, 0xc8, 0x37, 0x6d,
    0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f,
    0x4b, 0xbd, 0x8b, 0x8a, 0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e,
    0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e, 0xe1, 0xf8, 0x98, 0x11,
    0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f,
    0xb0, 0x54, 0xbb, 0x16};

// --- CRYPTO KERNELS ---

void expand_key_aes256(const uint8_t *key, uint8x16_t *rkeys) {
  uint32_t temp[60];
  for (int i = 0; i < 8; i++)
    memcpy(&temp[i], key + (i * 4), 4); // Fix: Safe unaligned access
  uint32_t rcon = 0x01000000;
  for (int i = 8; i < 60; i++) {
    uint32_t t = temp[i - 1];
    if (i % 8 == 0) {
      t = (sbox[(t >> 16) & 0xff] << 24) | (sbox[(t >> 8) & 0xff] << 16) |
          (sbox[t & 0xff] << 8) | sbox[t >> 24];
      t ^= rcon;
      rcon = (rcon << 1) ^ (rcon & 0x80000000 ? 0x1b : 0);
    } else if (i % 8 == 4) {
      t = (sbox[(t >> 24) & 0xff] << 24) | (sbox[(t >> 16) & 0xff] << 16) |
          (sbox[(t >> 8) & 0xff] << 8) | sbox[t & 0xff];
    }
    temp[i] = temp[i - 8] ^ t;
  }
  for (int i = 0; i < 15; i++)
    rkeys[i] = vld1q_u8((uint8_t *)&temp[i * 4]);
}

static inline uint8x16_t aes_encrypt_block(uint8x16_t b, uint8x16_t *k) {
  b = veorq_u8(b, k[0]); // Round 0: Initial AddRoundKey
  for (int i = 1; i < 14; i++) {
    b = vaeseq_u8(b, vdupq_n_u8(0));
    b = vaesmcq_u8(b);
    b = veorq_u8(b, k[i]);
  }
  b = vaeseq_u8(b, vdupq_n_u8(0));
  return veorq_u8(b, k[14]);
}

// 4-Way Interleaved AES-CTR ASM Kernel
// Optimizes pipeline utilization by processing 4 blocks concurrently.
static inline void aes_ctr_4way_asm(uint8_t *data, const uint8x16_t *rk,
                                    const uint8_t *nonce, uint64_t b_off) {
  uint8x16_t c0, c1, c2, c3, r;
  uint64_t n_val;
  memcpy(&n_val, nonce, 8);

  // Initial Counter Setup
  c0 = vcombine_u8(vcreate_u8(n_val), vcreate_u8(__builtin_bswap64(b_off)));
  c1 = vcombine_u8(vcreate_u8(n_val), vcreate_u8(__builtin_bswap64(b_off + 1)));
  c2 = vcombine_u8(vcreate_u8(n_val), vcreate_u8(__builtin_bswap64(b_off + 2)));
  c3 = vcombine_u8(vcreate_u8(n_val), vcreate_u8(__builtin_bswap64(b_off + 3)));

  // Initial XOR with Round Key 0
  c0 = veorq_u8(c0, rk[0]);
  c1 = veorq_u8(c1, rk[0]);
  c2 = veorq_u8(c2, rk[0]);
  c3 = veorq_u8(c3, rk[0]);

  // Rounds 1-13 (Interleaved AESE + AESMC)
  for (int i = 1; i < 14; i++) {
    r = rk[i];
    c0 = vaeseq_u8(c0, vdupq_n_u8(0));
    c0 = vaesmcq_u8(c0);
    c0 = veorq_u8(c0, r);
    c1 = vaeseq_u8(c1, vdupq_n_u8(0));
    c1 = vaesmcq_u8(c1);
    c1 = veorq_u8(c1, r);
    c2 = vaeseq_u8(c2, vdupq_n_u8(0));
    c2 = vaesmcq_u8(c2);
    c2 = veorq_u8(c2, r);
    c3 = vaeseq_u8(c3, vdupq_n_u8(0));
    c3 = vaesmcq_u8(c3);
    c3 = veorq_u8(c3, r);
  }

  // Round 14 (Last Round)
  r = rk[14];
  c0 = vaeseq_u8(c0, vdupq_n_u8(0));
  c0 = veorq_u8(c0, r);
  c1 = vaeseq_u8(c1, vdupq_n_u8(0));
  c1 = veorq_u8(c1, r);
  c2 = vaeseq_u8(c2, vdupq_n_u8(0));
  c2 = veorq_u8(c2, r);
  c3 = vaeseq_u8(c3, vdupq_n_u8(0));
  c3 = veorq_u8(c3, r);

  // XOR with Data and Store
  vst1q_u8(data + 0, veorq_u8(vld1q_u8(data + 0), c0));
  vst1q_u8(data + 16, veorq_u8(vld1q_u8(data + 16), c1));
  vst1q_u8(data + 32, veorq_u8(vld1q_u8(data + 32), c2));
  vst1q_u8(data + 48, veorq_u8(vld1q_u8(data + 48), c3));
}

// --- THREADING ---

void *worker_loop(void *arg) {
  int id = (int)(uintptr_t)arg;
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(id % pool.num_workers, &cpuset);
  pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
  while (1) {
    pthread_mutex_lock(&pool.mutex);
    while (!pool.stop && pool.current_tasks[id] == NULL)
      pthread_cond_wait(&pool.cond, &pool.mutex);
    if (pool.stop) {
      pthread_mutex_unlock(&pool.mutex);
      break;
    }
    Task *t = pool.current_tasks[id];
    pthread_mutex_unlock(&pool.mutex);
    size_t i = 0;
    size_t local_done = 0;
    for (; i + 64 <= t->len; i += 64) {
      aes_ctr_4way_asm(t->data + i, t->round_keys, t->nonce,
                       t->block_offset + (i / 16));
      local_done += 64;
      if (local_done >= 1024 * 1024) {
        pthread_mutex_lock(&stats.lock);
        stats.processed_bytes += local_done;
        pthread_mutex_unlock(&stats.lock);
        print_progress();
        local_done = 0;
      }
    }
    for (; i + 16 <= t->len; i += 16) {
      uint64_t ctr[2] = {0, __builtin_bswap64(t->block_offset + (i / 16))};
      memcpy(ctr, t->nonce, 8);
      vst1q_u8(t->data + i, veorq_u8(vld1q_u8(t->data + i),
                                     aes_encrypt_block(vld1q_u8((uint8_t *)ctr),
                                                       t->round_keys)));
      local_done += 16;
    }
    if (i < t->len) {
      uint64_t ctr[2] = {0, __builtin_bswap64(t->block_offset + (i / 16))};
      memcpy(ctr, t->nonce, 8);
      uint8_t tmp[16];
      vst1q_u8(tmp, aes_encrypt_block(vld1q_u8((uint8_t *)ctr), t->round_keys));
      for (size_t j = 0; i + j < t->len; j++)
        t->data[i + j] ^= tmp[j];
      local_done += (t->len - i);
    }
    pthread_mutex_lock(&stats.lock);
    stats.processed_bytes += local_done;
    pthread_mutex_unlock(&stats.lock);
    print_progress();
    pthread_mutex_lock(&pool.mutex);
    pool.current_tasks[id] = NULL;
    pool.active_workers--;
    pthread_cond_signal(&pool.done_cond);
    pthread_mutex_unlock(&pool.mutex);
  }
  return NULL;
}

void pool_init(int num_workers) {
  pool.num_workers = num_workers;
  pool.threads = malloc(sizeof(pthread_t) * num_workers);
  pool.current_tasks = calloc(num_workers, sizeof(Task *));
  if (!pool.threads || !pool.current_tasks) {
    perror("[!] Pool allocation failed");
    exit(1);
  }
  pthread_mutex_init(&pool.mutex, NULL);
  pthread_cond_init(&pool.cond, NULL);
  pthread_cond_init(&pool.done_cond, NULL);
  pool.stop = 0;
  pool.active_workers = 0;
  for (int i = 0; i < num_workers; i++) {
    if (pthread_create(&pool.threads[i], NULL, worker_loop,
                       (void *)(uintptr_t)i) != 0) {
      perror("[!] Thread creation failed");
      exit(1);
    }
  }
}

void pool_shutdown() {
  pthread_mutex_lock(&pool.mutex);
  pool.stop = 1;
  pthread_cond_broadcast(&pool.cond);
  pthread_mutex_unlock(&pool.mutex);
  for (int i = 0; i < pool.num_workers; i++)
    pthread_join(pool.threads[i], NULL);
  pthread_mutex_destroy(&pool.mutex);
  pthread_cond_destroy(&pool.cond);
  pthread_cond_destroy(&pool.done_cond);
  free(pool.threads);
  free(pool.current_tasks);
}

void dispatch_work(uint8_t *buf, size_t size, uint8x16_t *rkeys, uint8_t *nonce,
                   uint64_t base_offset) {
  if (size == 0)
    return;

  // Optimization: Process small chunks (headers, small files) synchronously
  // to avoid thread pool overhead. Threshold: 64KB.
  if (size < 65536) {
    size_t i = 0;
    for (; i + 64 <= size; i += 64) {
      aes_ctr_4way_asm(buf + i, rkeys, nonce, base_offset + (i / 16));
    }
    for (; i + 16 <= size; i += 16) {
      uint64_t ctr[2] = {0, __builtin_bswap64(base_offset + (i / 16))};
      memcpy(ctr, nonce, 8);
      vst1q_u8(buf + i, veorq_u8(vld1q_u8(buf + i),
                                 aes_encrypt_block(vld1q_u8((uint8_t *)ctr), rkeys)));
    }
    if (i < size) {
      uint64_t ctr[2] = {0, __builtin_bswap64(base_offset + (i / 16))};
      memcpy(ctr, nonce, 8);
      uint8_t tmp[16];
      vst1q_u8(tmp, aes_encrypt_block(vld1q_u8((uint8_t *)ctr), rkeys));
      for (size_t j = 0; i + j < size; j++)
        buf[i + j] ^= tmp[j];
    }
    pthread_mutex_lock(&stats.lock);
    stats.processed_bytes += size;
    pthread_mutex_unlock(&stats.lock);
    print_progress();
    return;
  }

  size_t part = (size / 16 / pool.num_workers) * 16;
  if (part == 0)
    part = size; // Single worker for small data

  int active = 0;
  for (int i = 0; i < pool.num_workers; i++) {
    if (i * part < size)
      active++;
    else
      break;
  }

  Task *tasks = calloc(active, sizeof(Task));
  if (!tasks) {
    perror("[!] Task allocation failed");
    return;
  }
  pthread_mutex_lock(&pool.mutex);
  pool.active_workers = active;
  for (int i = 0; i < active; i++) {
    tasks[i].data = buf + (i * part);
    tasks[i].len = (i == active - 1) ? (size - (i * part)) : part;
    tasks[i].round_keys = rkeys;
    memcpy(tasks[i].nonce, nonce, 8);
    tasks[i].block_offset = base_offset + (i * part) / 16;
    pool.current_tasks[i] = &tasks[i];
  }
  pthread_cond_broadcast(&pool.cond);
  while (pool.active_workers > 0)
    pthread_cond_wait(&pool.done_cond, &pool.mutex);
  pthread_mutex_unlock(&pool.mutex);
  free(tasks);
}

// --- FILE OPERATIONS ---

void process_file_atomic(const char *path, KeySet *ks, int encrypt) {
  char path_copy[4096];
  uint8_t *buffer = NULL;
  HMAC_CTX *hctx = NULL;
  int src_fd = -1;
  int dst_fd = -1;
  char tmp_path[4096] = {0};
  char final_path[4096] = {0};

  strncpy(path_copy, path, 4095);
  char *bname = basename(path_copy);

  // Strict filtering: Only skip exact matches or specific suffixes
  if (strcmp(bname, "vault.ky") == 0)
    return;
  size_t path_len = strlen(path);
  if (path_len > 4 && (strcmp(path + path_len - 4, ".tmp") == 0 || strcmp(path + path_len - 4, ".old") == 0))
    return;

  // Fix: TOCTOU & Symlink Attack Prevention
  // Open with O_NOFOLLOW first, then check the file descriptor
  src_fd = open(path, O_RDONLY | O_NOFOLLOW);
  if (src_fd < 0) {
    if (errno != ELOOP) { // ELOOP = It was a symlink
        perror("[!] Failed to open source file");
        pthread_mutex_lock(&stats.lock);
        stats.failed_files++;
        pthread_mutex_unlock(&stats.lock);
    }
    return;
  }

  struct stat st;
  if (fstat(src_fd, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size == 0) {
    close(src_fd);
    return;
  }

  snprintf(tmp_path, 4096, "%s.tmp", path);
  // Fix: Use O_EXCL to prevent symlink hijacking (CWE-362)
  // If the file exists, fail securely rather than overwriting.
  dst_fd = open(tmp_path, O_RDWR | O_CREAT | O_EXCL, 0600);
  if (dst_fd < 0) {
    if (errno == EEXIST) {
        fprintf(stderr, "[!] Security: Temp file exists, skipping: %s\n", tmp_path);
    }
    close(src_fd);
    return;
  }
  
  // Safety: Prevent underflow on corrupt files
  if (!encrypt && st.st_size < HEADER_SIZE) {
      close(src_fd); close(dst_fd); unlink(tmp_path); return;
  }

  buffer = malloc(CHUNK_SIZE);
  if (!buffer) {
    perror("[!] Memory allocation failed");
    close(src_fd); close(dst_fd); unlink(tmp_path); return;
  }

  hctx = HMAC_CTX_new();
  if (!hctx) {
    perror("[!] HMAC context failed");
    free(buffer); close(src_fd); close(dst_fd); unlink(tmp_path); return;
  }
  HMAC_Init_ex(hctx, ks->hmac_raw, HMAC_SIZE, EVP_sha256(), NULL);

  uint8_t nonce[8];
  char name_buf[NAME_SIZE];
  char dir_buf[4096];

  if (encrypt) {
    if (get_random_bytes(nonce, 8) != 0) {
      fprintf(stderr, "[!] Entropy failure\n");
      goto fail;
    }

    // Prepare Header
    uint8_t header[HEADER_SIZE];
    memset(header, 0, HEADER_SIZE);
    header[0] = MAGIC_BYTE_0; header[1] = MAGIC_BYTE_1;
    header[2] = MAGIC_BYTE_2; header[3] = MAGIC_BYTE_3;
    uint16_t v = 1, a = 1;
    memcpy(header + 4, &v, 2);
    memcpy(header + 6, &a, 2);
    memcpy(header + 8, nonce, 8);
    // Tag at +16 is already 0 from memset

    // Encrypt Filename
    const char *base_name = strrchr(path, '/');
    base_name = base_name ? base_name + 1 : path;
    strncpy((char *)(header + 48), base_name, NAME_SIZE - 1);
    // Encrypt filename part (offset 48, len 256) with block offset 0
    dispatch_work(header + 48, NAME_SIZE, ks->rkeys, nonce, 0);

    // Write Header
    if (write(dst_fd, header, HEADER_SIZE) != HEADER_SIZE) goto fail;

    // HMAC Update Header (Tag field is 0)
    HMAC_Update(hctx, header, HEADER_SIZE);

    // Process Body
    ssize_t r;
    uint64_t ctr = NAME_SIZE / 16; // Start after filename blocks
    while ((r = read(src_fd, buffer, CHUNK_SIZE)) > 0) {
      dispatch_work(buffer, r, ks->rkeys, nonce, ctr);
      ctr += (r + 15) / 16;
      HMAC_Update(hctx, buffer, r);
      if (write(dst_fd, buffer, r) != r) goto fail;
    }

    // Finalize HMAC and write tag
    uint8_t tag[HMAC_SIZE];
    HMAC_Final(hctx, tag, NULL);
    if (pwrite(dst_fd, tag, HMAC_SIZE, 16) != HMAC_SIZE) goto fail;

  } else { // Decrypt
    uint8_t header[HEADER_SIZE];
    if (read(src_fd, header, HEADER_SIZE) != HEADER_SIZE) goto fail;

    if (!(header[0] == MAGIC_BYTE_0 && header[1] == MAGIC_BYTE_1 &&
          header[2] == MAGIC_BYTE_2 && header[3] == MAGIC_BYTE_3)) {
      syslog(LOG_NOTICE, "{\"event\": \"decryption_rejected\", \"path\": \"%s\"}", path);
      pthread_mutex_lock(&stats.lock);
      stats.rejected_files++;
      pthread_mutex_unlock(&stats.lock);
      goto cleanup_no_fail_count;
    }

    memcpy(nonce, header + 8, 8);
    uint8_t stored_tag[HMAC_SIZE];
    memcpy(stored_tag, header + 16, HMAC_SIZE);

    // Verify HMAC (Pass 1)
    uint8_t header_copy[HEADER_SIZE];
    memcpy(header_copy, header, HEADER_SIZE);
    memset(header_copy + 16, 0, HMAC_SIZE);
    HMAC_Update(hctx, header_copy, HEADER_SIZE);

    ssize_t r;
    while ((r = read(src_fd, buffer, CHUNK_SIZE)) > 0) {
      HMAC_Update(hctx, buffer, r);
    }
    uint8_t calc_tag[HMAC_SIZE];
    HMAC_Final(hctx, calc_tag, NULL);

    if (CRYPTO_memcmp(stored_tag, calc_tag, HMAC_SIZE) != 0) {
      syslog(LOG_WARNING, "{\"event\": \"auth_failure\", \"path\": \"%s\"}", path);
      pthread_mutex_lock(&stats.lock);
      stats.auth_failures++;
      pthread_mutex_unlock(&stats.lock);
      goto cleanup_no_fail_count;
    }

    // Decrypt Filename
    dispatch_work(header + 48, NAME_SIZE, ks->rkeys, nonce, 0);
    strncpy(name_buf, (char *)(header + 48), NAME_SIZE - 1);
    name_buf[NAME_SIZE - 1] = '\0';

    if (strchr(name_buf, '/')) {
      syslog(LOG_CRIT, "{\"event\": \"path_traversal_attempt\", \"path\": \"%s\"}", path);
      goto fail;
    }

    // Decrypt Body (Pass 2)
    if (lseek(src_fd, HEADER_SIZE, SEEK_SET) != HEADER_SIZE) goto fail;
    uint64_t ctr = NAME_SIZE / 16;
    while ((r = read(src_fd, buffer, CHUNK_SIZE)) > 0) {
      dispatch_work(buffer, r, ks->rkeys, nonce, ctr);
      ctr += (r + 15) / 16;
      if (write(dst_fd, buffer, r) != r) goto fail;
    }
  }

  close(src_fd);
  close(dst_fd);
  free(buffer);
  HMAC_CTX_free(hctx);

  // ... Rename logic ...
  if (encrypt) {
    char hash_hex[65];
    uint8_t rnd_name[32];
    get_random_bytes(rnd_name, 32);
    for(int i=0; i<32; i++) sprintf(hash_hex + (i*2), "%02x", rnd_name[i]);
    hash_hex[64] = 0;
    strncpy(dir_buf, path, 4095);
    char *last_slash = strrchr(dir_buf, '/');
    if (last_slash) *last_slash = '\0';
    if (last_slash) snprintf(final_path, 4096, "%s/%s.vault", dir_buf, hash_hex);
    else snprintf(final_path, 4096, "%s.vault", hash_hex);
  } else {
    strncpy(dir_buf, path, 4095);
    char *last_slash = strrchr(dir_buf, '/');
    if (last_slash) *last_slash = '\0';
    if (last_slash) snprintf(final_path, 4096, "%s/%s", dir_buf, name_buf);
    else snprintf(final_path, 4096, "%s", name_buf);
  }

  char old_p[4096];
  snprintf(old_p, 4096, "%s.old", path);
  if (rename(path, old_p) == 0) {
    if (rename(tmp_path, final_path) == 0) {
      unlink(old_p);
      pthread_mutex_lock(&stats.lock);
      stats.total_files++;
      pthread_mutex_unlock(&stats.lock);
      printf("[+] Success: %s\n", final_path);
    } else {
      rename(old_p, path);
      unlink(tmp_path);
    }
  } else {
    // Rename of source failed, clean up temp file
    unlink(tmp_path);
  }
  return;

fail:
    pthread_mutex_lock(&stats.lock);
    stats.failed_files++;
    pthread_mutex_unlock(&stats.lock);
cleanup_no_fail_count:
  if (src_fd >= 0) close(src_fd);
  if (dst_fd >= 0) close(dst_fd);
  if (buffer) free(buffer);
  if (hctx) HMAC_CTX_free(hctx);
  unlink(tmp_path);
}

void walk_dir(const char *dir, KeySet *ks, int enc) {
  DIR *d = opendir(dir);
  if (!d)
    return;
  struct dirent *e;
  while ((e = readdir(d))) {
    if (e->d_name[0] == '.')
      continue;
    char p[4096];
    snprintf(p, 4096, "%s/%s", dir, e->d_name);
    struct stat s;
    lstat(p, &s);
    if (S_ISDIR(s.st_mode))
      walk_dir(p, ks, enc);
    else if (S_ISREG(s.st_mode))
      process_file_atomic(p, ks, enc);
  }
  closedir(d);
}

// --- MAIN ---

int main(int argc, char *argv[]) {
  if (argc < 2)
    return 1;

  uint8_t salt[SALT_SIZE] = {0};
  int rescue_mode = 0;
  char key_dir[4096] = {0};
  // Initialize Enterprise Logging
  openlog("vibevault", LOG_PID | LOG_CONS, LOG_USER);

  if (argc < 3 && strcmp(argv[1], "--keyinfo") != 0) {
      fprintf(stderr, "Usage: %s <enc/dec> <file> [options]\n", argv[0]);
      return 1;
  }

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--rescue") == 0) {
      rescue_mode = 1;
      syslog(LOG_WARNING,
             "{\"event\": \"rescue_mode_active\", \"user\": \"%s\"}",
             getenv("USER"));
    }
    if (strcmp(argv[i], "--key") == 0 && i + 1 < argc)
      strncpy(key_dir, argv[i + 1], 4095);
  }

  if (strcmp(argv[1], "--keyinfo") == 0) {
    if (key_dir[0] == 0 && argc > 2)
      strncpy(key_dir, argv[2], 4095);
    else if (key_dir[0] == 0) {
        fprintf(stderr, "[!] Missing key path\n"); return 1;
    }
    char vf[4096];
    snprintf(vf, 4096, "%s/vault.ky", key_dir);
    FILE *f = fopen(vf, "rb");
    if (!f)
      return 1;
    uint8_t s[SALT_SIZE];
    fread(s, 1, SALT_SIZE, f);
    printf("SALT (Hex): ");
    for (int i = 0; i < SALT_SIZE; i++)
      printf("%02x", s[i]);
    char l[4096];
    while (fgets(l, 4096, f))
      if (strstr(l, "# LEDGER"))
        printf("\n%s", l);
    fclose(f);
    return 0;
  }

  if (strcmp(argv[1], "--check") == 0) {
    if (argc < 3) {
      printf("[!] Usage: ./vibevault --check <file>\n");
      return 1;
    }
    int fd = open(argv[2], O_RDONLY | O_NOFOLLOW);
    if (fd < 0) {
      perror("[!] Failed to open file");
      return 1;
    }
    uint8_t magic[4];
    if (read(fd, magic, 4) != 4) {
      printf("[?] Status: Unknown (File too small or error)\n");
    } else {
      if (magic[0] == MAGIC_BYTE_0 && magic[1] == MAGIC_BYTE_1 &&
          magic[2] == MAGIC_BYTE_2 && magic[3] == MAGIC_BYTE_3) {
        printf("[+] Status: ENCRYPTED (VibeVault Magic Detected)\n");
      } else {
        printf("[-] Status: UNENCRYPTED (Plaintext)\n");
      }
    }
    close(fd);
    return 0;
  }

  if (!rescue_mode) {
    if (key_dir[0] == 0) {
      printf("Key directory: ");
      // Fix: Use fgets to handle paths with spaces and prevent buffer issues
      if (fgets(key_dir, sizeof(key_dir), stdin)) {
        key_dir[strcspn(key_dir, "\n")] = 0;
      }
    }
    char vf[4096];
    snprintf(vf, 4096, "%s/vault.ky", key_dir);
    FILE *f = fopen(vf, "rb");
    if (f) {
      fread(salt, 1, SALT_SIZE, f);
      fclose(f);
    } else if (strcmp(argv[1], "enc") == 0) {
      f = fopen(vf, "wb+");
      if (!f) {
        perror("[!] Failed to create vault key");
        return 1;
      }
      if (get_random_bytes(salt, SALT_SIZE) != 0) {
        fclose(f);
        return 1;
      }
      fwrite(salt, 1, SALT_SIZE, f);
      fflush(f);
      fsync(fileno(f));
      fclose(f);
    } else {
      printf("[!] Key not found.\n");
      return 1;
    }
  } else {
    printf("[!!!] RESCUE MODE: Using Emergency Static Salt\n");
    // Explicit 16-byte static salt (Matches first 16 chars of previous key for compat)
    const uint8_t static_salt[SALT_SIZE] = {
        'E', 'M', 'E', 'R', 'G', 'E', 'N', 'C',
        'Y', '_', 'S', 'A', 'L', 'T', '_', 'V'};
    memcpy(salt, static_salt, SALT_SIZE);
  }

  char up[256];
  SECURE_LOCK(up, sizeof(up));
  read_passphrase("Passphrase: ", up, sizeof(up));

  uint8_t out[KEY_SIZE + HMAC_SIZE];
  KeySet ks;
  SECURE_LOCK(&ks, sizeof(ks));
  SECURE_ZERO(out, sizeof(out));
  SECURE_LOCK(out, sizeof(out));
  PKCS5_PBKDF2_HMAC(up, strlen(up), salt, SALT_SIZE, ITERATIONS, EVP_sha256(),
                    sizeof(out), out);
  memcpy(ks.aes_raw, out, KEY_SIZE);
  memcpy(ks.hmac_raw, out + KEY_SIZE, HMAC_SIZE);
  expand_key_aes256(ks.aes_raw, ks.rkeys);

  SECURE_ZERO(out, sizeof(out));


  stats.total_bytes = get_total_size(argv[2]);
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);

  long cores = sysconf(_SC_NPROCESSORS_ONLN);
  pool_init((cores > 1) ? (int)cores - 1 : 1);

  struct stat s;
  if (stat(argv[2], &s) == 0) {
    if (S_ISDIR(s.st_mode))
      walk_dir(argv[2], &ks, strcmp(argv[1], "enc") == 0);
    else if (S_ISREG(s.st_mode))
      process_file_atomic(argv[2], &ks, strcmp(argv[1], "enc") == 0);
  }

  clock_gettime(CLOCK_MONOTONIC, &end);
  double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
  printf("\n[+] Finished in %.2fs\n", elapsed);

  SECURE_ZERO(up, sizeof(up));
  SECURE_ZERO(&ks, sizeof(ks));
  pool_shutdown();
  closelog();
  return 0;
}
