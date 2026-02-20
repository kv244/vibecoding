/*
 * VIBEVAULT v12.5: Final Rescue & Logic Alignment
 * VERSION: v12.5-H (Hardened)
 * COMPILE: gcc -O3 -mcpu=native -pthread vibevault.c -o vibevault -lcrypto
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
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

// --- CONSTANTS ---
#define MAGIC_BYTE_0 0x41
#define MAGIC_BYTE_1 0x42
#define MAGIC_BYTE_2 0x49
#define MAGIC_BYTE_3 0x56
#define KEY_SIZE 16
#define HMAC_SIZE 32
#define SALT_SIZE 16
#define ITERATIONS 600000
#define HEADER_SIZE 48

// --- SECURITY HELPERS ---
#ifdef __linux__
#include <sys/mman.h>
#define SECURE_ZERO(p, l) explicit_bzero(p, l)
#define SECURE_LOCK(p, l) mlock(p, l)
#else
#define SECURE_ZERO(p, l) memset(p, 0, l)
#define SECURE_LOCK(p, l)
#endif

// --- RANDOMNESS ---
int get_random_bytes(uint8_t *buf, size_t len) {
  int fd = open("/dev/hwrng", O_RDONLY);
  if (fd < 0) {
    fd = open("/dev/urandom", O_RDONLY);
  }
  if (fd < 0) {
    perror("[!] Critical: Failed to access entropy source");
    return -1;
  }
  ssize_t r = read(fd, buf, len);
  close(fd);
  return (r == (ssize_t)len) ? 0 : -1;
}

// --- STRUCTURES ---
typedef struct {
  uint64_t total_files;
  uint64_t auth_failures;
  uint64_t skipped_files;
  uint64_t rejected_files;
  pthread_mutex_t lock;
} GlobalStats;

typedef struct {
  uint8_t aes_raw[KEY_SIZE];
  uint8_t hmac_raw[HMAC_SIZE];
  uint8x16_t rkeys[11];
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
GlobalStats stats = {0, 0, 0, 0, PTHREAD_MUTEX_INITIALIZER};
ThreadPool pool;

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

void expand_key_aes128(const uint8_t *key, uint8x16_t *rkeys) {
  uint32_t temp[44];
  for (int i = 0; i < 4; i++)
    temp[i] = ((uint32_t *)key)[i];
  uint32_t rcon = 0x01000000;
  for (int i = 4; i < 44; i++) {
    uint32_t t = temp[i - 1];
    if (i % 4 == 0) {
      t = (sbox[(t >> 16) & 0xff] << 24) | (sbox[(t >> 8) & 0xff] << 16) |
          (sbox[t & 0xff] << 8) | sbox[t >> 24];
      t ^= rcon;
      rcon = (rcon << 1) ^ (rcon & 0x80000000 ? 0x1b : 0);
    }
    temp[i] = temp[i - 4] ^ t;
  }
  for (int i = 0; i < 11; i++)
    rkeys[i] = vld1q_u8((uint8_t *)&temp[i * 4]);
}

static inline uint8x16_t aes_encrypt_block(uint8x16_t b, uint8x16_t *k) {
  b = veorq_u8(b, k[0]); // Round 0: Initial AddRoundKey
  for (int i = 1; i < 10; i++) {
    b = vaeseq_u8(b, vdupq_n_u8(0)); // SubBytes + ShiftRows
    b = vaesmcq_u8(b);               // MixColumns
    b = veorq_u8(b, k[i]);           // AddRoundKey
  }
  // Round 10: SubBytes + ShiftRows + AddRoundKey (integrated in vaeseq)
  b = vaeseq_u8(b, vdupq_n_u8(0));
  return veorq_u8(b, k[10]);
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
    for (; i + 16 <= t->len; i += 16) {
      uint64_t ctr[2] = {0, t->block_offset + (i / 16)};
      memcpy(ctr, t->nonce, 8);
      vst1q_u8(t->data + i, veorq_u8(vld1q_u8(t->data + i),
                                     aes_encrypt_block(vld1q_u8((uint8_t *)ctr),
                                                       t->round_keys)));
    }
    if (i < t->len) {
      uint64_t ctr[2] = {0, t->block_offset + (i / 16)};
      memcpy(ctr, t->nonce, 8);
      uint8_t tmp[16];
      vst1q_u8(tmp, aes_encrypt_block(vld1q_u8((uint8_t *)ctr), t->round_keys));
      for (size_t j = 0; i + j < t->len; j++)
        t->data[i + j] ^= tmp[j];
    }
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

void dispatch_work(uint8_t *buf, size_t size, uint8x16_t *rkeys,
                   uint8_t *nonce) {
  if (size == 0)
    return;
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
    tasks[i].block_offset = (i * part) / 16;
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
  if (strstr(path, "vault.ky") || strstr(path, ".tmp") || strstr(path, ".old"))
    return;
  struct stat st;
  if (lstat(path, &st) != 0 || S_ISLNK(st.st_mode))
    return; // Safety: Reject symlinks to prevent hijacking

  int src_fd = open(path, O_RDONLY);
  if (src_fd < 0)
    return;
  if (fstat(src_fd, &st) != 0 || st.st_size == 0) {
    close(src_fd);
    return;
  }

  char tmp_path[4096];
  snprintf(tmp_path, 4096, "%s.tmp", path);
  int dst_fd = open(tmp_path, O_RDWR | O_CREAT | O_TRUNC, 0600);
  if (dst_fd < 0) {
    close(src_fd);
    return;
  }
  size_t out_size =
      encrypt ? (st.st_size + HEADER_SIZE) : (st.st_size - HEADER_SIZE);

  if (ftruncate(dst_fd, out_size) != 0) {
    close(src_fd);
    close(dst_fd);
    unlink(tmp_path);
    return;
  }

  void *src_map = mmap(NULL, st.st_size, PROT_READ | (encrypt ? 0 : PROT_WRITE),
                       MAP_PRIVATE, src_fd, 0);
  void *dst_map =
      mmap(NULL, out_size, PROT_READ | PROT_WRITE, MAP_SHARED, dst_fd, 0);

  if (src_map == MAP_FAILED || dst_map == MAP_FAILED) {
    if (src_map != MAP_FAILED)
      munmap(src_map, st.st_size);
    if (dst_map != MAP_FAILED)
      munmap(dst_map, out_size);
    close(src_fd);
    close(dst_fd);
    unlink(tmp_path);
    return;
  }

  uint8_t nonce[8], tag[HMAC_SIZE];
  uint8_t *s = (uint8_t *)src_map;
  uint8_t *d = (uint8_t *)dst_map;

  if (encrypt) {
    if (s[0] == MAGIC_BYTE_0 && s[1] == MAGIC_BYTE_1 && s[2] == MAGIC_BYTE_2 &&
        s[3] == MAGIC_BYTE_3) {
      pthread_mutex_lock(&stats.lock);
      stats.skipped_files++;
      pthread_mutex_unlock(&stats.lock);
      munmap(src_map, st.st_size);
      munmap(dst_map, out_size);
      close(src_fd);
      close(dst_fd);
      unlink(tmp_path);
      return;
    }
    if (get_random_bytes(nonce, 8) != 0) {
      munmap(src_map, st.st_size);
      munmap(dst_map, out_size);
      close(src_fd);
      close(dst_fd);
      unlink(tmp_path);
      return;
    }
    memcpy(d + HEADER_SIZE, s, st.st_size);
    dispatch_work(d + HEADER_SIZE, st.st_size, ks->rkeys, nonce);
    d[0] = MAGIC_BYTE_0;
    d[1] = MAGIC_BYTE_1;
    d[2] = MAGIC_BYTE_2;
    d[3] = MAGIC_BYTE_3;
    uint16_t v = 1, a = 1;
    memcpy(d + 4, &v, 2);
    memcpy(d + 6, &a, 2);
    memcpy(d + 8, nonce, 8);
    // Secure HMAC: Zero the field before calculation
    memset(d + 16, 0, HMAC_SIZE);
    HMAC(EVP_sha256(), ks->hmac_raw, HMAC_SIZE, d + 8, out_size - 8, tag, NULL);
    memcpy(d + 16, tag, HMAC_SIZE);
  } else {
    if (!(s[0] == MAGIC_BYTE_0 && s[1] == MAGIC_BYTE_1 &&
          s[2] == MAGIC_BYTE_2 && s[3] == MAGIC_BYTE_3)) {
      pthread_mutex_lock(&stats.lock);
      stats.rejected_files++;
      pthread_mutex_unlock(&stats.lock);
      munmap(src_map, st.st_size);
      munmap(dst_map, out_size);
      close(src_fd);
      close(dst_fd);
      unlink(tmp_path);
      return;
    }
    memcpy(nonce, s + 8, 8);
    uint8_t calc_tag[HMAC_SIZE];
    uint8_t stored_tag[HMAC_SIZE];
    memcpy(stored_tag, s + 16, HMAC_SIZE);

    // Secure HMAC: Zero the field in the private map before validation
    memset(s + 16, 0, HMAC_SIZE);
    HMAC(EVP_sha256(), ks->hmac_raw, HMAC_SIZE, s + 8, st.st_size - 8, calc_tag,
         NULL);
    // Secure HMAC: Constant-time verification
    if (CRYPTO_memcmp(stored_tag, calc_tag, HMAC_SIZE) != 0) {
      pthread_mutex_lock(&stats.lock);
      stats.auth_failures++;
      pthread_mutex_unlock(&stats.lock);
      munmap(src_map, st.st_size);
      munmap(dst_map, out_size);
      close(src_fd);
      close(dst_fd);
      unlink(tmp_path);
      return;
    }
    memcpy(d, s + HEADER_SIZE, out_size);
    dispatch_work(d, out_size, ks->rkeys, nonce);
  }
  msync(dst_map, out_size, MS_SYNC);
  munmap(src_map, st.st_size);
  munmap(dst_map, out_size);
  close(src_fd);
  close(dst_fd);

  char old_p[4096];
  snprintf(old_p, 4096, "%s.old", path);
  if (rename(path, old_p) == 0) {
    if (rename(tmp_path, path) == 0) {
      unlink(old_p);
      pthread_mutex_lock(&stats.lock);
      stats.total_files++;
      pthread_mutex_unlock(&stats.lock);
      printf("[+] Success: %s\n", path);
    } else
      rename(old_p, path);
  }
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

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--rescue") == 0)
      rescue_mode = 1;
    if (strcmp(argv[i], "--key") == 0 && i + 1 < argc)
      strncpy(key_dir, argv[i + 1], 4095);
  }

  if (strcmp(argv[1], "--keyinfo") == 0) {
    if (key_dir[0] == 0)
      strncpy(key_dir, argv[2], 4095);
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

  if (!rescue_mode) {
    if (key_dir[0] == 0) {
      printf("Key directory: ");
      scanf("%4095s", key_dir);
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
    printf("[!!!] RESCUE MODE: Using Null Salt\n");
  }

  char *pass = getpass("Passphrase: ");
  char up[256];
  SECURE_LOCK(up, sizeof(up));
  strncpy(up, pass, 255);
  up[255] = '\0';

  KeySet ks;
  SECURE_LOCK(&ks, sizeof(ks));
  uint8_t out[KEY_SIZE + HMAC_SIZE];
  PKCS5_PBKDF2_HMAC(up, strlen(up), salt, SALT_SIZE, ITERATIONS, EVP_sha256(),
                    sizeof(out), out);
  memcpy(ks.aes_raw, out, KEY_SIZE);
  memcpy(ks.hmac_raw, out + KEY_SIZE, HMAC_SIZE);
  expand_key_aes128(ks.aes_raw, ks.rkeys);

  SECURE_ZERO(out, sizeof(out));

  if (strcmp(argv[1], "enc") == 0 && !rescue_mode) {
    char vf[4096];
    snprintf(vf, 4096, "%s/vault.ky", key_dir);
    FILE *f = fopen(vf, "a");
    time_t n = time(NULL);
    fprintf(f, "\n# LEDGER | TARGET: %s | DATE: %s", argv[2], ctime(&n));
    fclose(f);
  }

  long cores = sysconf(_SC_NPROCESSORS_ONLN);
  pool_init((cores > 1) ? (int)cores - 1 : 1);

  struct stat s;
  if (stat(argv[2], &s) == 0)
    walk_dir(argv[2], &ks, strcmp(argv[1], "enc") == 0);

  char kvf[4096];
  snprintf(kvf, 4096, "%s/vault.ky", key_dir);

  printf("\n--- VibeVault Summary ---\n");
  printf("Version: %s\n", VERSION);
  printf("Target:  %s\n", argv[2]);
  printf("Vault:   %s\n", kvf);
  printf("Pass:    %s\n", up);
  printf("Files:   %lu | Skipped: %lu | Rejected: %lu | Auth Fails: %lu\n",
         stats.total_files, stats.skipped_files, stats.rejected_files,
         stats.auth_failures);

  SECURE_ZERO(up, sizeof(up));
  SECURE_ZERO(&ks, sizeof(ks));
  pool_shutdown();
  return 0;
}
