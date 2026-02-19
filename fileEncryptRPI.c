#define _FILE_OFFSET_BITS 64
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define KEY_SIZE 32
#define NUM_THREADS 4
#define CHUNK_SIZE (4 * 1024 * 1024) // 4MB chunks

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

/* to compile: gcc -O3 fileEncryptRPI.c -lpthread -o fileEncryptRPI */

#include <arm_neon.h>

/**
 * ARM NEON SIMD Optimized XOR
 * Processes 32 bytes per iteration using 128-bit V-registers.
 * Uses safer and more portable NEON intrinsics.
 */
void xor_buffer_neon(unsigned char *data, size_t data_len, unsigned char *key) {
  if (data_len == 0)
    return;

  // Load 32-byte key into two 128-bit NEON registers
  uint8x16_t k0 = vld1q_u8(key);
  uint8x16_t k1 = vld1q_u8(key + 16);

  size_t i = 0;
  // Main loop: Process 32 bytes at a time
  for (; i + 32 <= data_len; i += 32) {
    // Load 32 bytes from data
    uint8x16_t d0 = vld1q_u8(data + i);
    uint8x16_t d1 = vld1q_u8(data + i + 16);

    // XOR with key
    d0 = veorq_u8(d0, k0);
    d1 = veorq_u8(d1, k1);

    // Store back to data
    vst1q_u8(data + i, d0);
    vst1q_u8(data + i + 16, d1);
  }

  // Tail handling
  for (; i < data_len; i++) {
    data[i] ^= key[i % KEY_SIZE];
  }
}

typedef struct {
  unsigned char *data;
  size_t len;
  unsigned char key[KEY_SIZE];
  size_t key_offset; // Relative to 0 in the 32-byte key
} ThreadArgs;

void *thread_worker(void *arg) {
  ThreadArgs *ta = (ThreadArgs *)arg;

  // Rotate key if start is not 32-byte aligned
  unsigned char rotated_key[KEY_SIZE];
  if (ta->key_offset == 0) {
    memcpy(rotated_key, ta->key, KEY_SIZE);
  } else {
    for (int i = 0; i < KEY_SIZE; i++) {
      rotated_key[i] = ta->key[(i + ta->key_offset) % KEY_SIZE];
    }
  }

  xor_buffer_neon(ta->data, ta->len, rotated_key);
  return NULL;
}

/**
 * Raspberry Pi Hardware Entropy Access
 * Attempts to use /dev/hwrng for true hardware entropy first.
 * Falls back to /dev/urandom (which is seeded by HWRNG) if needed.
 * Note: Accessing /dev/hwrng directly may require root privileges.
 */
int get_hw_key(unsigned char *key, size_t len) {
  // Try primary hardware RNG first
  int fd = open("/dev/hwrng", O_RDONLY);
  if (fd >= 0) {
    ssize_t result = read(fd, key, len);
    close(fd);
    if (result == (ssize_t)len)
      return 0;
  }

  // Fallback to urandom
  fd = open("/dev/urandom", O_RDONLY);
  if (fd < 0)
    return -1;
  ssize_t result = read(fd, key, len);
  close(fd);
  return (result == (ssize_t)len) ? 0 : -1;
}

#include <time.h>

int main(int argc, char *argv[]) {
  if (argc < 3) {
    printf("Usage: %s [encrypt|decrypt] [filename]\n", argv[0]);
    return 1;
  }

  char *mode = argv[1];
  char *filename = argv[2];
  int is_encrypt = (strcmp(mode, "encrypt") == 0);
  if (!is_encrypt && strcmp(mode, "decrypt") != 0) {
    fprintf(stderr, "Invalid mode. Use 'encrypt' or 'decrypt'.\n");
    return 1;
  }

  char enc_name[PATH_MAX], key_name[PATH_MAX];
  if (snprintf(enc_name, sizeof(enc_name), "%s.enc", filename) >=
          (int)sizeof(enc_name) ||
      snprintf(key_name, sizeof(key_name), "%s.ky", filename) >=
          (int)sizeof(key_name)) {
    fprintf(stderr, "Filename too long\n");
    return 1;
  }

  unsigned char key[KEY_SIZE];
  FILE *fsrc = NULL, *fout = NULL, *fkey = NULL;
  unsigned char *buffers[2] = {NULL, NULL};
  size_t bytes_read[2] = {0, 0};

  // Double buffering: Use two 32-byte aligned buffers
  for (int i = 0; i < 2; i++) {
    if (posix_memalign((void **)&buffers[i], 32, CHUNK_SIZE) != 0) {
      perror("Aligned memory allocation failed");
      return 1;
    }
  }

  if (is_encrypt) {
    fsrc = fopen(filename, "rb");
    if (!fsrc) {
      perror("Source file error");
      goto cleanup;
    }

    if (get_hw_key(key, KEY_SIZE) != 0) {
      fprintf(stderr, "Failed to generate key from hardware entropy\n");
      goto cleanup;
    }

    fout = fopen(enc_name, "wb");
    if (!fout) {
      perror("Output file error");
      goto cleanup;
    }

    // Secure key file creation: Use O_NOFOLLOW to prevent symlink attacks
    int key_fd = open(key_name, O_WRONLY | O_CREAT | O_TRUNC | O_NOFOLLOW,
                      S_IRUSR | S_IWUSR);
    if (key_fd < 0) {
      perror("Key file security error (symlink suspected?)");
      goto cleanup;
    }
    if (write(key_fd, key, KEY_SIZE) != KEY_SIZE) {
      perror("Key write error");
      close(key_fd);
      goto cleanup;
    }
    close(key_fd);
  } else {
    // Key read: Use O_NOFOLLOW for safety
    int k_fd = open(key_name, O_RDONLY | O_NOFOLLOW);
    if (k_fd < 0) {
      perror("Key file access error (symlink suspected?)");
      goto cleanup;
    }
    if (read(k_fd, key, KEY_SIZE) != KEY_SIZE) {
      perror("Key read error");
      close(k_fd);
      goto cleanup;
    }
    close(k_fd);

    fsrc = fopen(enc_name, "rb");
    if (!fsrc) {
      perror("Encrypted file error");
      goto cleanup;
    }

    fout = fopen(filename, "wb");
    if (!fout) {
      perror("Output file error");
      goto cleanup;
    }
  }

  fseeko(fsrc, 0, SEEK_END);
  off_t total_size = ftello(fsrc);
  rewind(fsrc);

  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);

  off_t processed_bytes = 0;
  int current = 0;

  // Initial read for the first buffer
  bytes_read[current] = fread(buffers[current], 1, CHUNK_SIZE, fsrc);

  while (bytes_read[current] > 0) {
    int next = (current + 1) % 2;
    size_t current_len = bytes_read[current];

    // Pre-read the next chunk while threads process the current one
    // (This is the "Double Buffering" part - overlapping I/O and compute)
    // Note: In a real-world scenario, you might use a separate thread for
    // fread, but here we overlap by reading while the CPU is busy with
    // pthreads. However, since fread is blocking, the best we can do without a
    // dedicated I/O thread is to read the next chunk immediately after starting
    // threads.

    // Actually, to truly overlap, we need to read 'next' while 'current' is
    // being processed. I'll read 'next' into the background if possible,
    // but for now, I'll structure the code to be ready for it.

    // Multithreaded Processing for the current chunk
    pthread_t threads[NUM_THREADS];
    ThreadArgs args[NUM_THREADS];

    size_t total_blocks = current_len / 32;
    size_t blocks_per_thread = total_blocks / NUM_THREADS;
    size_t extra_blocks = total_blocks % NUM_THREADS;
    size_t current_chunk_offset = 0;

    for (int i = 0; i < NUM_THREADS; i++) {
      size_t thread_len;
      if (i == NUM_THREADS - 1) {
        thread_len = current_len - current_chunk_offset;
      } else {
        size_t thread_blocks = blocks_per_thread + (i < extra_blocks ? 1 : 0);
        thread_len = thread_blocks * 32;
      }

      if (thread_len > 0) {
        args[i].data = buffers[current] + current_chunk_offset;
        args[i].len = thread_len;
        args[i].key_offset =
            (processed_bytes + current_chunk_offset) % KEY_SIZE;
        memcpy(args[i].key, key, KEY_SIZE);

        if (pthread_create(&threads[i], NULL, thread_worker, &args[i]) != 0) {
          perror("Thread creation failed");
          goto cleanup;
        }
        current_chunk_offset += thread_len;
      } else {
        args[i].len = 0;
      }
    }

    // While threads are processing 'current', we can't easily fread into 'next'
    // without a separate I/O thread because fread is synchronous.
    // However, we can read the next chunk HERE before joining the threads.
    bytes_read[next] = fread(buffers[next], 1, CHUNK_SIZE, fsrc);

    for (int i = 0; i < NUM_THREADS; i++) {
      if (threads[i])
        pthread_join(threads[i], NULL);
    }

    if (fwrite(buffers[current], 1, current_len, fout) != current_len) {
      perror("Output write error");
      goto cleanup;
    }

    processed_bytes += current_len;
    current = next;
  }

  clock_gettime(CLOCK_MONOTONIC, &end);
  double elapsed =
      (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

  printf("NEON %s-Threaded %s Complete: %s\n",
         NUM_THREADS > 1 ? "Multi" : "Single",
         is_encrypt ? "Encryption" : "Decryption",
         is_encrypt ? enc_name : filename);
  printf("Performance: %.2f MB/s (Elapsed: %.3f sec, Total: %lld bytes)\n",
         (double)total_size / (1024 * 1024) / elapsed, elapsed,
         (long long)total_size);

cleanup:
  for (int i = 0; i < 2; i++) {
    if (buffers[i]) {
      memset(buffers[i], 0, CHUNK_SIZE); // Cleansing
      free(buffers[i]);
    }
  }
  memset(key, 0, KEY_SIZE); // Cleansing
  if (fsrc)
    fclose(fsrc);
  if (fout)
    fclose(fout);
  if (fkey)
    fclose(fkey);
  return 0;
}
