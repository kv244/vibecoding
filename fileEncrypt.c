#define _GNU_SOURCE
#define _FILE_OFFSET_BITS 64
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>


#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#define KEY_SIZE 32
#define CHUNK_SIZE (1024 * 1024) // 1MB chunks
#define NUM_THREADS 4            // Adjustable based on board cores

typedef struct {
  char filename[PATH_MAX];
  char out_name[PATH_MAX];
  unsigned char key[KEY_SIZE];
  off_t start_offset;
  off_t length;
  int success; // Thread status flag
} ThreadArgs;

/**
 * RISC-V Optimized XOR Cipher (Modified for key offsets)
 * Optimized for BeagleBoard-V Fire (RISC-V 64)
 */
void xor_buffer_asm(unsigned char *data, size_t data_len, unsigned char *key,
                    size_t key_len, size_t key_start_idx) {
  if (data_len == 0 || key_len == 0)
    return;

  unsigned char *data_end = data + data_len;
  unsigned char *key_ptr = key + (key_start_idx % key_len);
  unsigned char *key_base = key;
  unsigned char *key_end = key + key_len;

  __asm__ __volatile__(
      "1: \n\t"
      "beq %[curr_data], %[data_end], 3f \n\t" // If data_ptr == data_end, exit
      "lbu t0, 0(%[curr_data]) \n\t"           // t0 = *data
      "lbu t1, 0(%[curr_key]) \n\t"            // t1 = *key
      "xor t0, t0, t1 \n\t"                    // t0 = t0 ^ t1
      "sb t0, 0(%[curr_data]) \n\t"            // *data = t0

      "addi %[curr_data], %[curr_data], 1 \n\t" // data++
      "addi %[curr_key], %[curr_key], 1 \n\t"   // key++

      "blt %[curr_key], %[key_end], 2f \n\t" // If key_ptr < key_end, skip reset
      "mv %[curr_key], %[key_start] \n\t"    // Else, key_ptr = key_start (reset
                                             // key)

      "2: \n\t"
      "j 1b \n\t" // Loop
      "3: \n\t"
      : [curr_data] "+r"(data), [curr_key] "+r"(key_ptr)
      : [data_end] "r"(data_end), [key_end] "r"(key_end),
        [key_start] "r"(key_base)
      : "t0", "t1", "memory");
}

#ifdef USE_OPENCL
#include <CL/cl.h>

// OpenCL Global State
cl_context ocl_context;
cl_command_queue ocl_queue;
cl_program ocl_program;
cl_kernel ocl_kernel;

int init_opencl() {
  cl_platform_id platform;
  cl_device_id device;
  cl_int err;

  err = clGetPlatformIDs(1, &platform, NULL);
  if (err != CL_SUCCESS)
    return -1;

  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
  if (err != CL_SUCCESS)
    return -1;

  ocl_context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  ocl_queue = clCreateCommandQueue(ocl_context, device, 0, &err);

  FILE *kf = fopen("kernel.cl", "r");
  if (!kf)
    return -1;
  fseek(kf, 0, SEEK_END);
  size_t ksize = ftell(kf);
  rewind(kf);
  char *ksrc = malloc(ksize + 1);
  if (!ksrc) {
    fclose(kf);
    return -1;
  }
  fread(ksrc, 1, ksize, kf);
  ksrc[ksize] = '\0';
  fclose(kf);

  ocl_program = clCreateProgramWithSource(ocl_context, 1, (const char **)&ksrc,
                                          &ksize, &err);
  free(ksrc);
  err = clBuildProgram(ocl_program, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    char buffer[2048];
    clGetProgramBuildInfo(ocl_program, device, CL_PROGRAM_BUILD_LOG,
                          sizeof(buffer), buffer, NULL);
    fprintf(stderr, "CL Build Error: %s\n", buffer);
    return -1;
  }

  ocl_kernel = clCreateKernel(ocl_program, "xor_cipher", &err);
  return (err == CL_SUCCESS) ? 0 : -1;
}

int run_opencl_xor(unsigned char *data, size_t len, unsigned char *key) {
  cl_int err;
  cl_mem d_buf = clCreateBuffer(
      ocl_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, len, data, &err);
  cl_mem k_buf = clCreateBuffer(
      ocl_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 32, key, &err);

  clSetKernelArg(ocl_kernel, 0, sizeof(cl_mem), &d_buf);
  clSetKernelArg(ocl_kernel, 1, sizeof(cl_mem), &k_buf);
  unsigned int ksize = 32;
  clSetKernelArg(ocl_kernel, 2, sizeof(unsigned int), &ksize);

  size_t global_size = len / 16;
  err = clEnqueueNDRangeKernel(ocl_queue, ocl_kernel, 1, NULL, &global_size,
                               NULL, 0, NULL, NULL);
  clEnqueueReadBuffer(ocl_queue, d_buf, CL_TRUE, 0, len, data, 0, NULL, NULL);

  clReleaseMemObject(d_buf);
  clReleaseMemObject(k_buf);
  return (err == CL_SUCCESS) ? 0 : -1;
}
#endif

void *worker(void *args) {
  ThreadArgs *t_args = (ThreadArgs *)args;
  t_args->success = 0;

  FILE *fin = fopen(t_args->filename, "rb");
  if (!fin) {
    perror("Thread failed to open source file");
    return NULL;
  }

  FILE *fout = fopen(t_args->out_name, "r+b");
  if (!fout) {
    perror("Thread failed to open output file");
    fclose(fin);
    return NULL;
  }

  unsigned char *buffer = NULL;
  if (posix_memalign((void **)&buffer, 16, CHUNK_SIZE) != 0) {
    fprintf(stderr, "Thread memory allocation failed\n");
    fclose(fin);
    fclose(fout);
    return NULL;
  }

  if (fseeko(fin, t_args->start_offset, SEEK_SET) != 0 ||
      fseeko(fout, t_args->start_offset, SEEK_SET) != 0) {
    perror("Thread seek failed");
    goto cleanup;
  }

  off_t remaining = t_args->length;
  off_t current_pos = t_args->start_offset;

  while (remaining > 0) {
    size_t to_read = (remaining > CHUNK_SIZE) ? CHUNK_SIZE : (size_t)remaining;
    size_t bytes_read = fread(buffer, 1, to_read, fin);
    if (bytes_read == 0)
      break;

    size_t key_idx = (size_t)(current_pos % KEY_SIZE);

    int processed = 0;
#ifdef USE_OPENCL
    if (bytes_read % 16 == 0) {
      if (run_opencl_xor(buffer, bytes_read, t_args->key) == 0) {
        processed = 1;
      }
    }
#endif

    if (!processed) {
      xor_buffer_asm(buffer, bytes_read, t_args->key, KEY_SIZE, key_idx);
    }

    if (fwrite(buffer, 1, bytes_read, fout) != bytes_read) {
      perror("Write error");
      goto cleanup;
    }

    remaining -= bytes_read;
    current_pos += bytes_read;
  }

  if (remaining == 0)
    t_args->success = 1;

cleanup:
  free(buffer);
  fclose(fin);
  fclose(fout);
  return NULL;
}

#ifdef USE_OPENCL
int ocl_active = 0;
#endif

int main(int argc, char *argv[]) {
#ifdef USE_OPENCL
  if (init_opencl() == 0) {
    ocl_active = 1;
    printf("[INFO] OpenCL hardware acceleration ACTIVE.\n");
  } else {
    printf("[WARN] OpenCL failed. Using RISC-V ASM optimization.\n");
  }
#endif

  if (argc < 3) {
    printf("Usage: %s [encrypt|decrypt] [filename]\n", argv[0]);
    return 1;
  }

  char *mode = argv[1];
  char *filename = argv[2];
  unsigned char key[KEY_SIZE];
  char enc_name[PATH_MAX];
  char out_name[PATH_MAX];
  char key_name[PATH_MAX];

  if (strcmp(mode, "encrypt") == 0) {
    snprintf(enc_name, sizeof(enc_name), "%s.enc", filename);
    snprintf(key_name, sizeof(key_name), "%s.ky", filename);
    strncpy(out_name, enc_name, PATH_MAX);
  } else if (strcmp(mode, "decrypt") == 0) {
    snprintf(enc_name, sizeof(enc_name), "%s.enc", filename);
    snprintf(key_name, sizeof(key_name), "%s.ky", filename);
    strncpy(out_name, filename, PATH_MAX);
  } else {
    fprintf(stderr, "Invalid mode. Use 'encrypt' or 'decrypt'.\n");
    return 1;
  }

  off_t fsize = 0;
  struct stat st;
  if (stat((strcmp(mode, "encrypt") == 0) ? filename : enc_name, &st) == 0) {
    fsize = st.st_size;
  } else {
    perror("Stat failed");
    return 1;
  }

  if (strcmp(mode, "encrypt") == 0) {
    srand((unsigned int)time(NULL));
    for (int i = 0; i < KEY_SIZE; i++)
      key[i] = (unsigned char)(rand() % 256);

    FILE *fk = fopen(key_name, "wb");
    if (!fk) {
      perror("Key file error");
      return 1;
    }
    fwrite(key, 1, KEY_SIZE, fk);
    fclose(fk);
  } else {
    FILE *fk = fopen(key_name, "rb");
    if (!fk) {
      fprintf(stderr, "Key file %s not found\n", key_name);
      return 1;
    }
    if (fread(key, 1, KEY_SIZE, fk) != KEY_SIZE) {
      fprintf(stderr, "Key read error\n");
      fclose(fk);
      return 1;
    }
    fclose(fk);
  }

  // Pre-create output file
  FILE *fout = fopen(out_name, "wb");
  if (!fout) {
    perror("Output file error");
    return 1;
  }
  if (fsize > 0) {
    if (fseeko(fout, fsize - 1, SEEK_SET) != 0 || fputc(0, fout) == EOF) {
      perror("Pre-allocation failed");
      fclose(fout);
      return 1;
    }
  }
  fclose(fout);

  pthread_t threads[NUM_THREADS];
  ThreadArgs args[NUM_THREADS];
  off_t segment_size = fsize / NUM_THREADS;

  for (int i = 0; i < NUM_THREADS; i++) {
    strncpy(args[i].filename,
            (strcmp(mode, "encrypt") == 0) ? filename : enc_name, PATH_MAX);
    strncpy(args[i].out_name, out_name, PATH_MAX);
    memcpy(args[i].key, key, KEY_SIZE);
    args[i].start_offset = (off_t)i * segment_size;
    args[i].length = (i == NUM_THREADS - 1)
                         ? (fsize - ((off_t)i * segment_size))
                         : segment_size;

    if (pthread_create(&threads[i], NULL, worker, &args[i]) != 0) {
      perror("Thread creation failed");
    }
  }

  int all_success = 1;
  for (int i = 0; i < NUM_THREADS; i++) {
    pthread_join(threads[i], NULL);
    if (!args[i].success)
      all_success = 0;
  }

#ifdef USE_OPENCL
  if (ocl_active) {
    clReleaseKernel(ocl_kernel);
    clReleaseProgram(ocl_program);
    clReleaseCommandQueue(ocl_queue);
    clReleaseContext(ocl_context);
  }
#endif

  if (all_success) {
    printf("Successfully %sed %s using %d threads (%s)\n", mode, filename,
           NUM_THREADS,
#ifdef USE_OPENCL
           ocl_active ? "OPENCL MODE" : "RISC-V ASM FALLBACK"
#else
           "RISC-V ASM MODE"
#endif
    );
  } else {
    fprintf(stderr, "Operation failed.\n");
    return 1;
  }

  return 0;
}
