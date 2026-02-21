#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <malloc.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static inline void *port_aligned_alloc(size_t size, size_t alignment) {
#ifdef _WIN32
  return _aligned_malloc(size, alignment);
#else
  void *ptr = NULL;
  if (posix_memalign(&ptr, alignment, size) != 0)
    return NULL;
  return ptr;
#endif
}

static inline void port_aligned_free(void *ptr) {
#ifdef _WIN32
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}

static inline float clamp(float val, float min, float max) {
  if (val < min)
    return min;
  if (val > max)
    return max;
  return val;
}

// Helper macro for error checking with cleanup
#define checkErr(err, msg)                                                     \
  if (err != CL_SUCCESS) {                                                     \
    fprintf(stderr, "Error: %s (%d)\n", msg, err);                             \
    ret = 1;                                                                   \
    goto cleanup;                                                              \
  }

// Minimal WAV Header (44 bytes)
typedef struct {
  char chunkID[4];
  uint32_t chunkSize;
  char format[4];
  char subchunk1ID[4];
  uint32_t subchunk1Size;
  uint16_t audioFormat;
  uint16_t numChannels;
  uint32_t sampleRate;
  uint32_t byteRate;
  uint16_t blockAlign;
  uint16_t bitsPerSample;
  char subchunk2ID[4];
  uint32_t subchunk2Size;
} WavHeader;

void draw_waveform_ascii(const float *data, int numSamples, int width) {
  if (numSamples <= 0)
    return;
  printf("\nWaveform Visualization (Peak Amplitude):\n");
  int samplesPerCol = numSamples / width;
  if (samplesPerCol < 1)
    samplesPerCol = 1;

  for (int i = 0; i < width && i * samplesPerCol < numSamples; i++) {
    float peak = 0.0f;
    for (int j = 0; j < samplesPerCol; j++) {
      float val = fabsf(data[i * samplesPerCol + j]);
      if (val > peak)
        peak = val;
    }
    int barLen = (int)(peak * 50.0f);
    if (barLen > 50)
      barLen = 50;
    printf("%3d%% |", (int)(peak * 100));
    for (int k = 0; k < barLen; k++)
      printf("#");
    printf("\n");
  }
  printf("\n");
}

char *load_kernel_source(const char *filename) {
  FILE *fp = fopen(filename, "rb");
  if (!fp)
    return NULL;

  if (fseek(fp, 0, SEEK_END) != 0) {
    fclose(fp);
    return NULL;
  }
  long size = ftell(fp);
  if (size < 0) {
    fclose(fp);
    return NULL;
  }
  rewind(fp);

  char *source = malloc(size + 1);
  if (!source) {
    fclose(fp);
    return NULL;
  }

  size_t read_bytes = fread(source, 1, size, fp);
  if (read_bytes != (size_t)size) {
    free(source);
    fclose(fp);
    return NULL;
  }

  source[size] = '\0';
  fclose(fp);
  return source;
}

int main(int argc, char **argv) {
  int ret = 0;
  FILE *fp = NULL, *out_fp = NULL;
  int16_t *raw_data = NULL;
  float *h_input = NULL, *h_output = NULL;
  char *source = NULL;

  cl_platform_id platform = NULL;
  cl_device_id device = NULL;
  cl_context context = NULL;
  cl_command_queue queue = NULL;
  cl_program program = NULL;
  cl_kernel kernel = NULL;
  cl_mem d_in = NULL, d_out = NULL;
  cl_int err;

  if (argc < 4) {
    printf("Usage: %s <input.wav> <output.wav> <effect> [param1] [param2]\n",
           argv[0]);
    printf("Effects:\n");
    printf("  gain <multiplier>       (e.g., gain 1.5)\n");
    printf("  echo <delay_samples> <decay> (e.g., echo 4410 0.5)\n");
    printf("  lowpass <strength>      (e.g., lowpass 0.8)\n");
    printf("  bitcrush <bits>         (e.g., bitcrush 8)\n");
    printf("  tremolo <freq> <depth>  (e.g., tremolo 5.0 0.5)\n");
    printf("  widening <width>        (e.g., widening 1.5) [Stereo Only]\n");
    printf("  pingpong <delay> <decay>(e.g., pingpong 8820 0.5)\n");
    printf("  chorus                  (preset sweep)\n");
    printf("  autowah                 (preset sweep)\n");
    printf("\nOptions:\n");
    printf("  --visualize             Display ASCII waveform\n");
    return 1;
  }

  int do_visualize = 0;
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--visualize") == 0) {
      do_visualize = 1;
      break;
    }
  }

  int effect_type = 0;
  float param1 = 0.0f;
  float param2 = 0.0f;

  if (strcmp(argv[3], "gain") == 0) {
    effect_type = 0;
    param1 = (argc > 4) ? (float)atof(argv[4]) : 1.0f;
  } else if (strcmp(argv[3], "echo") == 0) {
    effect_type = 1;
    param1 = (argc > 4) ? (float)atof(argv[4]) : 4410.0f;
    param2 = (argc > 5) ? (float)atof(argv[5]) : 0.5f;
  } else if (strcmp(argv[3], "lowpass") == 0) {
    effect_type = 2;
    param1 = (argc > 4) ? (float)atof(argv[4]) : 0.5f;
  } else if (strcmp(argv[3], "bitcrush") == 0) {
    effect_type = 3;
    float bits = (argc > 4) ? (float)atof(argv[4]) : 8.0f;
    param1 = powf(2.0f, roundf(bits));
  } else if (strcmp(argv[3], "tremolo") == 0) {
    effect_type = 4;
    param1 = (argc > 4) ? (float)atof(argv[4]) : 5.0f;
    param2 = (argc > 5) ? (float)atof(argv[5]) : 0.5f;
  } else if (strcmp(argv[3], "widening") == 0) {
    effect_type = 5;
    param1 = (argc > 4) ? (float)atof(argv[4]) : 1.5f;
  } else if (strcmp(argv[3], "pingpong") == 0) {
    effect_type = 6;
    param1 = (argc > 4) ? (float)atof(argv[4]) : 8820.0f;
    param2 = (argc > 5) ? (float)atof(argv[5]) : 0.5f;
  } else if (strcmp(argv[3], "chorus") == 0) {
    effect_type = 7;
  } else if (strcmp(argv[3], "autowah") == 0) {
    effect_type = 8;
  } else {
    printf("Unknown effect: %s\n", argv[3]);
    return 1;
  }

  // --- 1. Load WAV File ---
  fp = fopen(argv[1], "rb");
  if (!fp) {
    perror("Failed to open input file");
    return 1;
  }

  WavHeader header;
  if (fread(&header, sizeof(WavHeader), 1, fp) != 1) {
    fprintf(stderr, "Failed to read WAV header\n");
    ret = 1;
    goto cleanup;
  }

  if (header.audioFormat != 1) {
    fprintf(stderr, "Only uncompressed PCM WAV supported (audioFormat=%d)\n",
            header.audioFormat);
    ret = 1;
    goto cleanup;
  }
  if (header.bitsPerSample != 16) {
    fprintf(stderr, "Only 16-bit WAV supported.\n");
    ret = 1;
    goto cleanup;
  }

  printf("Input WAV: %s\n", argv[1]);
  printf("  Channels:    %d (%s)\n", header.numChannels,
         (header.numChannels == 1) ? "Mono" : "Stereo");
  printf("  Sample Rate: %u Hz\n", header.sampleRate);
  printf("  Data Size:   %u bytes\n", header.subchunk2Size);

  int numSamples = header.subchunk2Size / sizeof(int16_t);
  raw_data = malloc(header.subchunk2Size);
  if (!raw_data) {
    perror("Failed to allocate raw_data");
    ret = 1;
    goto cleanup;
  }
  if (fread(raw_data, header.subchunk2Size, 1, fp) != 1) {
    fprintf(stderr, "Failed to read audio data\n");
    ret = 1;
    goto cleanup;
  }
  fclose(fp);
  fp = NULL;

  // --- 2. OpenCL Setup ---
  cl_uint num_platforms;
  err = clGetPlatformIDs(0, NULL, &num_platforms);
  if (err != CL_SUCCESS || num_platforms == 0) {
    fprintf(stderr, "No OpenCL platforms.\n");
    ret = 1;
    goto cleanup;
  }
  cl_platform_id *platforms = malloc(sizeof(cl_platform_id) * num_platforms);
  clGetPlatformIDs(num_platforms, platforms, NULL);

  int found_device = 0;
  // 1. Search for a GPU on any platform
  for (cl_uint i = 0; i < num_platforms; i++) {
    err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err == CL_SUCCESS) {
      platform = platforms[i];
      found_device = 1;
      break;
    }
  }

  // 2. Fallback: Search for a CPU on any platform
  if (!found_device) {
    for (cl_uint i = 0; i < num_platforms; i++) {
      err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU, 1, &device, NULL);
      if (err == CL_SUCCESS) {
        printf("No GPU found, falling back to CPU...\n");
        platform = platforms[i];
        found_device = 1;
        break;
      }
    }
  }
  free(platforms);

  if (!found_device) {
    fprintf(stderr, "No OpenCL devices (GPU or CPU) found.\n");
    ret = 1;
    goto cleanup;
  }

  char device_name[128];
  clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name,
                  NULL);
  // Check for Unified Memory (Zero-copy optimization)
  cl_bool unified;
  clGetDeviceInfo(device, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(unified),
                  &unified, NULL);
  printf("Unified Memory: %s\n", unified ? "Yes" : "No");

  size_t max_wg_size;
  clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_wg_size),
                  &max_wg_size, NULL);
  printf("Max Device Work-Group Size: %zu\n", max_wg_size);

  context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  checkErr(err, "clCreateContext");
  queue = clCreateCommandQueueWithProperties(context, device, NULL, &err);
  checkErr(err, "clCreateCommandQueueWithProperties");

  source = load_kernel_source("kernel.cl");
  if (!source) {
    fprintf(stderr, "Failed to load/read kernel.cl\n");
    ret = 1;
    goto cleanup;
  }
  program =
      clCreateProgramWithSource(context, 1, (const char **)&source, NULL, &err);
  checkErr(err, "clCreateProgramWithSource");
  free(source);
  source = NULL;

  // Initial build to query kernel info
  err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    char build_log[8192];
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                          sizeof(build_log), build_log, NULL);
    fprintf(stderr, "Initial Build Error:\n%s\n", build_log);
    ret = 1;
    goto cleanup;
  }

  kernel = clCreateKernel(program, "apply_effects", &err);
  checkErr(err, "clCreateKernel");

  size_t preferred_wg, kernel_max_wg;
  clGetKernelWorkGroupInfo(kernel, device,
                           CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                           sizeof(preferred_wg), &preferred_wg, NULL);
  clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE,
                           sizeof(kernel_max_wg), &kernel_max_wg, NULL);

  // Determine optimal and safe local_size
  size_t local_block_size = preferred_wg;
  if (local_block_size > kernel_max_wg)
    local_block_size = kernel_max_wg;
  if (local_block_size > 256)
    local_block_size = 256; // Cap for local memory tile size safety

  printf("Preferred WG: %zu, Kernel Max WG: %zu, Chosen WG: %zu\n",
         preferred_wg, kernel_max_wg, local_block_size);

  // Rebuild with TILE_SIZE macro for robustness
  char build_options[64];
  sprintf(build_options, "-DTILE_SIZE=%zu", local_block_size);
  clReleaseKernel(kernel);
  err = clBuildProgram(program, 1, &device, build_options, NULL, NULL);
  checkErr(err, "clBuildProgram (optimized)");

  kernel = clCreateKernel(program, "apply_effects", &err);
  checkErr(err, "clCreateKernel (optimized)");

  // Padding: Align numSamples to local_size * 4 (for float4 vectorization)
  int samplesPerWorkItem = 4;
  size_t alignSize = local_block_size * samplesPerWorkItem;
  int paddedSamples = ((numSamples + alignSize - 1) / alignSize) * alignSize;

  // Aligned allocation for zero-copy
  h_input = port_aligned_alloc(sizeof(float) * paddedSamples, 4096);
  h_output = port_aligned_alloc(sizeof(float) * paddedSamples, 4096);
  if (!h_input || !h_output) {
    perror("port_aligned_alloc failed");
    ret = 1;
    goto cleanup;
  }

  // Initialize with zero (padding) and copy data
  memset(h_input, 0, sizeof(float) * paddedSamples);
  for (int i = 0; i < numSamples; i++)
    h_input[i] = raw_data[i] / 32768.0f;

  cl_mem_flags in_flags = unified ? (CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR)
                                  : (CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR);
  cl_mem_flags out_flags =
      unified ? (CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR) : CL_MEM_WRITE_ONLY;

  d_in = clCreateBuffer(context, in_flags, sizeof(float) * paddedSamples,
                        h_input, &err);
  checkErr(err, "clCreateBuffer (d_in)");
  d_out = clCreateBuffer(context, out_flags, sizeof(float) * paddedSamples,
                         (unified ? h_output : NULL), &err);
  checkErr(err, "clCreateBuffer (d_out)");

  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_in);
  checkErr(err, "clSetKernelArg (d_in)");
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_out);
  checkErr(err, "clSetKernelArg (d_out)");
  err = clSetKernelArg(kernel, 2, sizeof(int), &effect_type);
  checkErr(err, "clSetKernelArg (effect_type)");
  err = clSetKernelArg(kernel, 3, sizeof(float), &param1);
  checkErr(err, "clSetKernelArg (param1)");
  err = clSetKernelArg(kernel, 4, sizeof(float), &param2);
  checkErr(err, "clSetKernelArg (param2)");
  err = clSetKernelArg(kernel, 5, sizeof(int), &numSamples);
  checkErr(err, "clSetKernelArg (numSamples)");
  float fs = (float)header.sampleRate;
  err = clSetKernelArg(kernel, 6, sizeof(float), &fs);
  checkErr(err, "clSetKernelArg (sampleRate)");

  size_t global_size = paddedSamples / 4;
  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size,
                               &local_block_size, 0, NULL, NULL);
  checkErr(err, "clEnqueueNDRangeKernel");

  if (unified) {
    cl_int map_err;
    void *mapped_ptr = clEnqueueMapBuffer(queue, d_out, CL_TRUE, CL_MAP_READ, 0,
                                          sizeof(float) * paddedSamples, 0,
                                          NULL, NULL, &map_err);
    checkErr(map_err, "clEnqueueMapBuffer");
    clEnqueueUnmapMemObject(queue, d_out, mapped_ptr, 0, NULL, NULL);
  } else {
    err = clEnqueueReadBuffer(queue, d_out, CL_TRUE, 0,
                              sizeof(float) * paddedSamples, h_output, 0, NULL,
                              NULL);
    checkErr(err, "clEnqueueReadBuffer");
  }

  // --- 3. Save WAV File ---
  for (int i = 0; i < numSamples; i++) {
    float s = h_output[i] * 32768.0f;
    raw_data[i] = (int16_t)clamp(s, -32768.0f, 32767.0f);
  }

  out_fp = fopen(argv[2], "wb");
  if (out_fp) {
    fwrite(&header, sizeof(WavHeader), 1, out_fp);
    fwrite(raw_data, header.subchunk2Size, 1, out_fp);
    fclose(out_fp);
    out_fp = NULL;
    printf("Safe optimized effect '%s' applied. Saved to %s\n", argv[3],
           argv[2]);
  }

  if (do_visualize) {
    draw_waveform_ascii(h_output, numSamples, 40);
  }

  ret = 0;
cleanup:
  if (fp)
    fclose(fp);
  if (out_fp)
    fclose(out_fp);
  free(raw_data);
  free(source);
  if (h_input)
    port_aligned_free(h_input);
  if (h_output)
    port_aligned_free(h_output);
  if (d_in)
    clReleaseMemObject(d_in);
  if (d_out)
    clReleaseMemObject(d_out);
  if (kernel)
    clReleaseKernel(kernel);
  if (program)
    clReleaseProgram(program);
  if (queue)
    clReleaseCommandQueue(queue);
  if (context)
    clReleaseContext(context);
  return ret;
}
