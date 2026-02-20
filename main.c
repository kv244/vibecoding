#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    printf("  echo <delay_samples> <decay> (e.g., echo 4410 0.5) [One-shot "
           "reflection]\n");
    printf("  lowpass <strength>      (e.g., lowpass 0.8)\n");
    printf("  bitcrush <bits>         (e.g., bitcrush 8)\n");
    return 1;
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
    param1 = (argc > 4) ? (float)atof(argv[4]) : 8.0f;
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
    fprintf(stderr, "Only 16-bit WAV supported (%d bits found).\n",
            header.bitsPerSample);
    ret = 1;
    goto cleanup;
  }

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

  h_input = malloc(sizeof(float) * numSamples);
  h_output = malloc(sizeof(float) * numSamples);
  if (!h_input || !h_output) {
    perror("Failed to allocate host buffers");
    ret = 1;
    goto cleanup;
  }

  for (int i = 0; i < numSamples; i++)
    h_input[i] = raw_data[i] / 32768.0f;

  // --- 2. OpenCL Setup ---
  cl_uint num_platforms;
  err = clGetPlatformIDs(0, NULL, &num_platforms);
  if (err != CL_SUCCESS || num_platforms == 0) {
    fprintf(stderr, "No OpenCL platforms found.\n");
    ret = 1;
    goto cleanup;
  }
  cl_platform_id *platforms = malloc(sizeof(cl_platform_id) * num_platforms);
  clGetPlatformIDs(num_platforms, platforms, NULL);
  platform = platforms[0]; // Just take the first one for now, but logged
  free(platforms);

  char platform_name[128];
  clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(platform_name),
                    platform_name, NULL);
  printf("Using platform: %s\n", platform_name);

  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  if (err != CL_SUCCESS) {
    printf("GPU not found, falling back to CPU...\n");
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
    checkErr(err, "clGetDeviceIDs (CPU)");
  }

  char device_name[128];
  clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name,
                  NULL);
  printf("Using device: %s\n", device_name);

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

  err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    char log[8192];
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log),
                          log, NULL);
    fprintf(stderr, "Build Error:\n%s\n", log);
    ret = 1;
    goto cleanup;
  }

  d_in = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                        sizeof(float) * numSamples, h_input, &err);
  checkErr(err, "clCreateBuffer (d_in)");
  d_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * numSamples,
                         NULL, &err);
  checkErr(err, "clCreateBuffer (d_out)");

  kernel = clCreateKernel(program, "apply_effects", &err);
  checkErr(err, "clCreateKernel");

  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_in);
  checkErr(err, "clSetKernelArg 0");
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_out);
  checkErr(err, "clSetKernelArg 1");
  err = clSetKernelArg(kernel, 2, sizeof(int), &effect_type);
  checkErr(err, "clSetKernelArg 2");
  err = clSetKernelArg(kernel, 3, sizeof(float), &param1);
  checkErr(err, "clSetKernelArg 3");
  err = clSetKernelArg(kernel, 4, sizeof(float), &param2);
  checkErr(err, "clSetKernelArg 4");
  err = clSetKernelArg(kernel, 5, sizeof(int), &numSamples);
  checkErr(err, "clSetKernelArg 5");

  size_t global_size = numSamples;
  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, NULL, 0,
                               NULL, NULL);
  checkErr(err, "clEnqueueNDRangeKernel");

  err =
      clEnqueueReadBuffer(queue, d_out, CL_TRUE, 0, sizeof(float) * numSamples,
                          h_output, 0, NULL, NULL);
  checkErr(err, "clEnqueueReadBuffer");

  // --- 3. Save WAV File ---
  for (int i = 0; i < numSamples; i++) {
    float sample = h_output[i] * 32768.0f;
    if (sample > 32767.0f)
      sample = 32767.0f;
    if (sample < -32768.0f)
      sample = -32768.0f;
    raw_data[i] = (int16_t)sample;
  }

  out_fp = fopen(argv[2], "wb");
  if (!out_fp) {
    perror("Failed to open output file");
    ret = 1;
  } else {
    fwrite(&header, sizeof(WavHeader), 1, out_fp);
    fwrite(raw_data, header.subchunk2Size, 1, out_fp);
    fclose(out_fp);
    out_fp = NULL;
    printf("Effect '%s' applied. Output saved to %s\n", argv[3], argv[2]);
  }

cleanup:
  if (fp)
    fclose(fp);
  if (out_fp)
    fclose(out_fp);
  free(raw_data);
  free(h_input);
  free(h_output);
  free(source);
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
