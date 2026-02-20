#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Helper macro for error checking
#define checkErr(err, msg)                                                     \
  if (err != CL_SUCCESS) {                                                     \
    fprintf(stderr, "Error: %s (%d)\n", msg, err);                             \
    return 1;                                                                  \
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
  fseek(fp, 0, SEEK_END);
  size_t size = ftell(fp);
  rewind(fp);
  char *source = malloc(size + 1);
  fread(source, 1, size, fp);
  source[size] = '\0';
  fclose(fp);
  return source;
}

int main(int argc, char **argv) {
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
    param1 = (argc > 4) ? atof(argv[4]) : 1.0f;
  } else if (strcmp(argv[3], "echo") == 0) {
    effect_type = 1;
    param1 = (argc > 4) ? atof(argv[4]) : 4410.0f;
    param2 = (argc > 5) ? atof(argv[5]) : 0.5f;
  } else if (strcmp(argv[3], "lowpass") == 0) {
    effect_type = 2;
    param1 = (argc > 4) ? atof(argv[4]) : 0.5f;
  } else if (strcmp(argv[3], "bitcrush") == 0) {
    effect_type = 3;
    param1 = (argc > 4) ? atof(argv[4]) : 8.0f;
  } else {
    printf("Unknown effect: %s\n", argv[3]);
    return 1;
  }

  // --- 1. Load WAV File ---
  FILE *fp = fopen(argv[1], "rb");
  if (!fp) {
    perror("Failed to open input file");
    return 1;
  }

  WavHeader header;
  fread(&header, sizeof(WavHeader), 1, fp);

  if (header.bitsPerSample != 16) {
    printf("Only 16-bit WAV supported.\n");
    fclose(fp);
    return 1;
  }

  int numSamples = header.subchunk2Size / sizeof(int16_t);
  int16_t *raw_data = malloc(header.subchunk2Size);
  if (!raw_data) {
    perror("Failed to allocate raw_data");
    fclose(fp);
    return 1;
  }
  fread(raw_data, header.subchunk2Size, 1, fp);
  fclose(fp);

  float *h_input = malloc(sizeof(float) * numSamples);
  float *h_output = malloc(sizeof(float) * numSamples);
  if (!h_input || !h_output) {
    perror("Failed to allocate host buffers");
    free(raw_data);
    if (h_input)
      free(h_input);
    if (h_output)
      free(h_output);
    return 1;
  }

  for (int i = 0; i < numSamples; i++)
    h_input[i] = raw_data[i] / 32768.0f;

  // --- 2. OpenCL Setup ---
  cl_int err;
  cl_platform_id platform;
  err = clGetPlatformIDs(1, &platform, NULL);
  checkErr(err, "clGetPlatformIDs");

  cl_device_id device;
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  if (err != CL_SUCCESS) {
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
    checkErr(err, "clGetDeviceIDs");
  }

  cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  checkErr(err, "clCreateContext");

  cl_command_queue queue =
      clCreateCommandQueueWithProperties(context, device, NULL, &err);
  checkErr(err, "clCreateCommandQueueWithProperties");

  char *source = load_kernel_source("kernel.cl");
  if (!source) {
    fprintf(stderr, "Failed to load kernel.cl\n");
    return 1;
  }

  cl_program program =
      clCreateProgramWithSource(context, 1, (const char **)&source, NULL, &err);
  checkErr(err, "clCreateProgramWithSource");
  free(source);

  err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    char log[4096];
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log),
                          log, NULL);
    fprintf(stderr, "Build Error:\n%s\n", log);
    return 1;
  }

  cl_mem d_in = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               sizeof(float) * numSamples, h_input, &err);
  checkErr(err, "clCreateBuffer (d_in)");
  cl_mem d_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                sizeof(float) * numSamples, NULL, &err);
  checkErr(err, "clCreateBuffer (d_out)");

  cl_kernel kernel = clCreateKernel(program, "apply_effects", &err);
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

  FILE *out_fp = fopen(argv[2], "wb");
  if (!out_fp) {
    perror("Failed to open output file");
  } else {
    fwrite(&header, sizeof(WavHeader), 1, out_fp);
    fwrite(raw_data, header.subchunk2Size, 1, out_fp);
    fclose(out_fp);
    printf("Effect '%s' applied. Output saved to %s\n", argv[3], argv[2]);
  }

  // Cleanup
  free(raw_data);
  free(h_input);
  free(h_output);
  clReleaseMemObject(d_in);
  clReleaseMemObject(d_out);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  return 0;
}
