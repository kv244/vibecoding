#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <malloc.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CLFX_VERSION "1.0.1"

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

#define MAX_EFFECTS 16

typedef struct {
  int type;
  float p1;
  float p2;
  char *ir_file;
} EffectConfig;

// Simple Radix-2 FFT for IR preparation
void host_fft(float *real, float *imag, int n) {
  for (int i = 0, j = 0; i < n; i++) {
    if (i < j) {
      float tr = real[i];
      real[i] = real[j];
      real[j] = tr;
      float ti = imag[i];
      imag[i] = imag[j];
      imag[j] = ti;
    }
    int m = n >> 1;
    while (m >= 1 && j >= m) {
      j -= m;
      m >>= 1;
    }
    j += m;
  }
  for (int len = 2; len <= n; len <<= 1) {
    float ang = -2.0f * 3.14159265f / (float)len;
    float wlen_r = cosf(ang), wlen_i = sinf(ang);
    for (int i = 0; i < n; i += len) {
      float w_r = 1.0f, w_i = 0.0f;
      for (int j = 0; j < len / 2; j++) {
        int idxA = i + j, idxB = i + j + len / 2;
        float u_r = real[idxA], u_i = imag[idxA];
        float v_r = real[idxB] * w_r - imag[idxB] * w_i;
        float v_i = real[idxB] * w_i + imag[idxB] * w_r;
        real[idxA] = u_r + v_r;
        imag[idxA] = u_i + v_i;
        real[idxB] = u_r - v_r;
        imag[idxB] = u_i - v_i;
        float next_w_r = w_r * wlen_r - w_i * wlen_i;
        w_i = w_r * wlen_i + w_i * wlen_r;
        w_r = next_w_r;
      }
    }
  }
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

int is_effect_name(const char *name) {
  const char *effects[] = {
      "gain",     "echo",   "lowpass", "bitcrush",   "tremolo",  "widening",
      "pingpong", "chorus", "autowah", "distortion", "ringmod",  "pitch",
      "gate",     "pan",    "eq",      "freeze",     "convolve", "compress",
      "reverb",   "flange", "phase",   "--mix"};
  int num_effects_list = sizeof(effects) / sizeof(effects[0]);
  for (int i = 0; i < num_effects_list; i++) {
    if (strcmp(name, effects[i]) == 0)
      return 1;
  }
  return 0;
}

int main(int argc, char *argv[]) {
  printf("--- CLFX Audio Engine v%s ---\n", CLFX_VERSION);
  int ret = 0;
  FILE *fp = NULL, *out_fp = NULL;
  int16_t *raw_data = NULL;
  float *h_input = NULL, *h_output = NULL, *h_ir = NULL;
  char *source = NULL;

  cl_platform_id platform = NULL;
  cl_device_id device = NULL;
  cl_context context = NULL;
  cl_command_queue queue = NULL;
  cl_program program = NULL;
  cl_kernel kernel = NULL;
  cl_mem d_in = NULL, d_out = NULL, d_ir = NULL, d_ir_dummy = NULL;
  cl_mem d_initial_in = NULL, d_initial_out = NULL;
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
    printf("  autowah <depth> <rate>       Autowah (depth 0-1, rate 0-1)\n");
    printf("  distortion <drive>      (e.g., distortion 5.0)\n");
    printf("  ringmod <freq>          (e.g., ringmod 440)\n");
    printf("  pitch <ratio>           (e.g., pitch 1.5)\n");
    printf("  gate <threshold> <red>  (e.g., gate 0.1 0.0)\n");
    printf("  pan <value>             (e.g., pan -0.5 [L..R])\n");
    printf(
        "  eq <center_f> <gain>         Spectral EQ (center 0-1, gain 0-10)\n");
    printf("  freeze <amount> <random>     Spectral Freeze (amount 0-1, random "
           "0-1)\n");
    printf("  convolve <ir_file>           Convolution Reverb (requires WAV IR "
           "file)\n");
    printf("  compress <threshold> <ratio> Dynamics Compressor (thresh 0-1, "
           "ratio 1-20)\n");
    printf("  reverb <size> <dry/wet>      Algorithmic Reverb (size 0-1, mix "
           "0-1)\n");
    printf("  flange <depth> <feedback>    Flanger effect\n");
    printf("  phase <depth> <rate>         Phaser effect\n");
    printf("\nOptions:\n");
    printf("  --visualize             Display ASCII waveform\n");
    printf("  --info                  Print system and OpenCL information\n");
    return 1;
  }

  // Handle --info early
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--info") == 0) {
      printf("CLFX Engine Version: %s\n", CLFX_VERSION);
      printf("OS: ");
#ifdef _WIN32
      printf("Windows\n");
#elif __APPLE__
      printf("macOS\n");
#elif __linux__
      printf("Linux\n");
#else
      printf("Unknown\n");
#endif

      cl_uint n_platforms;
      if (clGetPlatformIDs(0, NULL, &n_platforms) == CL_SUCCESS &&
          n_platforms > 0) {
        cl_platform_id *plat_list =
            malloc(sizeof(cl_platform_id) * n_platforms);
        clGetPlatformIDs(n_platforms, plat_list, NULL);
        for (cl_uint p = 0; p < n_platforms; p++) {
          char p_name[128], p_vendor[128], p_ver[128];
          clGetPlatformInfo(plat_list[p], CL_PLATFORM_NAME, sizeof(p_name),
                            p_name, NULL);
          clGetPlatformInfo(plat_list[p], CL_PLATFORM_VENDOR, sizeof(p_vendor),
                            p_vendor, NULL);
          clGetPlatformInfo(plat_list[p], CL_PLATFORM_VERSION, sizeof(p_ver),
                            p_ver, NULL);
          printf("Platform[%u]: %s (%s) %s\n", p, p_name, p_vendor, p_ver);

          cl_uint n_devices;
          if (clGetDeviceIDs(plat_list[p], CL_DEVICE_TYPE_ALL, 0, NULL,
                             &n_devices) == CL_SUCCESS) {
            cl_device_id *dev_list = malloc(sizeof(cl_device_id) * n_devices);
            clGetDeviceIDs(plat_list[p], CL_DEVICE_TYPE_ALL, n_devices,
                           dev_list, NULL);
            for (cl_uint d = 0; d < n_devices; d++) {
              char d_name[128], d_ver[128];
              clGetDeviceInfo(dev_list[d], CL_DEVICE_NAME, sizeof(d_name),
                              d_name, NULL);
              clGetDeviceInfo(dev_list[d], CL_DEVICE_VERSION, sizeof(d_ver),
                              d_ver, NULL);
              printf("  Device[%u]: %s [%s]\n", d, d_name, d_ver);
            }
            free(dev_list);
          }
        }
        free(plat_list);
      } else {
        printf("OpenCL: No platforms found or Error.\n");
      }
      return 0;
    }
  }

  int do_visualize = 0;
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--visualize") == 0) {
      do_visualize = 1;
      break;
    }
  }

  EffectConfig chain[MAX_EFFECTS];
  int num_effects = 0;
  float global_mix = 1.0f;

  int arg_idx = 3;
  while (arg_idx < argc && num_effects < MAX_EFFECTS) {
    if (strcmp(argv[arg_idx], "--mix") == 0) {
      if (arg_idx + 1 < argc) {
        global_mix = atof(argv[++arg_idx]);
      }
      arg_idx++;
      continue;
    }
    if (strcmp(argv[arg_idx], "--visualize") == 0) {
      arg_idx++;
      continue;
    }

    const char *effect_name = argv[arg_idx];
    EffectConfig *cfg = &chain[num_effects];
    cfg->type = -1;
    cfg->p1 = 0.0f;
    cfg->p2 = 0.0f;
    cfg->ir_file = NULL;

    if (strcmp(effect_name, "gain") == 0) {
      cfg->type = 0;
      if (arg_idx + 1 < argc && !is_effect_name(argv[arg_idx + 1]))
        cfg->p1 = atof(argv[++arg_idx]);
      else
        cfg->p1 = 1.0f;
    } else if (strcmp(effect_name, "echo") == 0) {
      cfg->type = 1;
      if (arg_idx + 1 < argc && !is_effect_name(argv[arg_idx + 1]))
        cfg->p1 = atof(argv[++arg_idx]);
      else
        cfg->p1 = 4410.0f;
      if (arg_idx + 1 < argc && !is_effect_name(argv[arg_idx + 1]))
        cfg->p2 = atof(argv[++arg_idx]);
      else
        cfg->p2 = 0.5f;
    } else if (strcmp(effect_name, "lowpass") == 0) {
      cfg->type = 2;
      if (arg_idx + 1 < argc && !is_effect_name(argv[arg_idx + 1]))
        cfg->p1 = atof(argv[++arg_idx]);
      else
        cfg->p1 = 0.5f;
    } else if (strcmp(effect_name, "bitcrush") == 0) {
      cfg->type = 3;
      float bits = 8.0f;
      if (arg_idx + 1 < argc && !is_effect_name(argv[arg_idx + 1]))
        bits = atof(argv[++arg_idx]);
      cfg->p1 = powf(2.0f, roundf(bits));
    } else if (strcmp(effect_name, "tremolo") == 0) {
      cfg->type = 4;
      if (arg_idx + 1 < argc && !is_effect_name(argv[arg_idx + 1]))
        cfg->p1 = atof(argv[++arg_idx]);
      else
        cfg->p1 = 5.0f;
      if (arg_idx + 1 < argc && !is_effect_name(argv[arg_idx + 1]))
        cfg->p2 = atof(argv[++arg_idx]);
      else
        cfg->p2 = 0.5f;
    } else if (strcmp(effect_name, "widening") == 0) {
      cfg->type = 5;
      if (arg_idx + 1 < argc && !is_effect_name(argv[arg_idx + 1]))
        cfg->p1 = atof(argv[++arg_idx]);
      else
        cfg->p1 = 1.5f;
    } else if (strcmp(effect_name, "pingpong") == 0) {
      cfg->type = 6;
      if (arg_idx + 1 < argc && !is_effect_name(argv[arg_idx + 1]))
        cfg->p1 = atof(argv[++arg_idx]);
      else
        cfg->p1 = 8820.0f;
      if (arg_idx + 1 < argc && !is_effect_name(argv[arg_idx + 1]))
        cfg->p2 = atof(argv[++arg_idx]);
      else
        cfg->p2 = 0.5f;
    } else if (strcmp(effect_name, "chorus") == 0) {
      cfg->type = 7;
    } else if (strcmp(effect_name, "autowah") == 0) {
      cfg->type = 8;
      if (arg_idx + 1 < argc && !is_effect_name(argv[arg_idx + 1]))
        cfg->p1 = atof(argv[++arg_idx]);
      else
        cfg->p1 = 0.5f;
      if (arg_idx + 1 < argc && !is_effect_name(argv[arg_idx + 1]))
        cfg->p2 = atof(argv[++arg_idx]);
      else
        cfg->p2 = 0.2f;
    } else if (strcmp(effect_name, "distortion") == 0) {
      cfg->type = 9;
      if (arg_idx + 1 < argc && !is_effect_name(argv[arg_idx + 1]))
        cfg->p1 = atof(argv[++arg_idx]);
      else
        cfg->p1 = 2.0f;
    } else if (strcmp(effect_name, "ringmod") == 0) {
      cfg->type = 10;
      if (arg_idx + 1 < argc && !is_effect_name(argv[arg_idx + 1]))
        cfg->p1 = atof(argv[++arg_idx]);
      else
        cfg->p1 = 440.0f;
    } else if (strcmp(effect_name, "pitch") == 0) {
      cfg->type = 11;
      if (arg_idx + 1 < argc && !is_effect_name(argv[arg_idx + 1]))
        cfg->p1 = atof(argv[++arg_idx]);
      else
        cfg->p1 = 1.5f;
    } else if (strcmp(effect_name, "gate") == 0) {
      cfg->type = 12;
      if (arg_idx + 1 < argc && !is_effect_name(argv[arg_idx + 1]))
        cfg->p1 = atof(argv[++arg_idx]);
      else
        cfg->p1 = 0.1f;
      if (arg_idx + 1 < argc && !is_effect_name(argv[arg_idx + 1]))
        cfg->p2 = atof(argv[++arg_idx]);
      else
        cfg->p2 = 0.0f;
    } else if (strcmp(effect_name, "pan") == 0) {
      cfg->type = 13;
      if (arg_idx + 1 < argc && !is_effect_name(argv[arg_idx + 1]))
        cfg->p1 = atof(argv[++arg_idx]);
      else
        cfg->p1 = 0.0f;
    } else if (strcmp(effect_name, "eq") == 0) {
      cfg->type = 14;
      if (arg_idx + 1 < argc && !is_effect_name(argv[arg_idx + 1]))
        cfg->p1 = atof(argv[++arg_idx]);
      else
        cfg->p1 = 0.5f;
      if (arg_idx + 1 < argc && !is_effect_name(argv[arg_idx + 1]))
        cfg->p2 = atof(argv[++arg_idx]);
      else
        cfg->p2 = 1.0f;
    } else if (strcmp(effect_name, "freeze") == 0) {
      cfg->type = 15;
      if (arg_idx + 1 < argc && !is_effect_name(argv[arg_idx + 1]))
        cfg->p1 = atof(argv[++arg_idx]);
      else
        cfg->p1 = 0.5f;
      if (arg_idx + 1 < argc && !is_effect_name(argv[arg_idx + 1]))
        cfg->p2 = atof(argv[++arg_idx]);
      else
        cfg->p2 = 0.0f;
    } else if (strcmp(effect_name, "convolve") == 0) {
      cfg->type = 16;
      if (arg_idx + 1 < argc && !is_effect_name(argv[arg_idx + 1]))
        cfg->ir_file = argv[++arg_idx];
    } else if (strcmp(effect_name, "compress") == 0) {
      cfg->type = 17;
      if (arg_idx + 1 < argc && !is_effect_name(argv[arg_idx + 1]))
        cfg->p1 = atof(argv[++arg_idx]);
      else
        cfg->p1 = 0.5f;
      if (arg_idx + 1 < argc && !is_effect_name(argv[arg_idx + 1]))
        cfg->p2 = atof(argv[++arg_idx]);
      else
        cfg->p2 = 4.0f;
    } else if (strcmp(effect_name, "reverb") == 0) {
      cfg->type = 18;
      if (arg_idx + 1 < argc && !is_effect_name(argv[arg_idx + 1]))
        cfg->p1 = atof(argv[++arg_idx]);
      else
        cfg->p1 = 0.6f;
      if (arg_idx + 1 < argc && !is_effect_name(argv[arg_idx + 1]))
        cfg->p2 = atof(argv[++arg_idx]);
      else
        cfg->p2 = 0.5f;
    } else if (strcmp(effect_name, "flange") == 0) {
      cfg->type = 19;
      if (arg_idx + 1 < argc && !is_effect_name(argv[arg_idx + 1]))
        cfg->p1 = atof(argv[++arg_idx]);
      else
        cfg->p1 = 0.5f;
      if (arg_idx + 1 < argc && !is_effect_name(argv[arg_idx + 1]))
        cfg->p2 = atof(argv[++arg_idx]);
      else
        cfg->p2 = 0.7f;
    } else if (strcmp(effect_name, "phase") == 0) {
      cfg->type = 20;
      if (arg_idx + 1 < argc && !is_effect_name(argv[arg_idx + 1]))
        cfg->p1 = atof(argv[++arg_idx]);
      else
        cfg->p1 = 0.5f;
      if (arg_idx + 1 < argc && !is_effect_name(argv[arg_idx + 1]))
        cfg->p2 = atof(argv[++arg_idx]);
      else
        cfg->p2 = 0.2f;
    } else {
      fprintf(stderr, "Error: Unknown effect '%s' in chain.\n", effect_name);
      return 1;
    }

    if (cfg->type != -1) {
      num_effects++;
    }
    arg_idx++;
  }

  if (num_effects == 0) {
    printf("No valid effects found in chain.\n");
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
  if (!platforms) {
    fprintf(stderr, "Failed to allocate platforms array.\n");
    ret = 1;
    goto cleanup;
  }
  err = clGetPlatformIDs(num_platforms, platforms, NULL);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "clGetPlatformIDs failed.\n");
    free(platforms);
    ret = 1;
    goto cleanup;
  }

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

  // Pre-bake TILE_SIZE if any spectral effect is in the chain
  size_t local_block_size = 64;
  for (int i = 0; i < num_effects; i++) {
    if (chain[i].type >= 14) { // Spectral effects start from type 14 (EQ)
      local_block_size = 256;
      break;
    }
  }

  char build_options[256];
  sprintf(build_options, "-DTILE_SIZE=%zu", local_block_size);
  err = clBuildProgram(program, 1, &device, build_options, NULL, NULL);
  if (err != CL_SUCCESS) {
    size_t log_size;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL,
                          &log_size);
    char *log = (char *)malloc(log_size);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log,
                          NULL);
    fprintf(stderr, "clBuildProgram failed: %s\n", log);
    free(log);
    ret = 1;
    goto cleanup;
  }

  kernel = clCreateKernel(program, "apply_effects", &err);
  checkErr(err, "clCreateKernel");

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

  // Create ping-pong buffers
  cl_mem_flags in_out_flags = unified
                                  ? (CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR)
                                  : (CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR);
  cl_mem_flags out_only_flags =
      unified ? (CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR) : CL_MEM_READ_WRITE;

  d_in = clCreateBuffer(context, in_out_flags, sizeof(float) * paddedSamples,
                        h_input, &err);
  d_out = clCreateBuffer(context, out_only_flags, sizeof(float) * paddedSamples,
                         (unified ? h_output : NULL), &err);
  d_initial_in = d_in;
  d_initial_out = d_out;
  checkErr(err, "clCreateBuffer (d_in/d_out)");

  err = clGetKernelWorkGroupInfo(kernel, device,
                                 CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                                 sizeof(size_t), &local_block_size, NULL);
  if (err != CL_SUCCESS)
    local_block_size = 64;
  printf("Device preferred workgroup size: %zu\n", local_block_size);

  int samplesTotal = (int)paddedSamples;
  float sRate = (float)header.sampleRate;
  int numChans = (int)header.numChannels;

  // Load IR file if any effect requires it
  // This assumes only one IR file can be used across the chain for simplicity
  // If multiple convolve effects with different IRs were needed, this logic
  // would need to be more complex
  char *ir_filename = NULL;
  for (int i = 0; i < num_effects; i++) {
    if (chain[i].type == 16 && chain[i].ir_file) {
      ir_filename = chain[i].ir_file;
      break;
    }
  }

  if (ir_filename) {
    char cache_name[256];
    snprintf(cache_name, 256, "%s.clir", ir_filename);

    int cache_loaded = 0;
    FILE *fcache = fopen(cache_name, "rb");
    if (fcache) {
      h_ir = (float *)port_aligned_alloc(sizeof(float) * 2048, 4096);
      if (h_ir && fread(h_ir, sizeof(float), 2048, fcache) == 2048) {
        printf("Loaded IR from cache: %s\n", cache_name);
        cache_loaded = 1;
      } else {
        if (h_ir)
          port_aligned_free(h_ir);
        h_ir = NULL;
      }
      fclose(fcache);
    }

    if (!cache_loaded) {
      FILE *fir = fopen(ir_filename, "rb");
      if (fir) {
        WavHeader ir_h;
        if (fread(&ir_h, sizeof(WavHeader), 1, fir) != 1) {
          fprintf(stderr, "Failed to read IR header from %s\n", ir_filename);
          fclose(fir);
          ret = 1;
          goto cleanup;
        }
        int ir_samples = (ir_h.subchunk2Size / sizeof(int16_t));
        if (ir_samples <= 0) {
          fprintf(stderr, "IR file contains no samples\n");
          fclose(fir);
          ret = 1;
          goto cleanup;
        }
        ir_samples = ir_samples > 1024
                         ? 1024
                         : ir_samples; // Cap for this simplified version

        int16_t *ir_raw = malloc(ir_samples * sizeof(int16_t));
        if (!ir_raw) {
          perror("Failed to allocate ir_raw");
          fclose(fir);
          ret = 1;
          goto cleanup;
        }
        if (fread(ir_raw, sizeof(int16_t), ir_samples, fir) != ir_samples) {
          fprintf(stderr, "Failed to read IR audio data\n");
          free(ir_raw);
          fclose(fir);
          ret = 1;
          goto cleanup;
        }
        fclose(fir);

        h_ir = (float *)port_aligned_alloc(sizeof(float) * 2048, 4096);
        if (!h_ir) {
          perror("Failed to allocate h_ir for FFT");
          free(ir_raw);
          ret = 1;
          goto cleanup;
        }
        float *real = (float *)malloc(sizeof(float) * 1024);
        float *imag = (float *)malloc(sizeof(float) * 1024);
        if (!real || !imag) {
          perror("Failed to allocate real/imag for FFT");
          free(ir_raw);
          free(real);
          free(imag);
          port_aligned_free(h_ir);
          h_ir = NULL;
          ret = 1;
          goto cleanup;
        }

        for (int i = 0; i < 1024; i++) {
          real[i] = (i < ir_samples) ? (ir_raw[i] / 32768.0f) : 0.0f;
          imag[i] = 0.0f;
        }
        host_fft(real, imag, 1024);
        for (int i = 0; i < 1024; i++) {
          h_ir[i * 2] = real[i];
          h_ir[i * 2 + 1] = imag[i];
        }
        free(real);
        free(imag);
        free(ir_raw);

        fcache = fopen(cache_name, "wb");
        if (fcache) {
          fwrite(h_ir, sizeof(float), 2048, fcache);
          fclose(fcache);
          printf("Saved IR cache: %s\n", cache_name);
        }
        cache_loaded = 1;
      } else {
        perror("Failed to open IR file for processing");
        ret = 1;
        goto cleanup;
      }
    }

    if (cache_loaded) {
      d_ir = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            sizeof(float) * 2048, h_ir, &err);
      checkErr(err, "clCreateBuffer (d_ir)");
    }
    if (d_ir == NULL) {
      float zero = 0.0f;
      d_ir_dummy =
          clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                         sizeof(float), &zero, &err);
      checkErr(err, "clCreateBuffer (d_ir_dummy)");
      err = clSetKernelArg(kernel, 8, sizeof(cl_mem), &d_ir_dummy);
    } else {
      err = clSetKernelArg(kernel, 8, sizeof(cl_mem), &d_ir);
    }
    checkErr(err, "clSetKernelArg (d_ir)");
  } else {
    // If no convolve effect, still need to set arg 8 to SOMETHING
    float zero = 0.0f;
    d_ir_dummy =
        clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       sizeof(float), &zero, &err);
    checkErr(err, "clCreateBuffer (d_ir_dummy)");
    err = clSetKernelArg(kernel, 8, sizeof(cl_mem), &d_ir_dummy);
    checkErr(err, "clSetKernelArg (d_ir_dummy fallback)");
  }

  size_t global_size = paddedSamples / 4;
  err = clSetKernelArg(kernel, 5, sizeof(int), &samplesTotal);
  err |= clSetKernelArg(kernel, 6, sizeof(float), &sRate);
  err |= clSetKernelArg(kernel, 7, sizeof(int), &numChans);
  err |= clSetKernelArg(kernel, 9, sizeof(float), &global_mix);
  checkErr(err, "clSetKernelArg (common)");

  // --- 2. Processing Loop for Chaining ---
  for (int i = 0; i < num_effects; i++) {
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_in);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_out);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &chain[i].type);
    err |= clSetKernelArg(kernel, 3, sizeof(float), &chain[i].p1);
    err |= clSetKernelArg(kernel, 4, sizeof(float), &chain[i].p2);
    checkErr(err, "clSetKernelArg (loop)");

    size_t current_global = global_size;
    if (chain[i].type >= 14) {
      current_global = ((current_global + 255) / 256) * 256;
    }

    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &current_global,
                                 &local_block_size, 0, NULL, NULL);
    checkErr(err, "clEnqueueNDRangeKernel");

    // Ping-pong swap
    cl_mem tmp = d_in;
    d_in = d_out;
    d_out = tmp;
  }

  // Final result is in d_in (due to the swap after the last kernel call)
  // --- 4. Read Result ---
  // The final result is in d_in due to the swap at the end of the loop
  if (unified) {
    void *mapped_ptr =
        clEnqueueMapBuffer(queue, d_in, CL_TRUE, CL_MAP_READ, 0,
                           sizeof(float) * numSamples, 0, NULL, NULL, &err);
    checkErr(err, "clEnqueueMapBuffer (final)");
    // If d_in is not already using h_output, we must copy.
    // If it is using it (odd swap), memcpy to self is harmless or we can skip.
    if (mapped_ptr != h_output) {
      memcpy(h_output, mapped_ptr, sizeof(float) * numSamples);
    }
    clEnqueueUnmapMemObject(queue, d_in, mapped_ptr, 0, NULL, NULL);
  } else {
    err =
        clEnqueueReadBuffer(queue, d_in, CL_TRUE, 0, sizeof(float) * numSamples,
                            h_output, 0, NULL, NULL);
    checkErr(err, "clEnqueueReadBuffer (final)");
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
  } else {
    perror("Failed to open output file for writing");
    ret = 1;
    goto cleanup;
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
  if (h_ir)
    port_aligned_free(h_ir);
  if (d_initial_in)
    clReleaseMemObject(d_initial_in);
  if (d_initial_out)
    clReleaseMemObject(d_initial_out);
  if (d_ir)
    clReleaseMemObject(d_ir);
  if (d_ir_dummy)
    clReleaseMemObject(d_ir_dummy);
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
