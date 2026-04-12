// TRON-like Maze Simulation with CUDA and OpenGL Visualization

#undef UNICODE
#undef _UNICODE
#include <windows.h>

// CUDA's device-pass compiler chokes on __declspec(dllimport) inside GL headers.
// Suppress __declspec for the device compilation pass only.
#ifdef __CUDACC__
#pragma push_macro("__declspec")
#undef __declspec
#define __declspec(x)
#endif
#include <GL/gl.h>
#include <GL/glu.h>
#ifdef __CUDACC__
#pragma pop_macro("__declspec")
#endif
#include <cuda_gl_interop.h>
// #include <GL/glext.h> // Removed: not available on this system

#include <cmath>
#include <cstdlib>
#include <ctime>
#include <curand_kernel.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <deque>

extern "C" {
    __declspec(dllexport) unsigned long NvOptimusEnablement = 0x00000001;
    __declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
}

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr, "GPU Error: %s in %s at line %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// OpenGL Extension definitions for VBO (if gl.h is too old)
#ifndef GL_SIZEIPTR_DEFINED
#define GL_SIZEIPTR_DEFINED
typedef ptrdiff_t GLsizeiptr;
typedef ptrdiff_t GLintptr;
#endif

#ifndef GL_ARRAY_BUFFER
#define GL_ARRAY_BUFFER 0x8892
#define GL_DYNAMIC_DRAW 0x88E8
#endif

typedef void (APIENTRY * PFNGLGENBUFFERSPROC) (GLsizei n, GLuint *buffers);
typedef void (APIENTRY * PFNGLBINDBUFFERPROC) (GLenum target, GLuint buffer);
typedef void (APIENTRY * PFNGLBUFFERDATAPROC) (GLenum target, GLsizeiptr size, const void *data, GLenum usage);
typedef void (APIENTRY * PFNGLBUFFERSUBDATAPROC) (GLenum target, GLintptr offset, GLsizeiptr size, const void *data);

PFNGLGENBUFFERSPROC glGenBuffers = NULL;
PFNGLBINDBUFFERPROC glBindBuffer = NULL;
PFNGLBUFFERDATAPROC glBufferData = NULL;
PFNGLBUFFERSUBDATAPROC glBufferSubData = NULL;

void init_gl_extensions() {
    glGenBuffers = (PFNGLGENBUFFERSPROC)wglGetProcAddress("glGenBuffers");
    glBindBuffer = (PFNGLBINDBUFFERPROC)wglGetProcAddress("glBindBuffer");
    glBufferData = (PFNGLBUFFERDATAPROC)wglGetProcAddress("glBufferData");
    glBufferSubData = (PFNGLBUFFERSUBDATAPROC)wglGetProcAddress("glBufferSubData");
}

#define MAZE_COLS 256
#define MAZE_ROWS 256
#define CELL_SIZE 300.0f
#define APP_VERSION "v21.0.0-PRO-4090"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define NUM_AGENTS 4096
#define NUM_RAYS 33

static float RADAR_ANGLES[NUM_RAYS];
void init_radar_angles() {
    float field_of_view = 2.1f; // Narrowed to 120 degrees (from 170) for stability
    for (int i = 0; i < NUM_RAYS; ++i) {
        RADAR_ANGLES[i] = -field_of_view/2.0f + (field_of_view / (NUM_RAYS - 1)) * i;
    }
}

// Maze grid in global device memory (scaled for 256x256)
__device__ int *d_maze_grid_ptr;
__constant__ float d_radar_angles[NUM_RAYS];

struct CycleState {
  float x;
  float z;
  float dir;
  float current_speed;
  float target_speed;
  int speed_transition_timer;
  float ai_memory_grid[64][64]; // Reverted to 64x64 to restore L1 cache efficiency (Phase 18.0)
  bool trapped;
  int step_count;
  int stuck_timer;
  float last_ray_distances[NUM_RAYS];
  int best_ray_idx;
  int thinking_id;
  float front_dist;
  float best_dist;
  bool is_scanning;
  int scan_count;
  float crumb_x[16];
  float crumb_z[16];
  int crumb_idx;
  int crumb_count;
  bool is_reversing;
  float dist_since_crumb;
  float scan_dir;
  bool is_deciding;
  int decision_timer;
  float last_junction_x;
  float last_junction_z;
  int active_rays;
  float x_min_recent, x_max_recent;
  float z_min_recent, z_max_recent;
  curandState rng;
};

struct CompactAgent {
  float x, z, dir;
  int thinking_id;
};

// Iterative Maze Generation (Stack-based) to prevent stack overflow on 256x256
void carve_maze(int start_r, int start_c, int *maze) {
  struct Pos { int r, c; };
  std::vector<Pos> stack;
  stack.push_back({start_r, start_c});
  maze[start_r * MAZE_COLS + start_c] = 0;

  while(!stack.empty()) {
    Pos curr = stack.back();
    int r = curr.r;
    int c = curr.c;

    int dirs[4][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
    // Shuffle directions
    for (int i = 3; i > 0; --i) {
      int r_idx = rand() % (i + 1);
      std::swap(dirs[i], dirs[r_idx]);
    }

    bool carved = false;
    for (int i = 0; i < 4; ++i) {
      int dr = dirs[i][0], dc = dirs[i][1];
      int nr = r + dr * 2, nc = c + dc * 2;
      if (nr >= 0 && nr < MAZE_ROWS && nc >= 0 && nc < MAZE_COLS) {
        if (maze[nr * MAZE_COLS + nc] == 1) {
          maze[(r + dr) * MAZE_COLS + (c + dc)] = 0;
          maze[nr * MAZE_COLS + nc] = 0;
          stack.push_back({nr, nc});
          carved = true;
          break;
        }
      }
    }
    if (!carved) stack.pop_back();
  }
}

__device__ bool is_wall(float wx, float wz, float offsetX, float offsetZ) {
  int c = (int)floorf((wx + offsetX) / CELL_SIZE);
  int r = (int)floorf((wz + offsetZ) / CELL_SIZE);
  if (c < 0 || c >= MAZE_COLS || r < 0 || r >= MAZE_ROWS)
    return true;
  return d_maze_grid_ptr[r * MAZE_COLS + c] == 1;
}

__global__ void init_cycles(CycleState *cycles, unsigned long long seed) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= NUM_AGENTS) return;
  CycleState &cycle = cycles[tid];
  
  // Initialize RNG and reset execution state
  curand_init(seed, tid, 0, &cycle.rng);
  
  cycle.trapped = false;
  cycle.step_count = 0;
  cycle.stuck_timer = 0;
  cycle.is_reversing = false;
  cycle.is_deciding = false;

  cycle.x_min_recent = cycle.x_max_recent = 0.0f; // Will be set on first movement
  cycle.z_min_recent = cycle.z_max_recent = 0.0f;

  // Explicitly clear memory grid
  for (int r = 0; r < 64; r++) {
    for (int c = 0; c < 64; c++) {
      cycle.ai_memory_grid[r][c] = 0.0f;
    }
  }
}

__global__ void pack_compact_agents(CycleState *cycles, CompactAgent *compact, float3 *vbo_ptr) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < NUM_AGENTS) {
    compact[tid].x = cycles[tid].x;
    compact[tid].z = cycles[tid].z;
    compact[tid].dir = cycles[tid].dir;
    compact[tid].thinking_id = cycles[tid].thinking_id;
    
    // Write directly to VBO if provided
    if (vbo_ptr) {
        vbo_ptr[tid] = make_float3(cycles[tid].x, 5.0f, cycles[tid].z);
    }
  }
}

__global__ void decay_memory(CycleState *cycles) {
    int agent_idx = blockIdx.x;
    if (agent_idx >= NUM_AGENTS) return;
    int row = threadIdx.x; // 64 threads per block, one per row
    if (row >= 64) return;
    
    CycleState &cycle = cycles[agent_idx];
    for (int c = 0; c < 64; c++) {
        cycle.ai_memory_grid[row][c] *= 0.999f;
    }
}

__global__ void step_cycle(CycleState *cycles, float offsetX, float offsetZ) {
  int tid = threadIdx.x; // NUM_RAYS (33) threads
  int cycle_idx = blockIdx.x; 
  CycleState &cycle = cycles[cycle_idx];
  
  __shared__ float s_ray_distances[NUM_RAYS];
  __shared__ float s_ray_penalties[NUM_RAYS];

  int curr_c = (int)floorf((cycle.x + offsetX) / CELL_SIZE);
  int curr_r = (int)floorf((cycle.z + offsetZ) / CELL_SIZE);
  
  // Parallel Ray Casting
  if (tid < NUM_RAYS) {
    float ray_dir = cycle.dir + d_radar_angles[tid];
    float dx = sinf(ray_dir);
    float dz = cosf(ray_dir);
    float dist = 0.0f;
    float mem_penalty = 0.0f;
    float max_ray = 4000.0f;
    int samples = 0;
    
    // Proximity Catch: Check immediate vicinity (5.0u) to catch walls the bike is "kissing"
    if (is_wall(cycle.x + dx * 5.0f, cycle.z + dz * 5.0f, offsetX, offsetZ)) {
        dist = 5.0f;
    } else {
      while (dist < max_ray) {
        float rx = cycle.x + dx * dist;
        float rz = cycle.z + dz * dist;
        if (is_wall(rx, rz, offsetX, offsetZ)) break;
      
      int mc = (int)floorf((rx + offsetX) / CELL_SIZE);
      int mr = (int)floorf((rz + offsetZ) / CELL_SIZE);
      
      // Memory Map: Scale global coordinates to localized 64x64 grid (4x4 maze cells)
      int local_c = (mc / (MAZE_COLS / 64)) % 64;
      int local_r = (mr / (MAZE_ROWS / 64)) % 64;
      if (local_c >= 0 && local_c < 64 && local_r >= 0 && local_r < 64) {
        mem_penalty += cycle.ai_memory_grid[local_r][local_c];
        samples++;
      }
      dist += 5.0f; // Optimized Precision (5.0u)
    }
  }
    s_ray_distances[tid] = dist;
    s_ray_penalties[tid] = (samples > 0) ? (mem_penalty / (float)samples) : 0.0f;
    
    // Safety: cycle is a reference to a single struct in global memory.
    // Threads 0-12 write their own respective indices into last_ray_distances.
    // This is race-free because indices are unique per tid.
    cycle.last_ray_distances[tid] = dist;
  }

  // Ensure all threads have finished raycasting before Thread 0 
  // starts modifying common state in the sequential block.
  __syncthreads();

  // Sequential Decision & Physics (Thread 0)
  if (tid == 0) {
    // Proximity Pressure Check: count how many rays are pressed against walls
    int proximity_hits = 0;
    for (int i = 0; i < NUM_RAYS; i++) {
        if (s_ray_distances[i] <= 5.01f) proximity_hits++;
    }

    int local_c = (curr_c / (MAZE_COLS / 64)) % 64;
    int local_r = (curr_r / (MAZE_ROWS / 64)) % 64;
    if (local_c >= 0 && local_c < 64 && local_r >= 0 && local_r < 64) {
      cycle.ai_memory_grid[local_r][local_c] += 0.08f; 
    }

    int straight_ray = NUM_RAYS / 2; // Ray 16 in a 33-ray sweep
    float dist_straight = s_ray_distances[straight_ray];
    cycle.front_dist = dist_straight;

    // Ray Resolution (Symmetric 33-ray sweep always)
    int n_rays = NUM_RAYS;
    cycle.active_rays = n_rays;

    // Junction Detection (Updated for 33 rays)
    int open_count = 0;
    if (dist_straight > 800.0f) open_count++;
    if (s_ray_distances[straight_ray - 8] > 800.0f) open_count++; 
    if (s_ray_distances[straight_ray + 8] > 800.0f) open_count++; 
    if (s_ray_distances[straight_ray - 16] > 800.0f) open_count++; 
    if (s_ray_distances[straight_ray + 16] > 800.0f) open_count++; 

    if (open_count >= 3 && !cycle.is_deciding && !cycle.is_scanning && !cycle.is_reversing) {
      float dx_j = cycle.x - cycle.last_junction_x;
      float dz_j = cycle.z - cycle.last_junction_z;
      if (sqrtf(dx_j*dx_j + dz_j*dz_j) > 400.0f) {
        cycle.is_deciding = true;
        cycle.decision_timer = 40;
        cycle.last_junction_x = cycle.x;
        cycle.last_junction_z = cycle.z;
      }
    }

    int best_ray = 0;
    float best_score = -99999.0f;
    
    // Dynamic penalty scaling (Blast-Through logic)
    float base_penalty = 1200.0f; // Scale increased because s_ray_penalties is now average (0-1 range)
    if (cycle.stuck_timer > 30) {
      // Scale down penalty significantly as stuck_timer progresses
      float panic_factor = fminf(1.0f, (float)(cycle.stuck_timer - 30) / 60.0f);
      base_penalty *= (1.0f - panic_factor * 0.95f);
    }

    for (int i = 0; i < n_rays; ++i) {
      // Normalized penalty: distance multiplied by (1.0 - k*avg_heat)
      float heat_factor = fmaxf(0.01f, 1.0f - (s_ray_penalties[i] * base_penalty / 1000.0f));
      // Symmetric Angular Penalty (Favor the center ray 16)
      float angle_penalty = fabsf(d_radar_angles[i]) * 15.0f; // Increased from 10
      float score = s_ray_distances[i] * heat_factor - angle_penalty;
      
      if (cycle.is_deciding && s_ray_penalties[i] < 0.05f) score += 500.0f;
      if (i == cycle.best_ray_idx) {
        score += 5.0f; // Lowered Stickiness (was 30)
        // Physical Stuck Avoidance: Penalize the current ray if we've been blocked for >30 frames 
        if (cycle.stuck_timer > 30) score *= 0.5f; 
      }
      if (i == straight_ray && s_ray_distances[i] > 300.0f) score += 50.0f; // Center Bias (was +15 at index 0)
      if (score > best_score) {
        best_score = score;
        best_ray = i;
      }
    }

    cycle.best_ray_idx = best_ray;
    cycle.best_dist = s_ray_distances[best_ray];

    // State Machine & Physics
    if (cycle.is_scanning) {
      cycle.thinking_id = 5;
      cycle.dir += cycle.scan_dir;
      cycle.current_speed = 1.0f;
      cycle.target_speed = 1.0f;
      cycle.scan_count++;
      if (dist_straight > 400.0f) {
        cycle.is_scanning = false;
        cycle.scan_count = 0;
      } else if (cycle.scan_count > 150) {
        // Falling into reversing/backtracking is safer than just stopping
        cycle.is_scanning = false;
        cycle.scan_count = 0;
        if (cycle.crumb_count > 0) {
          cycle.is_reversing = true;
          cycle.stuck_timer = 0;
        } else {
          // Last Resort: No crumbs at spawn? Randomly jitter direction by 90-deg increments
          cycle.dir += (float)(1 + curand(&cycle.rng) % 3) * (M_PI / 2.0f);
          cycle.stuck_timer = 0;
        }
      }
    } else if (cycle.is_deciding) {
      cycle.thinking_id = 7;
      cycle.current_speed = 0.0f;
      cycle.target_speed = 0.0f;
      cycle.decision_timer--;
      
      // Turn towards the best detected ray even while paused
      float target_diff = d_radar_angles[best_ray];
      cycle.dir += target_diff * 0.12f; 
      
      if (cycle.decision_timer <= 0) cycle.is_deciding = false;
    } else if (cycle.is_reversing) {
      cycle.thinking_id = 6;
      if (cycle.crumb_count <= 0) {
        cycle.is_reversing = false;
      } else {
        int best_bc_idx = (cycle.crumb_idx - 1 + 16) % 16;
        for (int i = cycle.crumb_count - 1; i >= 0; --i) {
          int check_idx = (cycle.crumb_idx - 1 - i + 16) % 16;
          float bx = cycle.crumb_x[check_idx];
          float bz = cycle.crumb_z[check_idx];
          bool los = true;
          for (float t = 0.2f; t < 1.0f; t += 0.4f) {
            float lx = cycle.x + (bx - cycle.x) * t;
            float lz = cycle.z + (bz - cycle.z) * t;
            if (is_wall(lx, lz, offsetX, offsetZ)) { los = false; break; }
          }
          if (los) {
            best_bc_idx = check_idx;
            cycle.crumb_count = i + 1; // Preserve target crumb until reached
            break;
          }
        }
        float tx = cycle.crumb_x[best_bc_idx];
        float tz = cycle.crumb_z[best_bc_idx];
        float dx = tx - cycle.x;
        float dz = tz - cycle.z;
        float dist_to_crumb = sqrtf(dx*dx + dz*dz);
        float target_f = atan2f(dx, dz);
        float diff = target_f - cycle.dir;
        while (diff > M_PI) diff -= 2.0f * (float)M_PI;
        while (diff < -M_PI) diff += 2.0f * (float)M_PI;
        cycle.dir += diff * 0.15f;
        // Bug 1: Sliding speed approach prevents oscillation at walls
        cycle.current_speed += (-3.0f - cycle.current_speed) * 0.2f; 
        if (dist_to_crumb < 25.0f) {
          cycle.crumb_idx = best_bc_idx;
          // Consumption: Decrement count to move to the previous crumb
          if (cycle.crumb_count > 0) cycle.crumb_count--;
        }
        
        // Backtracking Escape: If blocked while reversing for >40 frames, abort
        if (cycle.stuck_timer > 40) {
            cycle.is_reversing = false;
            cycle.stuck_timer = 0;
        }

        if (dist_straight > 450.0f) cycle.is_reversing = false;
      }
    } else {
      // Normal Navigation
      cycle.thinking_id = (best_ray == straight_ray) ? 0 : ((d_radar_angles[best_ray] < 0) ? 2 : 3);
      if (dist_straight < 150.0f) cycle.thinking_id = 1;
      
      float front_danger = fmaxf(0.0f, 1.0f - (dist_straight / 800.0f));
      // Proportional Steering Controller: Turn at a consistent rate until aligned with best_ray
      float error = d_radar_angles[best_ray];
      float max_turn = 0.08f + 0.25f * front_danger; // Danger-boosted cap
      float turn_amount = fminf(fabsf(error), max_turn);
      cycle.dir += copysignf(turn_amount, error);
      float base_speed = 6.0f;
      cycle.target_speed = base_speed * (1.1f - front_danger);
      if (dist_straight < 80.0f) cycle.target_speed = 0.5f; // Emergency slowdown for steering alignment
      if (cycle.target_speed < 1.0f && dist_straight > 80.0f) cycle.target_speed = 1.0f;

      cycle.current_speed += (cycle.target_speed - cycle.current_speed) * 0.1f;

      // Velocity-based stuck detection: if we want to move but aren't moving fast
      bool physically_blocked = (fabsf(cycle.current_speed) < 0.4f && fabsf(cycle.target_speed) > 1.0f);
      if (physically_blocked || dist_straight < 100.0f) {
        cycle.stuck_timer++;
      } else {
        // Persistent Stuck Avoidance: Only reset if we are actually moving forward cleanly
        if (cycle.current_speed > 1.5f && dist_straight > 200.0f) {
          cycle.stuck_timer = 0;
        }
      }

      if (cycle.stuck_timer > 30 && cycle.thinking_id < 4) {
        cycle.thinking_id = 8; // Special 'Blasting Through' status
      }

      if (cycle.stuck_timer > 60) {
        if (dist_straight < 50.0f && cycle.crumb_count > 0) {
          cycle.is_reversing = true;
          cycle.stuck_timer = 0;
        } else {
          cycle.thinking_id = 4; // Panic/Stuck recovery
          cycle.is_scanning = true;
          cycle.scan_dir = (curand(&cycle.rng) % 2 == 0) ? 0.08f : -0.08f;
          cycle.stuck_timer = 0;
        }
      }
      
      if (cycle.stuck_timer > 30 && cycle.thinking_id < 4) {
        cycle.thinking_id = 8; // Blasting
      }

      // Corner Escape: If surrounded by walls at proximity, flip or scan
      if (proximity_hits > 16) {
          cycle.dir += (float)M_PI; // Hard flip
          cycle.stuck_timer = 0;
          cycle.thinking_id = 4;
      } else if (proximity_hits > 6 && !cycle.is_scanning && !cycle.is_reversing) {
          cycle.thinking_id = 4; // Panic
          cycle.is_scanning = true;
          cycle.scan_dir = (curand(&cycle.rng) % 2 == 0) ? 0.3f : -0.3f; // Faster pivot
          cycle.stuck_timer = 0;
      }

      // Shuttle Detection: Track positional variance to break corridor locks
      if (cycle.step_count == 1) {
          cycle.x_min_recent = cycle.x_max_recent = cycle.x;
          cycle.z_min_recent = cycle.z_max_recent = cycle.z;
      } else {
          cycle.x_min_recent = fminf(cycle.x_min_recent, cycle.x);
          cycle.x_max_recent = fmaxf(cycle.x_max_recent, cycle.x);
          cycle.z_min_recent = fminf(cycle.z_min_recent, cycle.z);
          cycle.z_max_recent = fmaxf(cycle.z_max_recent, cycle.z);
      }

      if (cycle.step_count % 500 == 0 && cycle.step_count > 0) {
          float x_span = cycle.x_max_recent - cycle.x_min_recent;
          float z_span = cycle.z_max_recent - cycle.z_min_recent;
          // If trapped in a 1-cell corridor (300u) for 500 steps, force a break
          if (x_span < 300.0f && z_span < 300.0f) {
              if (!cycle.is_scanning) {
                  cycle.is_scanning = true;
                  cycle.scan_dir = (curand(&cycle.rng) % 2 == 0) ? (M_PI / 2.0f) : -(M_PI / 2.0f);
                  cycle.scan_count = 130; // Pre-load to almost finish scan immediately
                  cycle.thinking_id = 9; // Status: Force Exploration
              }
          }
          // Reset bounds for next window
          cycle.x_min_recent = cycle.x_max_recent = cycle.x;
          cycle.z_min_recent = cycle.z_max_recent = cycle.z;
      }
    }

    // Move with Collision Check
    float dx = sinf(cycle.dir) * cycle.current_speed;
    float dz = cosf(cycle.dir) * cycle.current_speed;
    float next_x = cycle.x + dx;
    float next_z = cycle.z + dz;

    // Hard collision check: Don't move if we hit a wall
    if (is_wall(next_x, next_z, offsetX, offsetZ)) {
      cycle.current_speed = 0.0f;
      // Robust stuck accumulation: increment timer whenever we are physically stopped
      if (!cycle.is_scanning) {
        cycle.stuck_timer++; 
      }
    } else {
      cycle.x = next_x;
      cycle.z = next_z;
    }

    cycle.dist_since_crumb += fabsf(cycle.current_speed);
    if (cycle.dist_since_crumb > 150.0f && !cycle.is_reversing) {
      cycle.crumb_x[cycle.crumb_idx] = cycle.x;
      cycle.crumb_z[cycle.crumb_idx] = cycle.z;
      cycle.crumb_idx = (cycle.crumb_idx + 1) % 16;
      if (cycle.crumb_count < 16) cycle.crumb_count++;
      cycle.dist_since_crumb = 0.0f;
    }

    float bounds_x = (MAZE_COLS * CELL_SIZE) / 2.0f - CELL_SIZE;
    float bounds_z = (MAZE_ROWS * CELL_SIZE) / 2.0f - CELL_SIZE;
    if (cycle.x > bounds_x) cycle.x = bounds_x;
    if (cycle.x < -bounds_x) cycle.x = -bounds_x;
    if (cycle.z > bounds_z) cycle.z = bounds_z;
    if (cycle.z < -bounds_z) cycle.z = -bounds_z;

    // Memory Decay (Logical expiration - Corrected bounds to 64x64)
    for (int r = 0; r < 64; ++r) {
      for (int c = 0; c < 64; ++c) {
        cycle.ai_memory_grid[r][c] *= 0.999f; 
      }
    }

    cycle.step_count++;
  }
}

struct Wall {
  float x1, z1, x2, z2;
};
struct TrailNode {
  float x, z;
};

// Global for font rendering
GLuint fontBase;

void renderText(const char *text, float x, float y) {
  glRasterPos2f(x, y);
  glPushAttrib(GL_LIST_BIT);
  glListBase(fontBase - 32); // Offset for ASCII 32 start
  glCallLists((GLsizei)strlen(text), GL_UNSIGNED_BYTE, (const GLvoid *)text);
  glPopAttrib();
}

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam,
                            LPARAM lParam) {
  if (uMsg == WM_CLOSE || uMsg == WM_DESTROY) {
    PostQuitMessage(0);
    return 0;
  }
  return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

int main() {
  cudaSetDevice(0);
  WNDCLASS wc = {};
  wc.lpfnWndProc = WindowProc;
  wc.hInstance = GetModuleHandle(NULL);
  wc.lpszClassName = "TronCUDA";
  RegisterClass(&wc);

  HWND hwnd =
      CreateWindowEx(0, "TronCUDA", "Tron CUDA Visualization OpenGL",
                     WS_OVERLAPPEDWINDOW | WS_VISIBLE, CW_USEDEFAULT,
                     CW_USEDEFAULT, 800, 600, NULL, NULL, wc.hInstance, NULL);

  HDC hdc = GetDC(hwnd);
  PIXELFORMATDESCRIPTOR pfd = {sizeof(PIXELFORMATDESCRIPTOR),
                               1,
                               PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL |
                                   PFD_DOUBLEBUFFER,
                               PFD_TYPE_RGBA,
                               32,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               24,
                               8,
                               0,
                               PFD_MAIN_PLANE,
                               0,
                               0,
                               0,
                               0};
  int nPixelFormat = ChoosePixelFormat(hdc, &pfd);
  SetPixelFormat(hdc, nPixelFormat, &pfd);
  HGLRC hrc = wglCreateContext(hdc);
  wglMakeCurrent(hdc, hrc);
  printf("Renderer: %s\n", glGetString(GL_RENDERER));
  printf("Vendor: %s\n", glGetString(GL_VENDOR));

  init_gl_extensions();

  // Initialize Font
  fontBase = glGenLists(96);
  HFONT hFont = CreateFont(-18, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE, ANSI_CHARSET,
                           OUT_TT_ONLY_PRECIS, CLIP_DEFAULT_PRECIS, ANTIALIASED_QUALITY,
                           DEFAULT_PITCH | FF_DONTCARE, "Courier New");
  SelectObject(hdc, hFont);
  wglUseFontBitmaps(hdc, 32, 96, fontBase);
  DeleteObject(hFont);

  srand((unsigned int)time(NULL));
  int *host_maze = (int*)malloc(MAZE_ROWS * MAZE_COLS * sizeof(int));
  for (int r = 0; r < MAZE_ROWS; r++)
    for (int c = 0; c < MAZE_COLS; c++)
      host_maze[r * MAZE_COLS + c] = 1;
  carve_maze(1, 1, host_maze);
  // Knock down extra walls to create many more open shortcuts / loops
  for (int i = 0; i < 500; ++i) {
    int rr = 1 + rand() % (MAZE_ROWS - 2);
    int rc = 1 + rand() % (MAZE_COLS - 2);
    host_maze[rr * MAZE_COLS + rc] = 0;
  }

  int *d_maze_mem;
  CUDA_CHECK(cudaMalloc(&d_maze_mem, MAZE_ROWS * MAZE_COLS * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_maze_mem, host_maze, MAZE_ROWS * MAZE_COLS * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpyToSymbol(d_maze_grid_ptr, &d_maze_mem, sizeof(int*)));
  init_radar_angles();
  CUDA_CHECK(cudaMemcpyToSymbol(d_radar_angles, RADAR_ANGLES, sizeof(float) * NUM_RAYS));

  std::vector<Wall> maze_walls;
  float offset_x = (MAZE_COLS * CELL_SIZE) / 2.0f;
  float offset_z = (MAZE_ROWS * CELL_SIZE) / 2.0f;
  for (int r = 0; r < MAZE_ROWS; r++) {
    for (int c = 0; c < MAZE_COLS; c++) {
      if (host_maze[r * MAZE_COLS + c] == 1) {
        float x1 = c * CELL_SIZE - offset_x;
        float z1 = r * CELL_SIZE - offset_z;
        float x2 = x1 + CELL_SIZE;
        float z2 = z1 + CELL_SIZE;
        if (r == 0 || host_maze[(r - 1) * MAZE_COLS + c] == 0)
          maze_walls.push_back({x1, z1, x2, z1});
        if (r == MAZE_ROWS - 1 || host_maze[(r + 1) * MAZE_COLS + c] == 0)
          maze_walls.push_back({x1, z2, x2, z2});
        if (c == 0 || host_maze[r * MAZE_COLS + (c - 1)] == 0)
          maze_walls.push_back({x1, z1, x1, z2});
        if (c == MAZE_COLS - 1 || host_maze[r * MAZE_COLS + (c + 1)] == 0)
          maze_walls.push_back({x2, z1, x2, z2});
      }
    }
  }

  // OpenGL Display List for Maze Walls (Quads - Pass 1)
  float wall_h = 100.0f;
  GLuint maze_quads_list = glGenLists(1);
  glNewList(maze_quads_list, GL_COMPILE);
  glBegin(GL_QUADS);
  for (const auto &w : maze_walls) {
      float r_var = 0.15f + 0.15f * sinf(w.x1 * 0.005f);
      float b_var = 0.4f + 0.2f * cosf(w.z1 * 0.005f);
      glColor4f(r_var, 0.0f, b_var, 0.35f); 
      glVertex3f(w.x1, 0, w.z1);
      glVertex3f(w.x2, 0, w.z2);
      glVertex3f(w.x2, wall_h, w.z2);
      glVertex3f(w.x1, wall_h, w.z1);
  }
  glEnd();
  glEndList();

  // OpenGL Display List for Maze Walls (Neon Lines - Pass 2)
  GLuint maze_lines_list = glGenLists(1);
  glNewList(maze_lines_list, GL_COMPILE);
  glBegin(GL_LINES);
  for (const auto &w : maze_walls) {
      float r_var = 0.5f + 0.4f * sinf(w.x1 * 0.01f);
      float b_var = 0.6f + 0.4f * cosf(w.z1 * 0.01f);
      glColor4f(r_var, 0.0f, b_var, 0.9f); 
      glVertex3f(w.x1, 0, w.z1); glVertex3f(w.x2, 0, w.z2);
      glVertex3f(w.x1, wall_h, w.z1); glVertex3f(w.x2, wall_h, w.z2);
      glVertex3f(w.x1, 0, w.z1); glVertex3f(w.x1, wall_h, w.z1);
      glVertex3f(w.x2, 0, w.z2); glVertex3f(w.x2, wall_h, w.z2);
  }
  glEnd();
  glEndList();

  CycleState *d_cycles;
  CUDA_CHECK(cudaMalloc(&d_cycles, NUM_AGENTS * sizeof(CycleState)));
  CycleState *h_agents = (CycleState*)calloc(NUM_AGENTS, sizeof(CycleState));
  
  float spawn_range_x = (MAZE_COLS * CELL_SIZE) - (CELL_SIZE * 4.0f);
  float spawn_range_z = (MAZE_ROWS * CELL_SIZE) - (CELL_SIZE * 4.0f);

  for (int i = 0; i < NUM_AGENTS; ++i) {
    bool found_spawn = false;
    for (int retry = 0; retry < 100; ++retry) {
        float rx = ((float)rand() / RAND_MAX) * spawn_range_x - (spawn_range_x / 2.0f);
        float rz = ((float)rand() / RAND_MAX) * spawn_range_z - (spawn_range_z / 2.0f);
        
        int grid_c = (int)floorf((rx + offset_x) / CELL_SIZE);
        int grid_r = (int)floorf((rz + offset_z) / CELL_SIZE);
        
        if (grid_c >= 0 && grid_c < MAZE_COLS && grid_r >= 0 && grid_r < MAZE_ROWS) {
            if (host_maze[grid_r * MAZE_COLS + grid_c] == 0) {
                h_agents[i].x = rx;
                h_agents[i].z = rz;
                found_spawn = true;
                break;
            }
        }
    }
    if (!found_spawn) {
        h_agents[i].x = CELL_SIZE - offset_x + (CELL_SIZE / 2.0f);
        h_agents[i].z = CELL_SIZE - offset_z + (CELL_SIZE / 2.0f);
    }
    
    h_agents[i].dir = (float)(rand() % 360) * (M_PI / 180.0f);
    h_agents[i].current_speed = 3.0f;
    h_agents[i].target_speed = 3.0f;
    h_agents[i].active_rays = NUM_RAYS;
    h_agents[i].last_junction_x = -9999.0f;
    h_agents[i].last_junction_z = -9999.0f;
  }
  
  CUDA_CHECK(cudaMemcpy(d_cycles, h_agents, NUM_AGENTS * sizeof(CycleState), cudaMemcpyHostToDevice));
  init_cycles<<<NUM_AGENTS/128, 128>>>(d_cycles, (unsigned long long)time(NULL));
  
  CompactAgent *d_compact;
  CUDA_CHECK(cudaMalloc(&d_compact, NUM_AGENTS * sizeof(CompactAgent)));
  CompactAgent *h_compact = (CompactAgent*)malloc(NUM_AGENTS * sizeof(CompactAgent));

  // Initialize VBO for Swarm
  GLuint swarm_vbo;
  glGenBuffers(1, &swarm_vbo);
  glBindBuffer(GL_ARRAY_BUFFER, swarm_vbo);
  glBufferData(GL_ARRAY_BUFFER, NUM_AGENTS * sizeof(float3), NULL, GL_DYNAMIC_DRAW);
  
  // Register VBO with CUDA (Only if using NVIDIA Hardware)
  cudaGraphicsResource *vbo_res = NULL;
  bool use_interop = false;
  const char* renderer = (const char*)glGetString(GL_RENDERER);
  if (renderer && strstr(renderer, "NVIDIA")) {
      cudaError_t interop_err = cudaGraphicsGLRegisterBuffer(&vbo_res, swarm_vbo, cudaGraphicsMapFlagsWriteDiscard);
      if (interop_err == cudaSuccess) {
          use_interop = true;
          printf("CUDA-GL Interop enabled on %s\n", renderer);
      } else {
          printf("WARNING: CUDA-GL Interop failed (%s) on %s.\n", cudaGetErrorString(interop_err), renderer);
      }
  } else {
      printf("Renderer is %s. CUDA-GL Interop disabled (fallback active).\n", renderer ? renderer : "Unknown");
  }
  
  // If we couldn't get a hardware renderer yet, we should probably warn
  if (!renderer) printf("WARNING: OpenGL renderer string is NULL!\n");

  CycleState &h_cycle = h_agents[0]; // Primary agent for visualization
  std::deque<TrailNode> trails;
  float last_trail_x = h_cycle.x;
  float last_trail_z = h_cycle.z;

  glEnable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE); 
  glEnable(GL_LINE_SMOOTH);         
  glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

  std::ofstream telemetry_file("telemetry.csv");
  telemetry_file << "Version: " << APP_VERSION << "\n";
  telemetry_file << "Step,Speed,TotalHeat,X,Z,FPS,FPS_Status,ThinkingID,ActiveRays,StuckTimer,RayL86,RayL43,RayCenter,RayR43,RayR86\n";

  std::string telemetry_buffer;
  int telemetry_frame_count = 0;

  // High-resolution timer for FPS measurement
  LARGE_INTEGER qpc_freq, frame_start;
  QueryPerformanceFrequency(&qpc_freq);
  QueryPerformanceCounter(&frame_start);

  MSG msg;
  bool running = true;
  while (running) {
    while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
      if (msg.message == WM_QUIT)
        running = false;
      TranslateMessage(&msg);
      DispatchMessage(&msg);
    }
    if (!running)
      break;

    decay_memory<<<NUM_AGENTS, 64>>>(d_cycles);
    step_cycle<<<NUM_AGENTS, NUM_RAYS>>>(d_cycles, offset_x, offset_z);
    
    // Swarm Update (with Interop Fallback)
    if (use_interop) {
        float3 *d_vbo_ptr;
        size_t vbo_size;
        CUDA_CHECK(cudaGraphicsMapResources(1, &vbo_res, 0));
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&d_vbo_ptr, &vbo_size, vbo_res));
        pack_compact_agents<<<NUM_AGENTS/128, 128>>>(d_cycles, d_compact, d_vbo_ptr);
        CUDA_CHECK(cudaGraphicsUnmapResources(1, &vbo_res, 0));
    } else {
        // Fallback: Pack to d_compact, then copy to host, then upload to GL
        pack_compact_agents<<<NUM_AGENTS/128, 128>>>(d_cycles, d_compact, NULL);
    }
    
    cudaDeviceSynchronize();
    
    // Bandwidth Optimized Sync: 
    // 1. Copy full state for Agent 0 only (~17KB)
    cudaMemcpy(&h_agents[0], d_cycles, sizeof(CycleState), cudaMemcpyDeviceToHost);
    // 2. Copy compact headers for the swarm (~64KB for 4096)
    cudaMemcpy(h_compact, d_compact, NUM_AGENTS * sizeof(CompactAgent), cudaMemcpyDeviceToHost);
    
    // 3. Fallback GL upload (if no interop)
    if (!use_interop) {
        glBindBuffer(GL_ARRAY_BUFFER, swarm_vbo);
        glBufferSubData(GL_ARRAY_BUFFER, 0, NUM_AGENTS * sizeof(float3), h_compact);
    }
    
    // h_cycle is a reference to h_agents[0], so it's already updated.

    // FPS measurement using high-resolution timer
    LARGE_INTEGER frame_end;
    QueryPerformanceCounter(&frame_end);
    double frame_ms = (double)(frame_end.QuadPart - frame_start.QuadPart) /
                      qpc_freq.QuadPart * 1000.0;
    double fps = (frame_ms > 0.0) ? (1000.0 / frame_ms) : 9999.0;
    frame_start = frame_end;

    float dx = h_cycle.x - last_trail_x;
    float dz = h_cycle.z - last_trail_z;
    if (sqrtf(dx * dx + dz * dz) > 20.0f) {
      trails.push_back({last_trail_x, last_trail_z});
      last_trail_x = h_cycle.x;
      last_trail_z = h_cycle.z;
      if (trails.size() > 100)
        trails.pop_front();
    }

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    RECT rect;
    GetClientRect(hwnd, &rect);
    float w = rect.right - rect.left;
    float h = rect.bottom - rect.top;
    if (h == 0)
      h = 1;

    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, w / h, 1.0, 10000.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // Dynamic Camera Logic
    static float cur_cam_dist = 150.0f;
    static float cur_cam_height = 80.0f;
    static float target_cam_dist = 150.0f;
    static float target_cam_height = 80.0f;
    static int cam_mode_timer = 0;
    
    cam_mode_timer--;
    if (cam_mode_timer <= 0) {
      int mode = rand() % 4; // 0: Follow, 1: Overhead, 2: Close, 3: God View
      if (mode == 0) { 
        target_cam_dist = 150.0f;
        target_cam_height = 80.0f;
      } else if (mode == 1) { 
        target_cam_dist = 50.0f;
        target_cam_height = 400.0f;
      } else if (mode == 2) { 
        target_cam_dist = 60.0f;
        target_cam_height = 40.0f;
      } else { // God View
        target_cam_dist = 600.0f; // Orbital radius
        target_cam_height = 8000.0f; // Stratospheric height
      }
      cam_mode_timer = 400 + rand() % 400; 
    }

    // Smooth Interpolation (Lerp)
    cur_cam_dist += (target_cam_dist - cur_cam_dist) * 0.02f;
    cur_cam_height += (target_cam_height - cur_cam_height) * 0.02f;

    float cx, cy, cz;
    if (target_cam_height > 1000.0f) { // God View Special Orbit
        float god_angle = (float)GetTickCount() * 0.0001f;
        cx = sinf(god_angle) * target_cam_dist;
        cz = cosf(god_angle) * target_cam_dist;
        cy = cur_cam_height;
        gluLookAt(cx, cy, cz, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
    } else {
        cx = h_cycle.x - sinf(h_cycle.dir) * cur_cam_dist;
        cz = h_cycle.z - cosf(h_cycle.dir) * cur_cam_dist;
        cy = cur_cam_height;
        gluLookAt(cx, cy, cz, h_cycle.x, 20.0f, h_cycle.z, 0.0f, 1.0f, 0.0f);
    }

    // Visualize all agents via VBO (Zero-Copy)
    glPointSize(4.0f);
    glEnableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, swarm_vbo);
    glVertexPointer(3, GL_FLOAT, 0, 0);
    
    // Draw all others faintly
    glColor4f(1.0f, 1.0f, 1.0f, 0.3f);
    glDrawArrays(GL_POINTS, 0, NUM_AGENTS);
    
    // Highlight Agent 0
    glColor4f(0.0f, 1.0f, 1.0f, 1.0f);
    glDrawArrays(GL_POINTS, 0, 1);
    
    glDisableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Grid
    glLineWidth(1.0f);
    glBegin(GL_LINES);
    glColor4f(0.0f, 0.3f, 0.6f, 0.5f);
    int step = 50;
    int grid_size = 1000;
    int start_x = ((int)cx / step) * step - grid_size;
    int end_x = start_x + grid_size * 2;
    int start_z = ((int)cz / step) * step - grid_size;
    int end_z = start_z + grid_size * 2;
    for (int x = start_x; x <= end_x; x += step) {
      for (int z = start_z; z <= end_z; z += step) {
        glVertex3f(x, 0, z);
        glVertex3f(x, 0, z + step);
        glVertex3f(x, 0, z);
        glVertex3f(x + step, 0, z);
      }
    }
    glEnd();

    // Maze Wall Shaded Quads (Display List)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glCallList(maze_quads_list);

    // Maze Wall Neon Edges (Display List)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glLineWidth(2.0f);
    glCallList(maze_lines_list);

    // Cycle Trail Ribbons
    glLineWidth(4.0f);
    glBegin(GL_LINES);
    std::deque<TrailNode> all_trails = trails;
    all_trails.push_back({last_trail_x, last_trail_z});
    all_trails.push_back({h_cycle.x, h_cycle.z});
    
    for (size_t i = 1; i < all_trails.size(); ++i) {
      auto &t1 = all_trails[i - 1];
      auto &t2 = all_trails[i];
      
      // Calculate alpha fading based on index
      float alpha = (float)i / (float)all_trails.size();
      float th = 25.0f * alpha; // Trail height also tapers for aesthetics
      
      glColor4f(1.0f, 0.5f, 0.0f, alpha);
      
      // Vertical ribbon
      glVertex3f(t1.x, 0, t1.z);
      glVertex3f(t2.x, 0, t2.z);
      glVertex3f(t1.x, th, t1.z);
      glVertex3f(t2.x, th, t2.z);
      
      // Vertical connectors
      glVertex3f(t1.x, 0, t1.z);
      glVertex3f(t1.x, th, t1.z);
      glVertex3f(t2.x, 0, t2.z);
      glVertex3f(t2.x, th, t2.z);
    }
    glEnd();

    // Draw Breadcrumbs (last ~15m of path)
    {
      glPointSize(6.0f);
      glBegin(GL_POINTS);
      float bc_pulse = 0.5f + 0.5f * sinf(h_cycle.step_count * 0.15f);
      for (int i = 0; i < h_cycle.crumb_count; ++i) {
        // Most recent crumbs are brighter
        float alpha = (float)(h_cycle.crumb_count - i) / (float)h_cycle.crumb_count;
        glColor4f(1.0f, 0.5f, 0.0f, (0.3f + 0.5f * bc_pulse) * alpha);
        int idx = (h_cycle.crumb_idx - 1 - i + 16) % 16;
        glVertex3f(h_cycle.crumb_x[idx], 2.0f, h_cycle.crumb_z[idx]);
      }
      glEnd();
    }

    // Draw Detailed Cycle Model (Ported from Python)
    glPushMatrix();
    glTranslatef(h_cycle.x, 0.0f, h_cycle.z);
    glRotatef(h_cycle.dir * (180.0f / (float)M_PI), 0, 1, 0);
    
    glColor4f(0.0f, 1.0f, 1.0f, 1.0f);
    glLineWidth(2.0f);

    auto add_wheel = [](float center_z, float radius, float width) {
      glBegin(GL_LINES);
      for (int i = 0; i < 8; ++i) {
        float angle = 2.0f * (float)M_PI * (float)i / 8.0f;
        float next_angle = 2.0f * (float)M_PI * (float)((i + 1) % 8) / 8.0f;
        
        float y = radius + radius * cosf(angle);
        float z = center_z + radius * sinf(angle);
        float next_y = radius + radius * cosf(next_angle);
        float next_z = center_z + radius * sinf(next_angle);

        // Left rim
        glVertex3f(-width/2, y, z);
        glVertex3f(-width/2, next_y, next_z);
        // Right rim
        glVertex3f(width/2, y, z);
        glVertex3f(width/2, next_y, next_z);
        // Cross beam
        glVertex3f(-width/2, y, z);
        glVertex3f(width/2, y, z);
      }
      glEnd();
    };

    add_wheel(12.0f, 6.0f, 4.0f);   // Front
    add_wheel(-12.0f, 8.0f, 8.0f);  // Rear

    // Body Shell
    float body_verts[10][3] = {
        {-3, 6, 15}, {3, 6, 15},   // Front bumper
        {-4, 14, 2}, {4, 14, 2},   // High canopy front
        {-4, 16, -8}, {4, 16, -8}, // Top canopy peak
        {-3, 8, -20}, {3, 8, -20}, // Tail end
        {-5, 4, -4}, {5, 4, -4}    // Side flares
    };
    int body_edges[19][2] = {
        {0,1}, {2,3}, {4,5}, {6,7}, {8,9}, // Lateral
        {0,2}, {2,4}, {4,6},               // Left top
        {1,3}, {3,5}, {5,7},               // Right top
        {0,8}, {8,6},                      // Left bottom
        {1,9}, {9,7},                      // Right bottom
        {2,8}, {4,8},                      // Left side tri
        {3,9}, {5,9}                       // Right side tri
    };

    glBegin(GL_LINES);
    for (int i = 0; i < 19; ++i) {
      glVertex3f(body_verts[body_edges[i][0]][0], body_verts[body_edges[i][0]][1], body_verts[body_edges[i][0]][2]);
      glVertex3f(body_verts[body_edges[i][1]][0], body_verts[body_edges[i][1]][1], body_verts[body_edges[i][1]][2]);
    }
    glEnd();
    
    glPopMatrix();

    // Render 3D Radar "Signals" (Pulsing Feelers)
    {
      glLineWidth(1.5f);
      glBegin(GL_LINES);
      float pulse = 0.5f + 0.5f * sinf(h_cycle.step_count * 0.2f);
      for (int i = 0; i < h_cycle.active_rays; ++i) {
        float angle = h_cycle.dir + RADAR_ANGLES[i];
        float dist = h_cycle.last_ray_distances[i];
        
        if (i == h_cycle.best_ray_idx) {
          glLineWidth(4.0f); // Thicker
          glColor4f(1.0f, 1.0f, 0.2f, 1.0f); // Bright yellow/white
        } else {
          glLineWidth(1.5f);
          glColor4f(0.0f, 0.8f, 1.0f, 0.2f + 0.3f * pulse);
        }
        
        glVertex3f(h_cycle.x, 15.0f, h_cycle.z); // Slightly higher off ground
        glVertex3f(h_cycle.x + sinf(angle) * dist, 15.0f, h_cycle.z + cosf(angle) * dist);
      }
      glLineWidth(1.5f);
      glEnd();
    }


    // ======================================================
    // 2D Minimap Overlay (matches Python minimap floor plan)
    // ======================================================
    // Switch to 2D orthographic projection for HUD/minimap drawing
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0.0, w, 0.0, h); // origin = bottom-left
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glDisable(GL_DEPTH_TEST);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Radar Telemetry Log (Static Overlay)
    {
      const int log_x = 20;
      int log_y = (int)h - 200; 

      for (int i = 0; i < h_cycle.active_rays; ++i) {
        char ray_buf[128];
        float dist_m = h_cycle.last_ray_distances[i] / 10.0f;
        float deg = RADAR_ANGLES[i] * (180.0f / (float)M_PI);
        sprintf(ray_buf, "ray %d hit a wall at %.1f meters, angle %.0f '", i, dist_m, deg);
        if (i == h_cycle.best_ray_idx) glColor4f(1.0f, 1.0f, 0.0f, 1.0f);
        else glColor4f(0.0f, 0.8f, 1.0f, 0.6f);
        renderText(ray_buf, (float)log_x, (float)log_y);
        log_y -= 15; 
      }
    }

    const int minimap_cell = 1; 
    const int map_w = MAZE_COLS * minimap_cell;
    const int map_h = MAZE_ROWS * minimap_cell;
    const int margin = 20;
    const int map_x = (int)w - map_w - margin;
    const int map_y = margin; 

    // Background panel
    glBegin(GL_QUADS);
    glColor4f(0.0f, 0.0f, 0.12f, 0.85f);
    glVertex2f(map_x - 5.0f,         map_y - 5.0f);
    glVertex2f(map_x + map_w + 5.0f, map_y - 5.0f);
    glVertex2f(map_x + map_w + 5.0f, map_y + map_h + 5.0f);
    glVertex2f(map_x - 5.0f,         map_y + map_h + 5.0f);
    glEnd();

    // Minimap Throttling: Render full detail only every 4th frame
    static int minimap_render_tick = 0;
    minimap_render_tick++;
    
    // Pass 1: Draw Maze Walls (Static)
    glBegin(GL_QUADS);
    glColor4f(0.0f, 0.19f, 0.59f, 0.8f);
    for (int r = 0; r < MAZE_ROWS; ++r) {
      int screen_r = MAZE_ROWS - 1 - r;
      for (int c = 0; c < MAZE_COLS; ++c) {
        if (host_maze[r * MAZE_COLS + c] == 1) {
          float px = map_x + c * (float)minimap_cell;
          float py = map_y + screen_r * (float)minimap_cell;
          glVertex2f(px, py);
          glVertex2f(px + (float)minimap_cell, py);
          glVertex2f(px + (float)minimap_cell, py + (float)minimap_cell);
          glVertex2f(px, py + (float)minimap_cell);
        }
      }
    }
    glEnd();

    // Pass 2: Swarm dots on minimap (Dynamic - Fast)
    glPointSize(2.0f);
    glBegin(GL_POINTS);
    for(int i=0; i<NUM_AGENTS; ++i) {
        float norm_x = (h_compact[i].x + offset_x) / (MAZE_COLS * CELL_SIZE);
        float norm_z = (h_compact[i].z + offset_z) / (MAZE_ROWS * CELL_SIZE);
        float dot_x = map_x + norm_x * (float)map_w;
        float dot_y = map_y + (1.0f - norm_z) * (float)map_h;
        if (i == 0) glColor4f(0.0f, 1.0f, 1.0f, 1.0f);
        else glColor4f(1.0f, 1.0f, 1.0f, 0.6f);
        glVertex2f(dot_x, dot_y);
    }
    glEnd();

    // Pass 3: Heat Map (Slow - Throttled)
    if (minimap_render_tick % 4 == 0) {
      glBegin(GL_QUADS);
      for (int r = 0; r < 64; ++r) {
        int screen_r = 63 - r;
        for (int c = 0; c < 64; ++c) {
          float mem = h_cycle.ai_memory_grid[r][c];
          if (mem > 0.01f) {
            float px = map_x + c * (minimap_cell * 4.0f);
            float py = map_y + screen_r * (minimap_cell * 4.0f);
            glColor4f(fminf(1.0f, mem * 0.08f), fmaxf(0.19f, 1.0f - mem * 0.02f), 0.0f, 0.6f);
            glVertex2f(px, py);
            glVertex2f(px + minimap_cell * 4.0f, py);
            glVertex2f(px + minimap_cell * 4.0f, py + minimap_cell * 4.0f);
            glVertex2f(px, py + minimap_cell * 4.0f);
          }
        }
      }
      glEnd();
    }
    
    // Cycle 0 detailed Pointer & Radar on map
    {
      float norm_x = (h_cycle.x + offset_x) / (MAZE_COLS * CELL_SIZE);
      float norm_z = (h_cycle.z + offset_z) / (MAZE_ROWS * CELL_SIZE);
      float dot_x = map_x + norm_x * (float)map_w;
      float dot_y = map_y + (1.0f - norm_z) * (float)map_h;

      // Draw Breadcrumbs for primary agent
      glPointSize(3.0f);
      glBegin(GL_POINTS);
      glColor4f(1.0f, 0.7f, 0.0f, 0.8f);
      for (int i = 0; i < h_cycle.crumb_count; ++i) {
        int idx = (h_cycle.crumb_idx - 1 - i + 16) % 16;
        float bcx = map_x + ((h_cycle.crumb_x[idx] + offset_x) / (MAZE_COLS * CELL_SIZE)) * (float)map_w;
        float bcy = map_y + (1.0f - ((h_cycle.crumb_z[idx] + offset_z) / (MAZE_ROWS * CELL_SIZE))) * (float)map_h;
        glVertex2f(bcx, bcy);
      }
      glEnd();

      // Radar Signals for primary agent
      glBegin(GL_LINES);
      float map_scale = (float)minimap_cell / CELL_SIZE;
      for (int i = 0; i < h_cycle.active_rays; ++i) {
        float angle = h_cycle.dir + RADAR_ANGLES[i];
        float dist = h_cycle.last_ray_distances[i] * map_scale;
        if (i == h_cycle.best_ray_idx) glColor4f(1.0f, 1.0f, 0.0f, 1.0f);
        else glColor4f(0.0f, 1.0f, 1.0f, 0.6f);
        glVertex2f(dot_x, dot_y);
        glVertex2f(dot_x + sinf(angle) * dist, dot_y + cosf(angle) * dist);
      }
      glEnd();
    }

    // Calculate total heat penalty accumulation (scaled for 64x64)
    float total_heat = 0.0f;
    for (int r = 0; r < 64; ++r) {
      for (int c = 0; c < 64; ++c) {
        total_heat += h_cycle.ai_memory_grid[r][c];
      }
    }

    // AI Thinking status text
    {
      char status_buf[128];
      int tid = h_cycle.thinking_id;
      if (tid == 0) sprintf(status_buf, "AI[0]: Path clear (%.0fu).", h_cycle.front_dist);
      else if (tid == 1) sprintf(status_buf, "AI[0]: Wall at %.0fu! Searching...", h_cycle.front_dist);
      else if (tid == 2) sprintf(status_buf, "AI[0]: Left seems open (%.0fu), going there.", h_cycle.best_dist);
      else if (tid == 3) sprintf(status_buf, "AI[0]: Right seems open (%.0fu), going there.", h_cycle.best_dist);
      else if (tid == 4) sprintf(status_buf, "AI[0]: Panic! Obstruction ahead...");
      else if (tid == 5) sprintf(status_buf, "AI[0]: Panic! Scanning for paths...");
      else if (tid == 6) sprintf(status_buf, "AI[0]: Stuck! Backtracking via breadcrumbs...");
      else if (tid == 7) sprintf(status_buf, "AI[0]: Junction detected! Pausing to decide...");
      else if (tid == 8) sprintf(status_buf, "AI[0]: Blasting through old trails...");
      else if (tid == 9) sprintf(status_buf, "AI[0]: Shuttle detected! Forcing turn...");
      else sprintf(status_buf, "AI[0]: Recalculating... Stuck or Panic!");

      glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
      renderText(status_buf, 20.0f, h - 35.0f);
      
      char info_buf[64];
      sprintf(info_buf, "Speed: %.1f | Learning: %.0f", h_cycle.current_speed, total_heat);
      glColor4f(0.0f, 1.0f, 1.0f, 0.8f);
      renderText(info_buf, 20.0f, h - 60.0f);
    }

    // Restore 3D projection state
    glEnable(GL_DEPTH_TEST);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
    // ======================================================

    // Buffered Telemetry
    char log_line[512];
    sprintf(log_line, "%d,%.2f,%.1f,%.1f,%.1f,%.1f,\"%s\",%d,%d,%d",
            h_cycle.step_count, h_cycle.current_speed, total_heat,
            h_cycle.x, h_cycle.z, fps, (fps < 50.0 ? "DROP" : "OK"),
            h_cycle.thinking_id, h_cycle.active_rays, h_cycle.stuck_timer);
    
    std::string line_str(log_line);
    // Specific rays for 33-ray configuration: 0, 8, 16, 24, 32
    int log_indices[] = {0, 8, 16, 24, 32};
    for (int i = 0; i < 5; ++i) {
        char ray_val[32];
        sprintf(ray_val, ",%.1f", h_cycle.last_ray_distances[log_indices[i]]);
        line_str += ray_val;
    }
    line_str += "\n";
    telemetry_buffer += line_str;
    telemetry_frame_count++;

    if (telemetry_frame_count >= 60) {
        telemetry_file << telemetry_buffer;
        telemetry_file.flush();
        telemetry_buffer.clear();
        telemetry_frame_count = 0;
    }

    SwapBuffers(hdc);
    // Reduced Sleep to 1ms to remove artificial 60fps ceiling
    Sleep(1);
  }

  if (!telemetry_buffer.empty()) telemetry_file << telemetry_buffer;
  telemetry_file.close();
  wglMakeCurrent(NULL, NULL);
  wglDeleteContext(hrc);
  ReleaseDC(hwnd, hdc);
  cudaFree(d_cycles);
  return 0;
}
