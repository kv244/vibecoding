#ifndef TILE_SIZE
#define TILE_SIZE 256
#endif

__kernel void apply_effects(__global const float4* input, 
                           __global float4* output, 
                           const int effect_type,
                           const float param1,
                           const float param2,
                           const int num_samples,
                           const float sample_rate) 
{
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int wg  = get_local_size(0);
    __local float4 tile[TILE_SIZE + 2]; // Defined via compiler -D flag

    // Each work-item processes 4 samples
    if (gid * 4 >= num_samples) return;

    float4 sample = input[gid];

    if (effect_type == 0) {
        // Gain (Vectorized)
        sample *= param1; 
    } 
    else if (effect_type == 1) {
        // Simple Delay (One-Shot Reflection)
        int delay_vec = (int)round(param1) / 4; 
        if (gid >= delay_vec) {
            sample += input[gid - delay_vec] * param2;
        }
    }
    else if (effect_type == 2) {
        // Local Memory Lowpass (Vectorized & Cached)
        tile[lid + 1] = sample;
        if (lid == 0) {
            tile[0] = (gid > 0) ? input[gid - 1] : (float4)(0.0f);
        }
        if (lid == wg - 1) {
            tile[wg + 1] = ((gid + 1) * 4 < num_samples) ? input[gid + 1] : (float4)(0.0f);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Vectorized 3-tap averaging (simple approximation)
        float4 prev = (float4)(tile[lid].w, sample.x, sample.y, sample.z);
        float4 next = (float4)(sample.y, sample.z, sample.w, tile[lid + 2].x);
        float4 filtered = (prev + sample + next) / 3.0f;
        
        sample = sample * (1.0f - param1) + filtered * param1;
    }
    else if (effect_type == 3) {
        // Bitcrush (Simplified)
        // param1 is now pre-computed 'levels' from host
        sample = round(sample * param1) / param1;
    }
    else if (effect_type == 4) {
        // Tremolo (Amplitude Modulation)
        // param1 = freq (Hz), param2 = depth (0-1)
        float t = gid * 4.0f / sample_rate; // Dynamic time
        float4 lfo = 1.0f - param2 + param2 * (float4)(
            sin(6.283185f * param1 * t),
            sin(6.283185f * param1 * (t + 1.0f/sample_rate)),
            sin(6.283185f * param1 * (t + 2.0f/sample_rate)),
            sin(6.283185f * param1 * (t + 3.0f/sample_rate))
        );
        sample *= lfo;
    }
    else if (effect_type == 5) {
        // Stereo Widening (Mid/Side)
        // param1 = width (1.0 = normal, >1.0 = wider)
        // For float4, we assume [L, R, L, R] interleaving
        float2 mid  = (sample.s02 + sample.s13) * 0.5f;
        float2 side = (sample.s02 - sample.s13) * 0.5f * param1;
        sample.s02 = mid + side;
        sample.s13 = mid - side;
    }
    else if (effect_type == 6) {
        // Ping-Pong Delay (Simplified Cross-Reflection)
        // param1 = delay_samples, param2 = decay
        int delay_vec = (int)round(param1) / 4;
        if (gid >= delay_vec) {
            float4 prev = input[gid - delay_vec];
            // Swap L/R in the reflection
            sample += (float4)(prev.y, prev.x, prev.w, prev.z) * param2;
        }
    }
    else if (effect_type == 7) {
        // Chorus (Modulated Delay)
        // Simplified: Fixed small sine modulation of read address
        float t = gid * 4.0f / sample_rate;
        float mod = sin(6.283185f * 0.25f * t) * 100.0f + 200.0f; // 0.25Hz sweep
        int delay_vec = (int)mod / 4;
        if (gid >= delay_vec) {
            sample = (sample + input[gid - delay_vec]) * 0.6f;
        }
    }
    else if (effect_type == 8) {
        // Auto-Wah (Modulated Low-pass Approximation)
        float t = gid * 4.0f / sample_rate;
        float sweep = 0.5f + 0.4f * sin(6.283185f * 2.0f * t); // 2Hz sweep
        // Use our existing filtered logic with a sweeping param
        tile[lid + 1] = sample;
        if (lid == 0) {
            tile[0] = (gid > 0) ? input[gid - 1] : (float4)(0.0f);
        }
        if (lid == wg - 1) {
            tile[wg + 1] = ((gid + 1) * 4 < num_samples) ? input[gid + 1] : (float4)(0.0f);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        float4 prev_v = (float4)(tile[lid].w, sample.x, sample.y, sample.z);
        float4 next_v = (float4)(sample.y, sample.z, sample.w, tile[lid + 2].x);
        float4 filtered = (prev_v + sample + next_v) / 3.0f;
        sample = sample * (1.0f - sweep) + filtered * sweep;
    }

    output[gid] = clamp(sample, -1.0f, 1.0f);
}
