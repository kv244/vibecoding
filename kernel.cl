#ifndef TILE_SIZE
#define TILE_SIZE 256
#endif

__kernel void apply_effects(__global const float4* input, 
                           __global float4* output, 
                           const int effect_type,
                           const float param1,
                           const float param2,
                           const int num_samples,
                           const float sample_rate,
                           const int num_channels) 
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
        float t = gid * 4.0f / (sample_rate * (float)num_channels); 
        float4 offsets;
        if (num_channels == 2) {
            // Stereo sample pairs share the same time instant
            offsets = (float4)(0.0f, 0.0f, 1.0f/sample_rate, 1.0f/sample_rate);
        } else {
            offsets = (float4)(0.0f, 1.0f/sample_rate, 2.0f/sample_rate, 3.0f/sample_rate);
        }
        float4 lfo = 1.0f - param2 + param2 * (float4)(
            sin(6.283185f * param1 * (t + offsets.x)),
            sin(6.283185f * param1 * (t + offsets.y)),
            sin(6.283185f * param1 * (t + offsets.z)),
            sin(6.283185f * param1 * (t + offsets.w))
        );
        sample *= lfo;
    }
    else if (effect_type == 5) {
        // Stereo Widening (Mid/Side)
        // param1 = width (0 to 4+)
        if (num_channels >= 2) {
            float mid = (sample.x + sample.y) * 0.5f;
            float side = (sample.x - sample.y) * 0.5f;
            sample.x = mid + side * param1;
            sample.y = mid - side * param1;

            mid = (sample.z + sample.w) * 0.5f;
            side = (sample.z - sample.w) * 0.5f;
            sample.z = mid + side * param1;
            sample.w = mid - side * param1;
        }
    }
    else if (effect_type == 6) {
        // Ping-Pong Delay (Simplified Cross-Reflection)
        // param1 = delay_samples, param2 = decay
        int delay_vec = (int)round(param1) / 4;
        if (delay_vec < 1) delay_vec = 1; // Prevent zero-delay gain
        if (gid >= delay_vec) {
            float4 prev_sample = input[gid - delay_vec];
            // Swap L/R for ping-pong effect
            sample.xy += prev_sample.yx * param2;
            sample.zw += prev_sample.wz * param2;
        }
    }
    else if (effect_type == 7) {
        // Chorus (Modulated Delay)
        // Simplified: Fixed small sine modulation of read address
        float t = gid * 4.0f / (sample_rate * (float)num_channels);
        float mod = sin(6.283185f * 0.25f * t) * 100.0f + 200.0f; // 0.25Hz sweep
        int delay_vec = (int)mod / 4;
        if (delay_vec < 1) delay_vec = 1; // Guard
        if (gid >= delay_vec) {
            sample += input[gid - delay_vec] * 0.5f;
        }
    }
    else if (effect_type == 8) {
        // Auto-Wah (Modulated Low-pass Approximation)
        float t = gid * 4.0f / (sample_rate * (float)num_channels);
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
        
        float4 prev_v, next_v;
        if (num_channels == 2) {
            // Stereo: Average samples i-2, i, i+2 (L-L-L and R-R-R)
            // Shift vectors by 2 samples
            prev_v = (float4)(tile[lid].z, tile[lid].w, sample.x, sample.y);
            next_v = (float4)(sample.z, sample.w, tile[lid + 2].x, tile[lid + 2].y);
        } else {
            // Mono: Average samples i-1, i, i+1
            // Shift vectors by 1 sample
            prev_v = (float4)(tile[lid].w, sample.x, sample.y, sample.z);
            next_v = (float4)(sample.y, sample.z, sample.w, tile[lid + 2].x);
        }
        float4 filtered = (prev_v + sample + next_v) / 3.0f;
        sample = sample * (1.0f - sweep) + filtered * sweep;
    }

    output[gid] = clamp(sample, -1.0f, 1.0f);
}
