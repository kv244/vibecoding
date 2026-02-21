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
    else if (effect_type == 9) {
        // Distortion (Tanh Saturation)
        // param1 = drive (> 1.0)
        sample = tanh(sample * param1);
    }
    else if (effect_type == 10) {
        // Ring Modulation
        // param1 = Carrier Freq
        float t = gid * 4.0f / (sample_rate * (float)num_channels);
        float4 offsets;
        if (num_channels == 2) {
            offsets = (float4)(0.0f, 0.0f, 1.0f/sample_rate, 1.0f/sample_rate);
        } else {
            offsets = (float4)(0.0f, 1.0f/sample_rate, 2.0f/sample_rate, 3.0f/sample_rate);
        }
        float4 carrier = (float4)(
            sin(6.283185f * param1 * (t + offsets.x)),
            sin(6.283185f * param1 * (t + offsets.y)),
            sin(6.283185f * param1 * (t + offsets.z)),
            sin(6.283185f * param1 * (t + offsets.w))
        );
        sample *= carrier;
    }
    else if (effect_type == 11) {
        // Pitch Shift (Resampling via Linear Interpolation)
        // param1 = ratio (0.5 to 2.0)
        __global const float* fin = (__global const float*)input;
        float base_idx = gid * 4.0f * param1;
        float4 out_val;
        
        // We handle each of the 4 samples in the float4
        for (int i = 0; i < 4; i++) {
            float in_idx = base_idx + (float)i * param1;
            int i0 = (int)floor(in_idx);
            int i1 = i0 + 1;
            float frac = in_idx - (float)i0;
            
            float s0 = (i0 < num_samples) ? fin[i0] : 0.0f;
            float s1 = (i1 < num_samples) ? fin[i1] : 0.0f;
            
            // Re-map index addressing to the component
            if (i == 0) out_val.x = s0 + frac * (s1 - s0);
            else if (i == 1) out_val.y = s0 + frac * (s1 - s0);
            else if (i == 2) out_val.z = s0 + frac * (s1 - s0);
            else if (i == 3) out_val.w = s0 + frac * (s1 - s0);
        }
        sample = out_val;
    }
    else if (effect_type == 12) {
        // Noise Gate
        // param1 = threshold, param2 = reduction (0.0 = mute)
        float4 abs_s = fabs(sample);
        if (abs_s.x < param1) sample.x *= param2;
        if (abs_s.y < param1) sample.y *= param2;
        if (abs_s.z < param1) sample.z *= param2;
        if (abs_s.w < param1) sample.w *= param2;
    }
    else if (effect_type == 13) {
        // Stereo Panning / Balance
        // param1 = pan (-1.0 to 1.0)
        if (num_channels == 2) {
            float left_gain = clamp(1.0f - param1, 0.0f, 1.0f);
            float right_gain = clamp(1.0f + param1, 0.0f, 1.0f);
            sample.x *= left_gain;  // L
            sample.y *= right_gain; // R
            sample.z *= left_gain;  // L
            sample.w *= right_gain; // R
        }
    }

    output[gid] = clamp(sample, -1.0f, 1.0f);
}
