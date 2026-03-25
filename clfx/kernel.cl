#ifndef TILE_SIZE
#define TILE_SIZE 256
#endif

// --- FFT Helpers ---
static inline float2 complex_mul(float2 a, float2 b) {
    return (float2)(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__kernel void apply_effects(__global const float4* input, 
                           __global float4* output, 
                           const int effect_type,
                           const float param1,
                           const float param2,
                           const int num_samples,
                           const float sample_rate,
                           const int num_channels,
                           __global const float* ir_data,
                           const float mix_amount) 
{
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int wg = get_local_size(0);
    
    // Shared memory tile for local effects (lowpass, autowah)
    __local float4 tile[TILE_SIZE + 2]; // Defined via compiler -D flag

    // --- FFT Shared Memory ---
    // For n=1024, if wg=256, we can handle it.
    // However, if wg is dynamic, we need to be careful.
    // Let's assume n = wg * 4 (since float4)
    __local float2 fft_data[1024]; 
    __local float2 fft_scratch[1024]; // Used for safe bit-reversal and IR FFT

    // Each work-item processes 4 samples
    if (gid * 4 >= num_samples) return;

    float4 sample = input[gid];
    float4 original_sample = sample; // Store for dry/wet mix

    if (effect_type < 14 || effect_type >= 17) {
        // ... (Existing effects) ...
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

            // Lowpass (Correct 3-tap FIR across samples)
            tile[lid + 1] = sample;
            barrier(CLK_LOCAL_MEM_FENCE);
            
            float4 prev = tile[lid];
            float4 next = tile[lid + 2];
            float4 filtered;
            filtered.x = (prev.w + sample.x + sample.y) / 3.0f;
            filtered.y = (sample.x + sample.y + sample.z) / 3.0f;
            filtered.z = (sample.y + sample.z + sample.w) / 3.0f;
            filtered.w = (sample.z + sample.w + next.x) / 3.0f;
            
            sample = sample * (1.0f - param1) + filtered * param1;
        }
        else if (effect_type == 17) {
            // Compressor/Limiter
            // param1: Threshold (0.0-1.0), param2: Ratio (1.0-20.0)
            tile[lid + 1] = sample * sample; // squared for power/rms calculation
            barrier(CLK_LOCAL_MEM_FENCE);

            // Simple local peak detection (max of 4 samples in current vector)
            float current_peak = max(max(fabs(sample.x), fabs(sample.y)), max(fabs(sample.z), fabs(sample.w)));
            
            // Soft envelope (rough approximation for single block)
            float threshold = param1;
            float ratio = max(1.0f, param2);
            
            if (current_peak > threshold && current_peak > 0.0001f) {
                float excess_db = 20.0f * log10(current_peak / threshold);
                float reduced_db = excess_db / ratio;
                float gain = pow(10.0f, (reduced_db - excess_db) / 20.0f);
                sample *= gain;
            }
        } else if (effect_type == 18) {
            // Algorithmic Reverb (Freeverb-style approximation)
            // Parallel Comb Filters (simplified to a single delay for GPU simplicity)
            int delay_smp = (int)(param1 * sample_rate * 0.05f); // 50ms max for comb
            float feedback = param2;
            int read_idx = (( (gid * 4 - delay_smp) % num_samples) + num_samples) % num_samples;
            float4 delayed = input[read_idx / 4];
            sample = sample + delayed * feedback;
        } else if (effect_type == 19) {
            // Flanger
            // Short modulated delay (1-10ms)
            float lfo = (sin(gid * 0.001f) + 1.0f) * 0.5f;
            float delay_ms = 1.0f + lfo * 9.0f;
            int delay_smp = (int)(delay_ms * (sample_rate / 1000.0f));
            int read_idx = (( (gid * 4 - delay_smp) % num_samples) + num_samples) % num_samples;
            float4 delayed = input[read_idx / 4];
            sample = (sample + delayed) * 0.5f; // simple mix
        } else if (effect_type == 20) {
            // Phaser
            // Modulated All-Pass Chain (approx)
            float lfo = (sin(gid * 0.0005f) + 1.0f) * 0.5f;
            float a = lfo * 0.8f;
            // Simple 1st order all-pass: y[n] = a*x[n] + x[n-1] - a*y[n-1]
            // Parallel approximation: notch filter via phase cancellation
            int delay_smp = (int)(lfo * 100.0f * (sample_rate / 44100.0f));
            int read_idx = (( (gid * 4 - delay_smp) % num_samples) + num_samples) % num_samples;
            float4 delayed = input[read_idx / 4];
            sample = (sample + delayed) * 0.5f;
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
            // Autowah (Modulated Filter Approximation)
            float t = gid * 4.0f / (sample_rate * (float)num_channels);
            float sweep = 0.5f + 0.45f * sin(6.283185f * 0.6f * t); 
            
            tile[lid + 1] = sample;
            if (lid == 0) {
                tile[0] = (gid > 0) ? input[gid - 1] : (float4)(0.0f);
            }
            if (lid == wg - 1) {
                tile[wg + 1] = ((gid + 1) * 4 < num_samples) ? input[gid + 1] : (float4)(0.0f);
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            float4 prev = (float4)(tile[lid].w, sample.x, sample.y, sample.z);
            float4 next = (float4)(sample.y, sample.z, sample.w, tile[lid + 2].x);
            
            float4 filtered = (prev + sample + next) / 3.0f;
            // Wah: accentuate moving band between LP and HP
            sample = filtered * (1.0f - sweep) + (sample - filtered) * sweep * 3.0f;
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
            // Pitch Shift (Stereo-Aware Resampling)
            __global const float* fin = (__global const float*)input;
            float4 out_val;
            
            if (num_channels == 2) {
                // Each work-item (gid) handles 2 stereo pairs (L0, R0, L1, R1)
                float base_pair_idx = (gid * 2.0f) * param1;
                
                for (int pair = 0; pair < 2; pair++) {
                    float target_pair = base_pair_idx + (float)pair * param1;
                    int p0 = (int)floor(target_pair);
                    int p1 = p0 + 1;
                    float frac = target_pair - (float)p0;
                    
                    for (int ch = 0; ch < 2; ch++) {
                        int i0 = p0 * 2 + ch;
                        int i1 = p1 * 2 + ch;
                        float s0 = (i0 < num_samples) ? fin[i0] : 0.0f;
                        float s1 = (i1 < num_samples) ? fin[i1] : 0.0f;
                        ((float*)&out_val)[pair * 2 + ch] = s0 + (s1 - s0) * frac;
                    }
                }
            } else {
                // Mono or Fallback
                float base_idx = gid * 4.0f * param1;
                for (int i = 0; i < 4; i++) {
                    float in_idx = base_idx + (float)i * param1;
                    int i0 = (int)floor(in_idx);
                    int i1 = i0 + 1;
                    float frac = in_idx - (float)i0;
                    float s0 = (i0 < num_samples) ? fin[i0] : 0.0f;
                    float s1 = (i1 < num_samples) ? fin[i1] : 0.0f;
                    ((float*)&out_val)[i] = s0 + (s1 - s0) * frac;
                }
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
    } 
    else {
        // --- Spectral Processing Block ---
        // Each work-group processes a 1024-sample block
        const int n = 1024;

        // 1. Data Loading to Local Memory & Bit-reversal Permutation (Safe via scatter to scratch)
        // Each work-item loads 4 samples and performs bit-reversal for them.
        for (int i = 0; i < 4; i++) {
            int idx = lid * 4 + i; // Current index in the 1024-sample block
            float s = 0.0f;
            if (idx < n) {
                // Map the local index (0-1023) back to the global input float4
                // gid is the float4 index for the current work-item's 'sample'
                // We need to read from the input array based on the overall block index.
                // The start of the current block in global input is (get_group_id() * n / 4)
                // The sample.x,y,z,w corresponds to input[gid]
                // So, we need to read from input[get_group_id() * (n/4) + (idx/4)]
                // This assumes gid is relative to the start of the block.
                // Let's simplify: the 'sample' variable already holds the 4 samples for 'gid'.
                // We need to distribute these 4 samples into fft_scratch based on 'idx'.
                if (i == 0) s = sample.x;
                else if (i == 1) s = sample.y;
                else if (i == 2) s = sample.z;
                else s = sample.w;
            }
            
            int m = idx;
            int rev = 0;
            // 1024 = 2^10
            #pragma unroll
            for (int k = 0; k < 10; k++) {
                rev = (rev << 1) | (m & 1);
                m >>= 1;
            }
            fft_scratch[rev] = (float2)(s, 0.0f);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // Copy back to fft_data
        for (int i = lid; i < n; i += wg) fft_data[i] = fft_scratch[i];
        barrier(CLK_LOCAL_MEM_FENCE);

        // 2. Iterative FFT (Optimized Twiddle Precomputation)
        for (int len = 2; len <= n; len <<= 1) {
            float ang = -6.2831853f / len;
            for (int j = lid; j < len / 2; j += wg) {
                // Twiddle is same for all butterflies in this stage at offset j
                float2 w = (float2)(cos(ang * j), sin(ang * j));
                for (int i = 0; i < n; i += len) {
                    int idxA = i + j;
                    int idxB = i + j + len / 2;
                    float2 u = fft_data[idxA];
                    float2 v = complex_mul(fft_data[idxB], w);
                    fft_data[idxA] = u + v;
                    fft_data[idxB] = u - v;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // 3. Spectral Effects
        // EQ: param1 = center freq norm (0-1), param2 = gain
        if (effect_type == 14) {
            for (int i = lid; i < n/2; i += wg) {
                float f = (float)i / (n/2);
                if (fabs(f - param1) < 0.2f) {
                    fft_data[i] *= param2;
                    if (i > 0) fft_data[n-i] *= param2; // Safe symmetric
                }
            }
        } else if (effect_type == 15) {
            // Spectral Freeze: Randomize phase to "smear" transients
            uint seed = gid + lid;
            for (int i = lid; i < n; i += wg) {
                float mag = length(fft_data[i]);
                seed = seed * 1103515245 + 12345;
                float phase = (float)(seed % 1000) / 1000.0f * 6.2831853f;
                fft_data[i] = (float2)(mag * cos(phase), mag * sin(phase));
            }
        } else if (effect_type == 16 && ir_data != NULL) {
            // Convolution: Pointwise complex multiply with pre-FFT'd IR
            __global const float2* ir_spectral = (__global const float2*)ir_data;
            for (int i = lid; i < n; i += wg) {
                fft_data[i] = complex_mul(fft_data[i], ir_spectral[i]);
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // 5. IFFT Stages
        for (int len = 2; len <= n; len <<= 1) {
            float ang = 6.2831853f / len;
            for (int i = 0; i < n; i += len) {
                for (int j = lid; j < len / 2; j += wg) {
                    float2 w = (float2)(cos(ang * j), sin(ang * j));
                    int idxA = i + j;
                    int idxB = i + j + len / 2;
                    float2 u = fft_data[idxA];
                    float2 v = complex_mul(fft_data[idxB], w);
                    fft_data[idxA] = u + v;
                    fft_data[idxB] = u - v;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        
        // IFFT back to time domain
        sample.x = fft_data[lid * 4 + 0].x / n;
        sample.y = fft_data[lid * 4 + 1].x / n;
        sample.z = fft_data[lid * 4 + 2].x / n;
        sample.w = fft_data[lid * 4 + 3].x / n;
    }

    // Global Dry/Wet Mix
    sample = mix(original_sample, sample, mix_amount);

    output[gid] = clamp(sample, -1.0f, 1.0f);
}
