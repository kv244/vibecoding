#ifndef TILE_SIZE
#define TILE_SIZE 256
#endif

__kernel void apply_effects(__global const float4* input, 
                           __global float4* output, 
                           const int effect_type,
                           const float param1,
                           const float param2,
                           const int num_samples) 
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

    output[gid] = clamp(sample, -1.0f, 1.0f);
}

