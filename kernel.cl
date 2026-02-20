__kernel void apply_effects(__global const float* input, 
                           __global float* output, 
                           const int effect_type,
                           const float param1,
                           const float param2,
                           const int num_samples) 
{
    int gid = get_global_id(0);
    if (gid >= num_samples) return;

    float sample = input[gid];

    // Effect types: 0=Gain, 1=Echo/Delay, 2=Lowpass, 3=Bitcrush
    if (effect_type == 0) {
        // Gain
        sample *= param1; // param1 = gain
    } 
    else if (effect_type == 1) {
        // Echo/Delay
        int delay_samples = (int)param1; // param1 = delay in samples
        float decay = param2;            // param2 = decay factor
        if (gid >= delay_samples) {
            sample += input[gid - delay_samples] * decay;
        }
    }
    else if (effect_type == 2) {
        // Simple FIR Lowpass (3-tap averaging)
        // param1 = strength (interpolation between original and filtered)
        if (gid > 0 && gid < num_samples - 1) {
            float filtered = (input[gid-1] + input[gid] + input[gid+1]) / 3.0f;
            sample = sample * (1.0f - param1) + filtered * param1;
        }
    }
    else if (effect_type == 3) {
        // Bitcrush
        // param1 = number of bits (e.g. 4.0, 8.0)
        float levels = pown(2.0f, (int)param1);
        sample = round(sample * levels) / levels;
    }

    output[gid] = sample;
}

