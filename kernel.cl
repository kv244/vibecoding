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
        sample *= param1; 
    } 
    else if (effect_type == 1) {
        // Simple Delay (One-Shot Reflection)
        // NOTE: True feedback echo requires serial processing or multiple passes.
        // Reading from 'input' ensures deterministic parallel execution but
        // lacks the 'echo-of-echo' feedback loop.
        int delay_samples = (int)round(param1); 
        float decay = param2;            
        if (gid >= delay_samples) {
            sample += input[gid - delay_samples] * decay;
        }
    }
    else if (effect_type == 2) {
        // Simple FIR Lowpass (3-tap averaging)
        if (gid > 0 && gid < num_samples - 1) {
            float filtered = (input[gid-1] + input[gid] + input[gid+1]) / 3.0f;
            sample = sample * (1.0f - param1) + filtered * param1;
        }
    }
    else if (effect_type == 3) {
        // Bitcrush
        // Use round() for param1 to avoid float imprecision issues with (int) cast
        float levels = pown(2.0f, (int)round(param1));
        sample = round(sample * levels) / levels;
    }

    // Clipping: Ensure signal stays in [-1.0, 1.0] to prevent intermediate distortion
    output[gid] = clamp(sample, -1.0f, 1.0f);
}

