#include "arm_math.h"

float compute_f0_cepstrum(const float *audio, int len, float sr) {
    static float spectrum[1024];
    static float log_spectrum[1024];
    static float cepstrum[1024];

    arm_rfft_fast_instance_f32 S;
    arm_rfft_fast_init_f32(&S, 1024);

    arm_rfft_fast_f32(&S, (float*)audio, spectrum, 0);

    for (int i = 0; i < 1024; i++)
        log_spectrum[i] = logf(fabsf(spectrum[i]) + 1e-9f);

    arm_rfft_fast_f32(&S, log_spectrum, cepstrum, 0);

    int idx = 10;
    float maxv = cepstrum[10];
    for (int i = 11; i < 2000; i++) {
        if (cepstrum[i] > maxv) {
            maxv = cepstrum[i];
            idx = i;
        }
    }

    float q = (float)idx / sr;
    return 1.0f / q;
}
