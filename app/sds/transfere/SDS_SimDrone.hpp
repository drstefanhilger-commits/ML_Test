#pragma once
#include <cstdint>
#include <cmath>
#include <vector>

class SDS_SimDrone {
public:
    SDS_SimDrone();

    void setParams(float f0,
                   int n_harmonics,
                   float am_depth,
                   float fm_depth,
                   float jitter_amount,
                   float drift_amount,
                   float base_mag,
                   float decay_mag);

    void generateChunk(float* out, int n_samples);

private:
    // Seeds für deterministische LCG
    uint32_t seed_jitter;
    uint32_t seed_drift;
    uint32_t seed_fm;

    // Rotor-Parameter
    float f0;
    int   n_harmonics;
    float am_depth;
    float fm_depth;
    float jitter_amount;
    float drift_amount;
    float base_mag;
    float decay_mag;

    // Hilfsfunktionen
    void lcgNoise(float* out, int n_samples, uint32_t& state);
};
