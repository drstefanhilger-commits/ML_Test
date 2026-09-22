#include "SDS_SimDrone.hpp"

#define SR 48000.0f

// ---------------------------------------------------------
// Konstruktor
// ---------------------------------------------------------
SDS_SimDrone::SDS_SimDrone()
{
    seed_jitter = 111u;
    seed_drift  = 222u;
    seed_fm     = 333u;

    f0 = 110.0f;
    n_harmonics = 12;
    am_depth = 0.4f;
    fm_depth = 0.1f;
    jitter_amount = 0.03f;
    drift_amount  = 0.02f;
    base_mag = 2.0f;
    decay_mag = 1.2f;
}

// ---------------------------------------------------------
// Parameter setzen
// ---------------------------------------------------------
void SDS_SimDrone::setParams(float f0_,
                             int n_harmonics_,
                             float am_depth_,
                             float fm_depth_,
                             float jitter_amount_,
                             float drift_amount_,
                             float base_mag_,
                             float decay_mag_)
{
    f0 = f0_;
    n_harmonics = n_harmonics_;
    am_depth = am_depth_;
    fm_depth = fm_depth_;
    jitter_amount = jitter_amount_;
    drift_amount = drift_amount_;
    base_mag = base_mag_;
    decay_mag = decay_mag_;
}

// ---------------------------------------------------------
// Deterministischer LCG
// ---------------------------------------------------------
void SDS_SimDrone::lcgNoise(float* out, int n_samples, uint32_t& state)
{
    const uint32_t a = 1103515245u;
    const uint32_t c = 12345u;
    const uint32_t m = 2147483648u;

    for (int i = 0; i < n_samples; i++) {
        state = (a * state + c) % m;
        out[i] = (static_cast<float>(state) / static_cast<float>(m)) * 2.0f - 1.0f;
    }
}

// ---------------------------------------------------------
// Chunk erzeugen
// ---------------------------------------------------------
void SDS_SimDrone::generateChunk(float* out, int n_samples)
{
    std::vector<float> jitter(n_samples);
    std::vector<float> drift(n_samples);
    std::vector<float> fm_noise(n_samples);
    std::vector<float> t(n_samples);
    std::vector<float> am(n_samples);

    // deterministische Noise-Quellen
    lcgNoise(jitter.data(), n_samples, seed_jitter);
    lcgNoise(drift.data(),  n_samples, seed_drift);
    lcgNoise(fm_noise.data(), n_samples, seed_fm);

    for (int i = 0; i < n_samples; i++) {
        jitter[i] *= jitter_amount;
        drift[i]  *= drift_amount;
        fm_noise[i] *= fm_depth;
        t[i] = static_cast<float>(i) / SR;
    }

    // AM-Modulation
    for (int i = 0; i < n_samples; i++) {
        am[i] = 1.0f + am_depth * std::sin(2.0f * M_PI * 7.0f * t[i] + 0.5f);
    }

    // Ausgangssignal initialisieren
    for (int i = 0; i < n_samples; i++)
        out[i] = 0.0f;

    // Rotor-Synthese
    for (int k = 1; k <= n_harmonics; k++) {
        float fk = f0 * k;
        float amp = base_mag / std::pow(k, decay_mag);

        float theta = 0.0f;
        for (int i = 0; i < n_samples; i++) {
            float fk_time = fk * (1.0f + jitter[i] + drift[i]);
            theta += 2.0f * M_PI * fk_time / SR;
            float th = theta + fm_noise[i];
            out[i] += amp * am[i] * std::sin(th);
        }
    }

    // RMS-Normalisierung
    float rms = 0.0f;
    for (int i = 0; i < n_samples; i++)
        rms += out[i] * out[i];

    rms = std::sqrt(rms / n_samples + 1e-9f);

    for (int i = 0; i < n_samples; i++)
        out[i] /= rms;
}
