#include "SDS_FeatureExtractor.hpp"
#include <cmath>
#include <vector>

SDS_FeatureExtractor::SDS_FeatureExtractor(float sr_, int fft_size_)
    : sr(sr_), fft_size(fft_size_)
{}

// Hauptfunktion
void SDS_FeatureExtractor::extract(const float* x, int n_samples, SDS_Features& out)
{
    // 1) F0 schätzen
    out.f0 = estimateF0(x, n_samples);

    // 2) Spektrum berechnen
    std::vector<float> mag(fft_size / 2);
    int n_bins = 0;
    computeSpectrum(x, n_samples, mag.data(), n_bins);

    // 3) F0- und Harmonic-Energien
    out.f0_energy    = bandEnergyAround(mag.data(), n_bins, out.f0,        10.0f);
    out.harm2_energy = bandEnergyAround(mag.data(), n_bins, out.f0 * 2.0f, 10.0f);
    out.harm3_energy = bandEnergyAround(mag.data(), n_bins, out.f0 * 3.0f, 10.0f);
    out.harm4_energy = bandEnergyAround(mag.data(), n_bins, out.f0 * 4.0f, 10.0f);
    out.harm5_energy = bandEnergyAround(mag.data(), n_bins, out.f0 * 5.0f, 10.0f);
    out.harm6_energy = bandEnergyAround(mag.data(), n_bins, out.f0 * 6.0f, 10.0f);

    // 4) Spektrale Kennwerte
    out.spectral_centroid = computeSpectralCentroid(mag.data(), n_bins);
    out.spectral_flatness = computeSpectralFlatness(mag.data(), n_bins);

    // 5) Breitbandenergie
    out.broadband_energy = computeBroadbandEnergy(x, n_samples);
}

// --- Platzhalter-Implementationen (du ersetzt sie durch deine BPP-Logik / CMSIS-DSP) ---

float SDS_FeatureExtractor::estimateF0(const float* x, int n_samples)
{
    // Autokorrelation-basierte F0-Schätzung (Stub)
    // Hier später: exakt wie in BPP (Lag-Suche, Peak, sr / Lag)
    return 120.0f;
}

void SDS_FeatureExtractor::computeSpectrum(const float* x, int n_samples,
                                           float* mag, int& n_bins)
{
    // FFT + Betragsspektrum (Stub)
    // Hier später: CMSIS-DSP FFT, Windowing, Magnitude
    n_bins = fft_size / 2;
    for (int i = 0; i < n_bins; i++)
        mag[i] = 0.0f;
}

float SDS_FeatureExtractor::bandEnergyAround(const float* mag, int n_bins,
                                             float f_center, float bw_hz)
{
    float df = sr / static_cast<float>(fft_size);
    int k_center = static_cast<int>(f_center / df);
    int k_bw = static_cast<int>(bw_hz / df);

    int k0 = std::max(0, k_center - k_bw);
    int k1 = std::min(n_bins - 1, k_center + k_bw);

    float e = 0.0f;
    for (int k = k0; k <= k1; k++)
        e += mag[k] * mag[k];
    return e;
}

float SDS_FeatureExtractor::computeSpectralCentroid(const float* mag, int n_bins)
{
    float df = sr / static_cast<float>(fft_size);
    float num = 0.0f;
    float den = 0.0f;
    for (int k = 0; k < n_bins; k++) {
        float f = df * k;
        float m = mag[k];
        num += f * m;
        den += m;
    }
    return (den > 0.0f) ? (num / den) : 0.0f;
}

float SDS_FeatureExtractor::computeSpectralFlatness(const float* mag, int n_bins)
{
    float geo = 0.0f;
    float arith = 0.0f;
    int count = 0;

    for (int k = 0; k < n_bins; k++) {
        float m = mag[k] + 1e-9f;
        geo += std::log(m);
        arith += m;
        count++;
    }

    geo = std::exp(geo / count);
    arith /= count;

    return (arith > 0.0f) ? (geo / arith) : 0.0f;
}

float SDS_FeatureExtractor::computeBroadbandEnergy(const float* x, int n_samples)
{
    float e = 0.0f;
    for (int i = 0; i < n_samples; i++)
        e += x[i] * x[i];
    return e / n_samples;
}
