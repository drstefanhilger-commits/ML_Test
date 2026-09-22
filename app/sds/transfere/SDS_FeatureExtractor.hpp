#pragma once
#include <cstdint>

struct SDS_Features {
    float f0;                 // Feature 1
    float f0_energy;          // Feature 2
    float harm2_energy;       // Feature 3
    float harm3_energy;       // Feature 4
    float harm4_energy;       // Feature 5
    float harm5_energy;       // Feature 6
    float harm6_energy;       // Feature 7
    float spectral_centroid;  // Feature 8
    float spectral_flatness;  // Feature 9
    float broadband_energy;   // Feature 10
};

class SDS_FeatureExtractor {
public:
    SDS_FeatureExtractor(float sr, int fft_size);

    // Hauptfunktion: extrahiert alle 10 Features aus einem Chunk
    void extract(const float* x, int n_samples, SDS_Features& out);

private:
    float sr;
    int   fft_size;

    // Hilfsfunktionen
    float estimateF0(const float* x, int n_samples);
    float bandEnergyAround(const float* mag, int n_bins, float f_center, float bw_hz);
    void  computeSpectrum(const float* x, int n_samples, float* mag, int& n_bins);
    float computeSpectralCentroid(const float* mag, int n_bins);
    float computeSpectralFlatness(const float* mag, int n_bins);
    float computeBroadbandEnergy(const float* x, int n_samples);
};
