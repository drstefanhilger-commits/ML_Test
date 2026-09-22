#pragma once
#include <cstdint>
#include <cmath>

class SDS_Bandpass {
public:
    SDS_Bandpass();

    // Setzt die Koeffizienten (falls du später dynamisch ändern willst)
    void setCoefficients(const float* b, const float* a);

    // Filtert einen Chunk (in-place)
    void process(float* x, int n_samples);

private:
    // IIR-Koeffizienten
    float b[5];   // b0..b4
    float a[5];   // a0..a4 (a0 = 1)

    // interner Zustand (Pfad-B: explizit, keine globalen Variablen)
    float z1, z2, z3, z4;
};
