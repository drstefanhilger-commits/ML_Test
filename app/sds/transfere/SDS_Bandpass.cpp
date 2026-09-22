#include "SDS_Bandpass.hpp"

// ---------------------------------------------------------
// Standard-Bandpass-Koeffizienten (48 kHz, 4th order Butterworth)
// Beispiel: 200 Hz – 3000 Hz
// Diese Koeffizienten kannst du später ersetzen.
// ---------------------------------------------------------
static const float BP_B[5] = {
    0.01809893f, 0.0f, -0.03619786f, 0.0f, 0.01809893f
};

static const float BP_A[5] = {
    1.0f, -3.36225606f, 4.34856225f, -2.50947845f, 0.52379516f
};

// ---------------------------------------------------------
// Konstruktor
// ---------------------------------------------------------
SDS_Bandpass::SDS_Bandpass()
{
    // Default-Koeffizienten setzen
    for (int i = 0; i < 5; i++) {
        b[i] = BP_B[i];
        a[i] = BP_A[i];
    }

    // Pfad-B: explizite Initialisierung
    z1 = z2 = z3 = z4 = 0.0f;
}

// ---------------------------------------------------------
// Dynamisches Setzen der Koeffizienten
// ---------------------------------------------------------
void SDS_Bandpass::setCoefficients(const float* b_in, const float* a_in)
{
    for (int i = 0; i < 5; i++) {
        b[i] = b_in[i];
        a[i] = a_in[i];
    }
}

// ---------------------------------------------------------
// Direct Form II Transposed IIR-Filter
// ---------------------------------------------------------
void SDS_Bandpass::process(float* x, int n_samples)
{
    for (int i = 0; i < n_samples; i++) {

        float in = x[i];

        // DF2T Struktur
        float out = b[0] * in + z1;

        z1 = b[1] * in + z2 - a[1] * out;
        z2 = b[2] * in + z3 - a[2] * out;
        z3 = b[3] * in + z4 - a[3] * out;
        z4 = b[4] * in        - a[4] * out;

        x[i] = out;
    }
}
