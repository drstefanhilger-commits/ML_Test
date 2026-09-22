#include "SDS_ModelInference.hpp"

SDS_ModelInference::SDS_ModelInference()
{
    // keine Hidden States, Modell ist rein stateless
}

void SDS_ModelInference::toVector(const SDS_Features& f, float x[10]) const
{
    x[0] = f.f0;
    x[1] = f.f0_energy;
    x[2] = f.harm2_energy;
    x[3] = f.harm3_energy;
    x[4] = f.harm4_energy;
    x[5] = f.harm5_energy;
    x[6] = f.harm6_energy;
    x[7] = f.spectral_centroid;
    x[8] = f.spectral_flatness;
    x[9] = f.broadband_energy;
}

float SDS_ModelInference::evalTrees(const float x[10]) const
{
    // Platzhalter: hier kommt der generierte GBM-Code hin
    // z.B. Summe über 200 Trees, jeder Tree als if-Block
    // aktuell: Dummy-Score
    float score = 0.0f;

    // Beispiel: Bias
    score += -0.2f;

    // hier später: echte Tree-Logik

    return score;
}

float SDS_ModelInference::predict(const SDS_Features& f) const
{
    float x[10];
    toVector(f, x);

    float score = evalTrees(x);

    // Logistic-Link: Probability = 1 / (1 + exp(-score))
    float prob = 1.0f / (1.0f + std::exp(-score));
    return prob;
}
