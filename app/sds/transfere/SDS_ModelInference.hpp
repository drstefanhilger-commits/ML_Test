#pragma once
#include "SDS_FeatureExtractor.hpp"

class SDS_ModelInference {
public:
    SDS_ModelInference();

    // Input: 10 Features, Output: Probability (Drone)
    float predict(const SDS_Features& f) const;

private:
    // Beispiel: 10-D Feature-Vektor → internes Array
    void toVector(const SDS_Features& f, float x[10]) const;

    // Hier später: generierter Tree-Code
    float evalTrees(const float x[10]) const;
};
