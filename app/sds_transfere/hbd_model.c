float hbd_infer(const float *x) {
    // Layer 1: 12 → 128
    float h1[128];
    for (int i = 0; i < 128; i++) {
        float s = bias1[i];
        for (int j = 0; j < 12; j++)
            s += w1[i][j] * x[j];
        h1[i] = s > 0 ? s : 0; // ReLU
    }

    // Layer 2: 128 → 64
    float h2[64];
    for (int i = 0; i < 64; i++) {
        float s = bias2[i];
        for (int j = 0; j < 128; j++)
            s += w2[i][j] * h1[j];
        h2[i] = s > 0 ? s : 0;
    }

    // Output: 64 → 1 (sigmoid)
    float s = bias3[0];
    for (int j = 0; j < 64; j++)
        s += w3[0][j] * h2[j];

    return 1.0f / (1.0f + expf(-s));
}
