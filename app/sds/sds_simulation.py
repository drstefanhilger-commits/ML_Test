import numpy as np
import math


SDS_NUM_MICS = 8
SDS_SAMPLE_RATE = 48000
SDS_FRAME_LEN = 256
c = 343.0

SDS_MIC_POSITIONS = np.array([
    [ 0.03,  0.03],
    [ 0.03, -0.03],
    [-0.03,  0.03],
    [-0.03, -0.03],
    [ 0.05,  0.00],
    [-0.05,  0.00],
    [ 0.00,  0.05],
    [ 0.00, -0.05]
])

def sample_with_delay(src, delay_samples, n):
    i0 = int(n - delay_samples)
    frac = (n - delay_samples) - i0
    if i0 < 0 or i0 + 1 >= len(src):
        return 0.0
    return (1 - frac) * src[i0] + frac * src[i0 + 1]

def make_sds_frame(angleDeg, distance_m, noise):
    theta = math.radians(angleDeg)
    dx = math.cos(theta)
    dy = math.sin(theta)

    A = 1.0 / distance_m

    mic = np.zeros((SDS_NUM_MICS, SDS_FRAME_LEN), dtype=np.float32)

    # Dummy source (wird im Exporter ersetzt)
    src = np.random.randn(SDS_FRAME_LEN).astype(np.float32)

    for ch in range(SDS_NUM_MICS):
        x, y = SDS_MIC_POSITIONS[ch]
        proj = x * dx + y * dy
        tau = proj / c
        delaySamples = tau * SDS_SAMPLE_RATE

        for n in range(SDS_FRAME_LEN):
            s = sample_with_delay(src, delaySamples, n)
            v = A * s
            if noise > 0.0:
                v += (np.random.rand() - 0.5) * noise * A
            mic[ch][n] = v

    return mic
