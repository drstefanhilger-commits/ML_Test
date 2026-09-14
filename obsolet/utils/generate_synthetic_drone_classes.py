import numpy as np
import soundfile as sf
import os
import math

# ----------------------------------------
# CONFIG
# ----------------------------------------
SAMPLE_RATE = 16000
FRAME_LEN = 4096
c = 343.0

# Mikrofonpositionen (SDS)
MIC_POS = np.array([
    [ 0.03,  0.03],
    [ 0.03, -0.03],
    [-0.03,  0.03],
    [-0.03, -0.03],
    [ 0.05,  0.00],
    [-0.05,  0.00],
    [ 0.00,  0.05],
    [ 0.00, -0.05]
])

# ----------------------------------------
# Synthetic drone source
# ----------------------------------------
def generate_sds_drone_source(
        f0,
        harmonics,
        harmonic_gain,
        amp_mod_freq=12.0,
        amp_mod_depth=0.3,
        freq_mod_freq=3.0,
        freq_mod_depth=0.02,
        noise_level=0.05
    ):
    t = np.arange(FRAME_LEN) / SAMPLE_RATE

    fm = freq_mod_depth * np.sin(2 * np.pi * freq_mod_freq * t)
    f_inst = f0 * (1.0 + fm)

    sig = np.zeros_like(t, dtype=np.float32)

    for k, g in zip(harmonics, harmonic_gain):
        sig += g * np.sin(2 * np.pi * f_inst * k * t)

    am = 1.0 + amp_mod_depth * np.sin(2 * np.pi * amp_mod_freq * t)
    sig *= am

    sig += noise_level * np.random.randn(len(t))
    sig /= np.max(np.abs(sig) + 1e-6)

    return sig.astype(np.float32)

# ----------------------------------------
# Fractional delay
# ----------------------------------------
def sample_with_delay(src, delay_samples, n):
    i0 = int(n - delay_samples)
    frac = (n - delay_samples) - i0

    if i0 < 0 or i0 + 1 >= len(src):
        return 0.0

    return (1 - frac) * src[i0] + frac * src[i0 + 1]

# ----------------------------------------
# SDS far-field simulation
# ----------------------------------------
def make_sds_frame(src_frame, angleDeg, distance_m, noise):
    theta = math.radians(angleDeg)
    dx = math.cos(theta)
    dy = math.sin(theta)

    A = 1.0 / distance_m

    mic = np.zeros((8, FRAME_LEN), dtype=np.float32)

    for ch in range(8):
        x, y = MIC_POS[ch]
        proj = x * dx + y * dy
        tau = proj / c
        delaySamples = tau * SAMPLE_RATE

        for n in range(FRAME_LEN):
            s = sample_with_delay(src_frame, delaySamples, n)
            v = A * s
            if noise > 0.0:
                v += (np.random.rand() - 0.5) * noise * A
            mic[ch][n] = v

    return mic

# ----------------------------------------
# WAV writer
# ----------------------------------------
def write_wav(path, audio):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, audio, SAMPLE_RATE)
    print(f"[OK] {path} geschrieben. Samples={len(audio)}")

# ----------------------------------------
# Generate multiple drone classes
# ----------------------------------------
def generate_drone_classes():
    np.random.seed(12345)

    classes = [
        ("class_A", 160.0, [1,2,3], [1.0,0.5,0.25], 0.02),
        ("class_B", 180.0, [1,2,3,4], [1.0,0.6,0.35,0.2], 0.05),
        ("class_C", 220.0, [1,3,5], [1.0,0.4,0.2], 0.03),
        ("class_D", 260.0, [1,2], [1.0,0.3], 0.01),
        ("class_E", 300.0, [1,2,3,4,5], [1.0,0.7,0.4,0.2,0.1], 0.04),
    ]

    distances = [50, 100, 150, 200]
    angles = [0, 45, 90, 135, 180, 225, 270, 315]

    for cname, f0, harms, gains, noise_level in classes:
        print(f"\n=== Erzeuge {cname} ===")

        base_dir = f"data/simulation/{cname}"

        for dist in distances:
            for ang in angles:
                src = generate_sds_drone_source(
                    f0=f0,
                    harmonics=harms,
                    harmonic_gain=gains,
                    noise_level=noise_level
                )

                sds = make_sds_frame(src, ang, dist, noise=0.01)
                mono = sds.mean(axis=0)

                filename = f"{cname}_f{int(f0)}_d{dist}m_a{ang}.wav"
                full_path = os.path.join(base_dir, filename)

                write_wav(full_path, mono)

# ----------------------------------------
# MAIN
# ----------------------------------------
if __name__ == "__main__":
    generate_drone_classes()
