import numpy as np

def synthesize_parrot_drone(
    sr=48000,
    duration=1.0,
    f0=118.0,
    n_harmonics=15,
    base_mag=2.2,
    decay_mag=1.2,
    roughness=1.47,
    am_depth=0.48,
    fm_depth=0.15,
    noise_level=0.15,
    jitter_amount=0.068,
    drift_amount=0.05,
    wind_noise=0.023,
):
    N = int(sr * duration)
    t = np.arange(N) / sr

    signal = np.zeros_like(t)

    drift = 1.0 + drift_amount * np.sin(2 * np.pi * 0.12 * t)

    for k in range(1, n_harmonics + 1):
        fk = f0 * k

        am = 1.0 + am_depth * np.sin(2 * np.pi * f0 * t)
        fm = fk * (1.0 + fm_depth * np.sin(2 * np.pi * 0.8 * t))

        jitter = 1.0 + jitter_amount * np.random.randn()

        fk_eff = fm * jitter * drift

        ak = base_mag / k

        signal += ak * am * np.sin(2 * np.pi * fk_eff * t)

    noise = np.random.randn(N)
    alpha = 0.2
    for i in range(1, N):
        noise[i] = alpha * noise[i] + (1 - alpha) * noise[i - 1]

    signal += decay_mag * roughness * noise
    signal += wind_noise * np.random.randn(N)

    signal /= np.max(np.abs(signal) + 1e-12)

    return signal
