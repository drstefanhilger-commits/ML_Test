import numpy as np
from joblib import load
from simulation.drone_synth_parrot import synthesize_parrot_drone
from app.train_hbo.dataset import load_dataset_single

sr = 48000
n_fft = 2048
hop = 512
n_harm = 6   # alter Zustand: 6 Harmonische → 10 Features

print("[INFO] Lade Modell: models/hbd/uav_band_model.joblib")
clf = load("models/hbd/uav_band_model.joblib")
print("[INFO] Modell erfolgreich geladen.\n")

audio_sim = synthesize_parrot_drone(
    sr=sr,
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
)

X = load_dataset_single(audio_sim, sr, n_fft, hop, n_harm)

prob = clf.predict_proba([X])[0][1]
pred = int(prob >= 0.5)

print("=== PARROT SIM-DRONE TEST RESULT ===")
print(f"Predicted class: {pred}")
print(f"Probability (Drone): {prob:.3f}\n")
print("[INFO] Test abgeschlossen.")
