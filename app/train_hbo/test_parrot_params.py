import numpy as np
import soundfile as sf
from joblib import load

from simulation.drone_synth_parrot import synthesize_parrot_drone
from app.train_hbo.dataset import load_dataset_single

# ---------------------------------------------------------
# Modell- und Feature-Parameter
# ---------------------------------------------------------
SR = 48000
N_FFT = 2048
HOP = 512
N_HARM = 6   # alte Pipeline → 10 Features

# ---------------------------------------------------------
# Optimizer-Parameter (Best probability: 0.900)
# ---------------------------------------------------------
BEST_PARAMS = {
    "f0": 100.94979220798596,
    "n_harmonics": 15,
    "base_mag": 2.954781330106917,
    "decay_mag": 1.2469952454897135,
    "roughness": 1.6215517269848985,
    "am_depth": 0.758980479273123,
    "fm_depth": 0.29277389672407644,
    "noise_level": 0.05232909739386191,
    "jitter_amount": 0.0570574448577833,
    "drift_amount": 0.023151149455716693,
    "wind_noise": 0.019203448050019126,
}

# ---------------------------------------------------------
# Hauptfunktion
# ---------------------------------------------------------
def main():
    print("[INFO] Lade Modell: models/hbd/uav_band_model.joblib")
    clf = load("models/hbd/uav_band_model.joblib")
    print("[INFO] Modell erfolgreich geladen.\n")

    print("[INFO] Erzeuge WAV-Datei mit Optimizer-Parametern…")
    audio = synthesize_parrot_drone(sr=SR, **BEST_PARAMS)

    out_path = "./sim_parrot_best.wav"
    sf.write(out_path, audio, SR)
    print(f"[INFO] WAV-Datei gespeichert unter: {out_path}\n")

    print("[INFO] Extrahiere Features und klassifiziere…")
    X = load_dataset_single(audio, SR, N_FFT, HOP, N_HARM)

    prob = clf.predict_proba([X])[0][1]
    pred = int(prob >= 0.5)

    print("=== PARROT OPTIMIZER PARAMETER TEST RESULT ===")
    print(f"Predicted class: {pred}")
    print(f"Probability (Drone): {prob:.3f}")
    print("\n[INFO] Test abgeschlossen.")


if __name__ == "__main__":
    main()
