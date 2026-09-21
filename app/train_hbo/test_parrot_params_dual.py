import numpy as np
import soundfile as sf
from joblib import load

from simulation.drone_synth_parrot import synthesize_parrot_drone
from app.train_hbo.dataset import load_dataset_single

SR = 48000
N_FFT = 2048
HOP = 512
N_HARM = 6

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

def main():
    clf = load("models/hbd/uav_band_model.joblib")

    # -----------------------------------------
    # A) Direkt testen (wie Optimizer)
    # -----------------------------------------
    audio = synthesize_parrot_drone(sr=SR, **BEST_PARAMS)
    X_direct = load_dataset_single(audio, SR, N_FFT, HOP, N_HARM)
    prob_direct = clf.predict_proba([X_direct])[0][1]

    # -----------------------------------------
    # B) WAV speichern + laden (wie dein Test)
    # -----------------------------------------
    out_path = "./sim_parrot_best.wav"
    sf.write(out_path, audio, SR)

    audio_wav, _ = sf.read(out_path)
    X_wav = load_dataset_single(audio_wav, SR, N_FFT, HOP, N_HARM)
    prob_wav = clf.predict_proba([X_wav])[0][1]

    # -----------------------------------------
    # Ausgabe
    # -----------------------------------------
    print("=== PARROT OPTIMIZER PARAMETER TEST RESULT ===")
    print(f"Probability (direct): {prob_direct:.3f}")
    print(f"Probability (wav):    {prob_wav:.3f}")
    print("\n[INFO] Test abgeschlossen.")

if __name__ == "__main__":
    main()
