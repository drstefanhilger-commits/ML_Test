import numpy as np
from joblib import load
import soundfile as sf

from simulation.drone_synth_parrot import synthesize_parrot_drone
from app.train_bpp.bpp_dataset import extract_features_from_wav

SR = 48000
MODEL_PATH = "./models/bpp/uav_bandpass_model.joblib"
TMP_WAV = "./tmp_bpp_sim.wav"


BEST_PARAMS_TEMPLATE = {
    "f0": (90.0, 130.0),
    "n_harmonics": 15,
    "base_mag": (2.0, 3.5),
    "decay_mag": (0.8, 1.5),
    "roughness": (0.5, 2.0),
    "am_depth": (0.2, 0.8),
    "fm_depth": (0.05, 0.3),
    "noise_level": (0.03, 0.15),
    "jitter_amount": (0.02, 0.10),
    "drift_amount": (0.01, 0.06),
    "wind_noise": (0.01, 0.05),
}


def sample_params():
    p = {}
    p["f0"] = np.random.uniform(*BEST_PARAMS_TEMPLATE["f0"])
    p["n_harmonics"] = BEST_PARAMS_TEMPLATE["n_harmonics"]
    p["base_mag"] = np.random.uniform(*BEST_PARAMS_TEMPLATE["base_mag"])
    p["decay_mag"] = np.random.uniform(*BEST_PARAMS_TEMPLATE["decay_mag"])
    p["roughness"] = np.random.uniform(*BEST_PARAMS_TEMPLATE["roughness"])
    p["am_depth"] = np.random.uniform(*BEST_PARAMS_TEMPLATE["am_depth"])
    p["fm_depth"] = np.random.uniform(*BEST_PARAMS_TEMPLATE["fm_depth"])
    p["noise_level"] = np.random.uniform(*BEST_PARAMS_TEMPLATE["noise_level"])
    p["jitter_amount"] = np.random.uniform(*BEST_PARAMS_TEMPLATE["jitter_amount"])
    p["drift_amount"] = np.random.uniform(*BEST_PARAMS_TEMPLATE["drift_amount"])
    p["wind_noise"] = np.random.uniform(*BEST_PARAMS_TEMPLATE["wind_noise"])
    return p


def evaluate_params(params, clf):
    audio = synthesize_parrot_drone(sr=SR, **params)
    sf.write(TMP_WAV, audio, SR)

    X = extract_features_from_wav(TMP_WAV)
    prob = clf.predict_proba([X])[0][1]
    return prob


def main():
    clf = load(MODEL_PATH)
    best_prob = -1.0
    best_params = None

    n_iter = 200

    for i in range(n_iter):
        params = sample_params()
        prob = evaluate_params(params, clf)

        if prob > best_prob:
            best_prob = prob
            best_params = params
            print(f"[UPDATE] Iter {i}: best_prob={best_prob:.3f}")
            print(best_params)

    print("\n=== BPP OPTIMIZER FINAL BEST PARAMETERS ===")
    print(best_params)
    print(f"Best probability (Drone): {best_prob:.3f}")


if __name__ == "__main__":
    main()
