import numpy as np
from joblib import load
from app.train_hbo.dataset import load_dataset_single
from simulation.drone_synth_parrot import synthesize_parrot_drone

SR = 48000
N_FFT = 2048
HOP = 512
N_HARM = 6

def evaluate(params, clf):
    audio = synthesize_parrot_drone(
        sr=SR,
        f0=params["f0"],
        n_harmonics=params["n_harmonics"],
        base_mag=params["base_mag"],
        decay_mag=params["decay_mag"],
        roughness=params["roughness"],
        am_depth=params["am_depth"],
        fm_depth=params["fm_depth"],
        noise_level=params["noise_level"],
        jitter_amount=params["jitter_amount"],
        drift_amount=params["drift_amount"],
        wind_noise=params["wind_noise"],
    )

    X = load_dataset_single(audio, SR, N_FFT, HOP, N_HARM)
    prob = clf.predict_proba([X])[0][1]
    return prob

def main():
    clf = load("models/hbd/uav_band_model.joblib")

    best_prob = -1
    best_params = None

    for i in range(300):
        params = {
            "f0": np.random.uniform(100, 130),
            "n_harmonics": 15,
            "base_mag": np.random.uniform(1.5, 3.0),
            "decay_mag": np.random.uniform(0.8, 1.5),
            "roughness": np.random.uniform(0.5, 2.0),
            "am_depth": np.random.uniform(0.2, 0.8),
            "fm_depth": np.random.uniform(0.05, 0.3),
            "noise_level": np.random.uniform(0.05, 0.2),
            "jitter_amount": np.random.uniform(0.02, 0.12),
            "drift_amount": np.random.uniform(0.01, 0.08),
            "wind_noise": np.random.uniform(0.01, 0.05),
        }

        prob = evaluate(params, clf)

        if prob > best_prob:
            best_prob = prob
            best_params = params
            print(f"[UPDATE] New best: {best_prob:.3f}")
            print(best_params)

    print("\n=== FINAL BEST PARAMETERS ===")
    print(best_params)
    print(f"Best probability: {best_prob:.3f}")

if __name__ == "__main__":
    main()
