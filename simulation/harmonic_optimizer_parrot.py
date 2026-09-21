import os
import numpy as np
import joblib
import yaml

from app.train_hbo.dataset import load_dataset_single
from simulation.drone_synth_parrot import synthesize_parrot_drone


class ParrotHarmonicOptimizer:
    """
    Harmonic-Optimizer für Parrot-Drohnen-Synthese.
    Optimiert gezielt:
      - base_mag (Gesamt-Harmonische Energie)
      - decay_mag (Abfall der Rotor-Roughness)
      - n_harmonics (Anzahl der Harmonischen)
    und speichert immer das beste Ergebnis.
    """

    def __init__(self, model_path, config_path, target_prob=0.95):
        self.target_prob = target_prob

        # Modell laden
        self.clf = joblib.load(model_path)

        # Config laden
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        self.sr = cfg["audio"]["sr"]
        self.n_fft = cfg["audio"]["n_fft"]
        self.hop = cfg["audio"]["hop_length"]
        self.n_harm = cfg["audio"]["n_harmonics"]

        # Basis-Parameter (aus deinem besten Ergebnis)
        self.base_params = {
            "f0": 118.0,
            "n_harmonics": 15,
            "base_mag": 2.05,
            "decay_mag": 1.16,
            "roughness": 1.47,
            "am_depth": 0.48,
            "fm_depth": 0.15,
            "noise_level": 0.15,
            "jitter_amount": 0.068,
            "drift_amount": 0.05,
            "wind_noise": 0.023,
        }

        self.best_prob = 0.0
        self.best_params = self.base_params.copy()

    def evaluate(self, params):
        audio = synthesize_parrot_drone(sr=self.sr, **params)
        X = load_dataset_single(audio, self.sr, self.n_fft, self.hop, self.n_harm)
        return self.clf.predict_proba([X])[0][1]

    def update_best(self, prob, params):
        if prob > self.best_prob:
            self.best_prob = prob
            self.best_params = params.copy()
            print(f"*** Neues Bestes Ergebnis: P={prob:.3f} ***")

    def optimize(self):
        print("\n=== PARROT HARMONIC-OPTIMIZER START ===")
        print(f"Ziel: P(Drone) > {self.target_prob}")

        # Suchräume um dein aktuelles Optimum
        base_mag_values = np.linspace(1.6, 2.4, 5)      # Gesamt-Harmonische Energie
        decay_mag_values = np.linspace(0.9, 1.5, 5)     # Rotor-Roughness-Abfall
        n_harmonics_values = [12, 13, 14, 15, 16, 17]   # Anzahl Harmonischer

        for nh in n_harmonics_values:
            for bm in base_mag_values:
                for dm in decay_mag_values:
                    params = self.base_params.copy()
                    params["n_harmonics"] = int(nh)
                    params["base_mag"] = float(bm)
                    params["decay_mag"] = float(dm)

                    prob = self.evaluate(params)
                    print(
                        f"Test: n_h={nh}, base_mag={bm:.2f}, decay_mag={dm:.2f} → P={prob:.3f}"
                    )
                    self.update_best(prob, params)

                    if prob >= self.target_prob:
                        print("\n>>> ZIEL ERREICHT! <<<")
                        print(f"P(Drone) = {prob:.3f}")
                        print("Optimierte Parameter:")
                        for k, v in params.items():
                            print(f"  {k}: {v}")
                        return params, prob

        print("\n>>> ENDE: BESTES ERGEBNIS <<<")
        print(f"P(Drone) = {self.best_prob:.3f}")
        print("Beste Parameter:")
        for k, v in self.best_params.items():
            print(f"  {k}: {v}")

        return self.best_params, self.best_prob


def main():
    ROOT = os.path.dirname(os.path.dirname(__file__))

    model_path = os.path.join(ROOT, "models/hbd/uav_band_model.joblib")
    config_path = os.path.join(ROOT, "app/train_hbo/config.yaml")

    optimizer = ParrotHarmonicOptimizer(model_path, config_path, target_prob=0.95)
    optimizer.optimize()


if __name__ == "__main__":
    main()
