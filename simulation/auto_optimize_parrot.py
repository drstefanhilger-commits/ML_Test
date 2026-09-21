import numpy as np
import joblib
import yaml
import os
from app.train_hbo.dataset import load_dataset_single
from simulation.drone_synth_parrot import synthesize_parrot_drone


class ParrotAutoOptimizer:
    """
    Zweistufiger Auto-Optimizer:
    1) Grobe Optimierung (steigende Parameter)
    2) Feintuning (Hill-Climbing + Random Variation)
    Speichert in jedem Schritt das beste Ergebnis.
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

        # Startparameter (realistisch)
        self.params = {
            "f0": 118.0,
            "n_harmonics": 12,
            "base_mag": 1.2,
            "decay_mag": 1.2,
            "roughness": 1.0,
            "am_depth": 0.30,
            "fm_depth": 0.07,
            "noise_level": 0.05,
            "jitter_amount": 0.03,
            "drift_amount": 0.02,
            "wind_noise": 0.04,
        }

        # Physikalische Grenzen
        self.bounds = {
            "f0": (105, 135),
            "n_harmonics": (8, 18),  # integer!
            "base_mag": (0.8, 2.5),
            "decay_mag": (0.8, 2.0),
            "roughness": (0.4, 1.8),
            "am_depth": (0.1, 0.6),
            "fm_depth": (0.02, 0.15),
            "noise_level": (0.01, 0.15),
            "jitter_amount": (0.005, 0.08),
            "drift_amount": (0.005, 0.05),
            "wind_noise": (0.01, 0.08),
        }

        # Bestes Ergebnis
        self.best_prob = 0.0
        self.best_params = self.params.copy()

    def clamp(self, key):
        lo, hi = self.bounds[key]
        if key == "n_harmonics":
            self.params[key] = int(max(lo, min(hi, self.params[key])))
        else:
            self.params[key] = max(lo, min(hi, self.params[key]))

    def evaluate(self, params):
        audio = synthesize_parrot_drone(sr=self.sr, **params)
        X = load_dataset_single(audio, self.sr, self.n_fft, self.hop, self.n_harm)
        return self.clf.predict_proba([X])[0][1]

    def update_best(self, prob):
        if prob > self.best_prob:
            self.best_prob = prob
            self.best_params = self.params.copy()
            print(f"*** Neues Bestes Ergebnis: {prob:.3f} ***")

    def optimize(self, max_steps=60):
        print("\n=== AUTO-OPTIMIZER START ===")
        print(f"Ziel: P(Drone) > {self.target_prob}")

        # -----------------------------
        # PHASE 1: GROBE OPTIMIERUNG
        # -----------------------------
        print("\n--- PHASE 1: GROBE OPTIMIERUNG ---")

        for step in range(20):
            prob = self.evaluate(self.params)
            print(f"Schritt {step+1}/20 → P={prob:.3f}")

            self.update_best(prob)

            if prob >= self.target_prob:
                print("\n>>> ZIEL ERREICHT! <<<")
                return self.params, prob

            # leichte Verstärkung
            self.params["fm_depth"] += 0.01
            self.params["am_depth"] += 0.01
            self.params["roughness"] += 0.03
            self.params["noise_level"] += 0.01
            self.params["jitter_amount"] += 0.003
            self.params["drift_amount"] += 0.003
            self.params["base_mag"] += 0.05

            for key in self.params:
                self.clamp(key)

        # -----------------------------
        # PHASE 2: FEINTUNING
        # -----------------------------
        print("\n--- PHASE 2: FEINTUNING (Hill-Climbing) ---")

        self.params = self.best_params.copy()

        for step in range(40):
            prob = self.evaluate(self.params)
            print(f"Feintuning {step+1}/40 → P={prob:.3f}")

            self.update_best(prob)

            if prob >= self.target_prob:
                print("\n>>> ZIEL ERREICHT! <<<")
                return self.params, prob

            # kleine zufällige Variation
            candidate = self.params.copy()
            for key in candidate:
                if key == "n_harmonics":
                    # integer Variation
                    candidate[key] += np.random.choice([-1, 0, 1])
                else:
                    candidate[key] += np.random.uniform(-0.02, 0.02)

                lo, hi = self.bounds[key]
                if key == "n_harmonics":
                    candidate[key] = int(max(lo, min(hi, candidate[key])))
                else:
                    candidate[key] = max(lo, min(hi, candidate[key]))

            new_prob = self.evaluate(candidate)

            if new_prob > prob:
                print(f"Verbesserung gefunden: {new_prob:.3f}")
                self.params = candidate.copy()
                self.update_best(new_prob)
            else:
                self.params = self.best_params.copy()

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

    optimizer = ParrotAutoOptimizer(model_path, config_path, target_prob=0.95)
    optimizer.optimize()


if __name__ == "__main__":
    main()
