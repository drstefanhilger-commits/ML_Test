# app/train/hard_boost.py

import json
from pathlib import Path

CONFIG_PATH = Path("app/train/hard_sample_boost.json")

def load_boost_config():
    if not CONFIG_PATH.exists():
        print("[WARN] Hard-Boost-Konfiguration fehlt:", CONFIG_PATH)
        return None

    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def apply_hard_boost(X, Y):
    cfg = load_boost_config()
    if cfg is None:
        return X, Y

    boost_X = []
    boost_Y = []

    # FN-Samples (Label 1)
    for name in cfg["fn_samples"]:
        path = Path("data/train/drone_hard") / name
        if path.exists():
            for _ in range(cfg["boost_factor_fn"]):
                boost_X.append(path)
                boost_Y.append(1)

    # Borderline-Samples (Label 1)
    for name in cfg["borderline_samples"]:
        path = Path("data/train/drone_hard") / name
        if path.exists():
            for _ in range(cfg["boost_factor_borderline"]):
                boost_X.append(path)
                boost_Y.append(1)

    # False-Sound-Samples (Label 0)
    for name in cfg["false_sound_samples"]:
        path = Path("app/train/no_drone") / name
        if path.exists():
            for _ in range(cfg["boost_factor_false_sound"]):
                boost_X.append(path)
                boost_Y.append(0)

    print(f"[BOOST] FN: {len(cfg['fn_samples'])} × {cfg['boost_factor_fn']}")
    print(f"[BOOST] Borderline: {len(cfg['borderline_samples'])} × {cfg['boost_factor_borderline']}")
    print(f"[BOOST] False-Sounds: {len(cfg['false_sound_samples'])} × {cfg['boost_factor_false_sound']}")

    return boost_X, boost_Y
