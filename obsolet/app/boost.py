# app/train/boost.py

import json
from pathlib import Path

HARD_FN_DIR = Path("data/hard_samples/fn")
HARD_FP_DIR = Path("data/hard_samples/fp")

def load_hard_samples():
    fn = sorted(HARD_FN_DIR.glob("*.wav"))
    fp = sorted(HARD_FP_DIR.glob("*.wav"))
    return fn, fp

def apply_hard_boost(train_set):
    fn, fp = load_hard_samples()

    # FN = echte Drohnen → Label 1
    for wav in fn:
        train_set.append({"file": str(wav), "label": 1})

    # FP = Umweltgeräusche → Label 0
    for wav in fp:
        train_set.append({"file": str(wav), "label": 0})

    return train_set
