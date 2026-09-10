import os
import random
import shutil
from pathlib import Path

RAW = Path("data/raw/DroneAudioDataset-master")
OUT = Path("data/selected")
N = 200

def collect_drone_files():
    drone_dirs = [
        RAW / "Binary_Drone_Audio" / "yes_drone",
        RAW / "Multiclass_Drone_Audio" / "bebop_1",
        RAW / "Multiclass_Drone_Audio" / "membo_1",
    ]
    files = []
    for d in drone_dirs:
        for f in d.glob("*.wav"):
            files.append(f)
    return files

def select(files, n):
    random.seed(42)  # deterministisch
    return random.sample(files, n)

def copy(files):
    OUT.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy(f, OUT / f.name)

def main():
    files = collect_drone_files()
    selected = select(files, N)
    copy(selected)
    print(f"Copied {len(selected)} files to {OUT}")

if __name__ == "__main__":
    main()
