import os
import soundfile as sf
from collections import Counter

BASE_DIR = "/mnt/c/Users/310004/Documents/Projects/Drone_Notebook"

wav_dirs = [
    "data/raw/DroneAudioDataset-master/Binary_Drone_Audio/yes_drone",
    "data/raw/DroneAudioDataset-master/Multiclass_Drone_Audio/bebop_1",
    "data/raw/DroneAudioDataset-master/Multiclass_Drone_Audio/membo_1",
    "data/ESC-50/audio",
]

sr_counter = Counter()

for rel_dir in wav_dirs:
    full_dir = os.path.join(BASE_DIR, rel_dir)
    for root, _, files in os.walk(full_dir):
        for f in files:
            if f.lower().endswith(".wav"):
                path = os.path.join(root, f)
                try:
                    info = sf.info(path)
                    sr_counter[info.samplerate] += 1
                except Exception as e:
                    print(f"Fehler bei {path}: {e}")

print("Sample-Rate-Verteilung:")
for sr, count in sr_counter.items():
    print(f"{sr} Hz: {count} Dateien")
