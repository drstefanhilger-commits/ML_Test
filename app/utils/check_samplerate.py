import soundfile as sf
import os

base = "./data/train_48k/drone_real_train"

rates = {}

for root, _, files in os.walk(base):
    for f in files:
        if f.lower().endswith(".wav"):
            path = os.path.join(root, f)
            try:
                info = sf.info(path)
                sr = info.samplerate
                rates[sr] = rates.get(sr, 0) + 1
            except Exception as e:
                print(f"Fehler bei Datei: {path} -> {e}")

print("\nSample-Rate Übersicht:")
for sr, count in rates.items():
    print(f"{sr} Hz : {count} Dateien")
