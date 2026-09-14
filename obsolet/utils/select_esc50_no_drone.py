import csv
import shutil
from pathlib import Path

ESC50_META = Path("data/ESC-50/meta/esc50.csv")
ESC50_AUDIO = Path("data/ESC-50/audio")
OUT_DIR = Path("data/train/no_drone")

# Relevante NO-DRONE Klassen
NO_DRONE_CLASSES = {
    "dog",
    "rain",
    "wind",
    "water",
    "engine",
    "helicopter",
    "chainsaw",
    "siren",
    "car_horn",
    "traffic",
    "fireworks",
    "insects",
    "breathing",
    "coughing",
    "laughing",
    "fan"
}

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(ESC50_META, "r") as f:
        reader = csv.DictReader(f)

        count = 0
        for row in reader:
            label = row["category"]
            filename = row["filename"]

            if label in NO_DRONE_CLASSES:
                src = ESC50_AUDIO / filename
                dst = OUT_DIR / filename

                if src.exists():
                    shutil.copy(src, dst)
                    count += 1

    print(f"[OK] {count} NO-DRONE Dateien nach {OUT_DIR} kopiert.")

if __name__ == "__main__":
    main()
