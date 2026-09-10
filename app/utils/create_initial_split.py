import json
from pathlib import Path

MEL_DIR = Path("data/mel")
OUT = Path("data/splits/train.json")

def infer_label(filename):
    name = filename.lower()
    if "bebop" in name:
        return 1
    if "membo" in name:
        return 2
    return 0  # unknown

def main():
    items = []

    # alle .npy Dateien außer selected/
    for f in sorted(MEL_DIR.glob("*.npy")):
        label = infer_label(f.name)
        items.append({
            "mel": f.name,
            "label": label
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT, "w") as fp:
        json.dump(items, fp, indent=2)

    print(f"Created initial split: {OUT}")
    print(f"Total items: {len(items)}")

if __name__ == "__main__":
    main()
