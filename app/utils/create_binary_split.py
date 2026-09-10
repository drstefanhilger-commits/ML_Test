import json
from pathlib import Path

MEL_DIR = Path("data/mel")
SELECTED_DIR = Path("data/mel/selected")
OUT = Path("data/splits/train_binary.json")

def infer_label(filename):
    name = filename.lower()

    # neue Daten
    if "bebop" in name or "membo" in name:
        return 1
    if "unknown" in name:
        return 0

    # alte Daten
    if "drone" in name:
        return 1
    if "human" in name:
        return 0
    if "background" in name:
        return 0
    if "wind" in name:
        return 0

    return 0

def main():
    items = []

    # alte Daten
    for f in sorted(MEL_DIR.glob("*.npy")):
        label = infer_label(f.name)
        items.append({
            "mel": f.name,
            "label": label
        })

    # neue Daten
    for f in sorted(SELECTED_DIR.glob("*.npy")):
        label = infer_label(f.name)
        items.append({
            "mel": f"selected/{f.name}",
            "label": label
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT, "w") as fp:
        json.dump(items, fp, indent=2)

    print(f"Binary split saved: {OUT}")
    print(f"Total items: {len(items)}")

if __name__ == "__main__":
    main()
