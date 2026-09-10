import json
from pathlib import Path

MEL_DIR = Path("data/mel")
SELECTED_DIR = Path("data/mel/selected")
OLD_SPLIT = Path("data/splits/train.json")
NEW_SPLIT = Path("data/splits/extended_train.json")

def infer_label(filename):
    name = filename.lower()
    if "bebop" in name:
        return 1
    if "membo" in name:
        return 2
    return 0  # unknown

def main():
    # alte Trainingsdaten laden
    with open(OLD_SPLIT) as f:
        old_items = json.load(f)

    new_items = []

    for f in sorted(SELECTED_DIR.glob("*.npy")):
        label = infer_label(f.name)
        new_items.append({
            "mel": f"selected/{f.name}",
            "label": label
        })

    merged = old_items + new_items

    with open(NEW_SPLIT, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"Extended split saved: {NEW_SPLIT}")
    print(f"Old items: {len(old_items)}")
    print(f"New items: {len(new_items)}")
    print(f"Total: {len(merged)}")

if __name__ == "__main__":
    main()
