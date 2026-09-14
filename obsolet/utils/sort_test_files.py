import shutil
from pathlib import Path

TRUE_DIR = Path("test/true_test")
FALSE_DIR = Path("test/false_test")

TRUE_DIR.mkdir(parents=True, exist_ok=True)
FALSE_DIR.mkdir(parents=True, exist_ok=True)

# Drohnen-Erkennungsregeln
DRONE_KEYWORDS = [
    "bebop", "membo", "extra_membo", "mixed_membo",
    "drone", "uav", "quad", "quadrocopter"
]

TEST_DIR = Path("test")

for wav in TEST_DIR.glob("*.wav"):
    name = wav.name.lower()

    if any(k in name for k in DRONE_KEYWORDS):
        shutil.move(str(wav), TRUE_DIR / wav.name)
        print(f"[DRONE] {wav.name} -> test/true_test/")
    else:
        shutil.move(str(wav), FALSE_DIR / wav.name)
        print(f"[NO-DRONE] {wav.name} -> test/false_test/")
