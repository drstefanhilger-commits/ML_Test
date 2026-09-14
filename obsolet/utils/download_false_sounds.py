import os
import requests
from pathlib import Path

OUT_DIR = Path("data/false_sounds")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 100% funktionierende WAV-Quellen (CC0 / Public Domain)
SOURCES = {
    "wind.wav": "https://opengameart.org/sites/default/files/Wind%20Loop.wav",
    "rain.wav": "https://opengameart.org/sites/default/files/Rain_0.wav",
    "traffic.wav": "https://opengameart.org/sites/default/files/Traffic%20Ambience.wav",
    "human_voice.wav": "https://opengameart.org/sites/default/files/Man%20Talking.wav",
    "dog.wav": "https://opengameart.org/sites/default/files/Dog%20Bark.wav",
    "fan_noise.wav": "https://opengameart.org/sites/default/files/Fan%20Noise.wav",
    "helicopter.wav": "https://opengameart.org/sites/default/files/Helicopter.wav"
}

def download_file(name, url):
    print(f"Downloading {name}...")
    try:
        r = requests.get(url, stream=True, timeout=20)
        r.raise_for_status()
    except Exception as e:
        print(f"Failed: {name} ({e})")
        return

    out_path = OUT_DIR / name
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=4096):
            if chunk:
                f.write(chunk)

    print(f"Saved: {out_path}")

def main():
    print("Downloading False-Test Sounds...\n")
    for name, url in SOURCES.items():
        download_file(name, url)

    print("\nDone. Files stored in:", OUT_DIR)

if __name__ == "__main__":
    main()
