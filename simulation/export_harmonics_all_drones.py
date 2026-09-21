import os
import csv
from dsp_core import (
    load_wav,
    bandpass_filter,
    find_harmonics_in_timesignal
)

real_folder = "./data/train_48k/drone_real_train"
real_files = sorted([f for f in os.listdir(real_folder) if f.endswith(".wav")])

output_file = "harmonics_all_drones.csv"

with open(output_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)

    writer.writerow([
        "filename",
        "k",
        "freq_hz",
        "magnitude",
        "width_hz",
        "noise_floor",
        "snr"
    ])

    for fname in real_files:
        path = os.path.join(real_folder, fname)
        print(f"[INFO] Processing: {fname}")

        audio, sr = load_wav(path)
        audio = bandpass_filter(audio, sr)

        harmonics = find_harmonics_in_timesignal(audio, sr, max_harmonics=10)

        for h in harmonics:
            writer.writerow([
                fname,
                h["k"],
                f"{h['freq']:.4f}",
                f"{h['mag']:.6e}",
                f"{h['width']:.4f}",
                f"{h['noise_floor']:.6e}",
                f"{h['snr']:.4f}"
            ])

print(f"[INFO] Fertig! Daten gespeichert in: {output_file}")
