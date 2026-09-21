import os
import random
import numpy as np
import soundfile as sf

from dsp_core import (
    load_wav, spectrum_to_wav,
    compute_fft, detect_f0, detect_f0_sim,
    simulate_harmonic_drone, get_mag_at_f0,
    synth_freq_domain_harmonics_and_triangle,
    synth_freq_domain_decay_1_over_x,
    synth_freq_domain_harmonic,
    synth_freq_domain_harmonic_decay,
    bandpass_filter,
    add_roughness_to_decay,
    find_harmonics_in_timesignal
)

from dsp_plots import (
    plot_fft_real, plot_fft_sim, plot_fft_compare,
    plot_freq_domain_synth,
    plot_harmonics_on_spectrum
)

# ============================================================
# DATEN LADEN
# ============================================================
real_folder = "./data/train_48k/drone_real_train"
real_files = sorted([f for f in os.listdir(real_folder) if f.endswith(".wav")])

print(f"[INFO] Anzahl realer Dateien: {len(real_files)}")
REAL_INDEX = random.randint(0, len(real_files) - 1)
real_path = os.path.join(real_folder, real_files[REAL_INDEX])
print("[INFO] Real:", real_path)

real_audio, sr_real = load_wav(real_path)

# ============================================================
# Bandpass direkt nach dem Laden anwenden
# ============================================================
real_audio = bandpass_filter(real_audio, sr_real, low=40, high=800)

# ============================================================
# FFT REAL
# ============================================================
freqs_real, mag_real = compute_fft(real_audio, sr_real)

# ============================================================
# Finde Harmonische im Zeitsignal
# ============================================================
harmonics_real = find_harmonics_in_timesignal(real_audio, sr_real, max_harmonics=10)
f0_real = harmonics_real[0]["freq"]
print(f"[INFO] Detected f0 = {f0_real:.2f} Hz")

# ============================================================
# Frequenzachse für Simulation (fein aufgelöst)
# ============================================================
freqs_sim = np.linspace(0, 300, 5000)

# ============================================================
# Harmonische erzeugen (synthetisch)
# ============================================================
mag_sim_harm, mag_harm = synth_freq_domain_harmonic_decay(
    f0_real,
    freqs_sim,
    K=10,
    decay="1/k2",
    base_mag=1.0
)

# ============================================================
# 1/r-Abfall erzeugen
# ============================================================
D_raw = synth_freq_domain_decay_1_over_x(f0_real, freqs_sim, mag=1.0)

# reale Referenz-Magnitude (erste Harmonische)
ref_mag = harmonics_real[0]["mag"]

# 30% der realen Magnitude
D_scaled = 0.30 * ref_mag * D_raw

# Rauhigkeit hinzufügen
D_rough = add_roughness_to_decay(D_scaled, amount=0.5, filtered=True)

# ============================================================
# Harmonische erzeugen
# ============================================================
mag_sim_harm, mag_harm = synth_freq_domain_harmonic_decay(
    f0_real,
    freqs_sim,
    K=10,
    decay="1/k2",
    base_mag=1.0
)

# Harmonische skalieren
harm_scale = ref_mag
mag_harm *= harm_scale

# ============================================================
# Gesamtspektrum
# ============================================================
mag_sim = mag_harm + D_rough
mag_sim *= 0.4
mag_harm *= 0.4


# ============================================================
# Spektrum zu WAV
# ============================================================
audio_sim = spectrum_to_wav(freqs_sim, mag_sim, sr=48000, duration=1.0)
sf.write("simulated_drone.wav", audio_sim, 48000)


# ============================================================
# PLOTS
# ============================================================

# Real Spectrum
plot_harmonics_on_spectrum(
    freqs_real, mag_real, harmonics_real,
    title="Real Spectrum with Harmonics",
    final_plot=False
)

# Simulated Spectrum
plot_freq_domain_synth(
    freqs_sim, mag_harm, None, mag_sim,
    final_plot=False
)

# Vergleich Real vs Sim
plot_fft_compare(
    freqs_real, mag_real,
    freqs_sim, mag_sim,
    final_plot=True
)
