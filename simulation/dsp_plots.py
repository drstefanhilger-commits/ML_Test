import matplotlib.pyplot as plt


def plot_fft_real(freqs_real, mag_real, label, final_plot=False):
    plt.figure(figsize=(12,5))
    plt.plot(freqs_real, mag_real, label=label, color="blue")
    plt.title("FFT – Real Drone")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.xlim(0, 1000)
    plt.grid(True)
    plt.legend()
    plt.show(block=final_plot)


def plot_fft_sim(freqs_sim, mag_sim, label, final_plot=False):
    plt.figure(figsize=(12,5))
    plt.plot(freqs_sim, mag_sim, label=label, color="orange")
    plt.title("FFT – Simulated Drone")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.xlim(0, 1000)
    plt.grid(True)
    plt.legend()
    plt.show(block=final_plot)


def plot_fft_compare(freqs_real, mag_real, freqs_sim, mag_sim, final_plot=False):
    plt.figure(figsize=(12,6))
    plt.plot(freqs_real, mag_real, label="Real", color="blue")
    plt.plot(freqs_sim,  mag_sim,  label="Simulated", color="orange", alpha=0.7)
    plt.title("FFT Comparison – Real vs Simulated")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.xlim(0, 1000)
    plt.grid(True)
    plt.legend()
    plt.show(block=final_plot)


def plot_freq_domain_synth(freqs, S_harm, S_tri, S_total, final_plot=False):
    plt.figure(figsize=(12,6))

    if S_harm is not None:
        plt.plot(freqs, S_harm, label="Harmonics", color="green")

    if S_tri is not None:
        plt.plot(freqs, S_tri, label="Triangle", color="red", alpha=0.7)

    plt.plot(freqs, S_total, label="Combined / Single Spectrum", color="black")

    plt.title("Frequency-Domain Synthetic Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.xlim(0, 1000)
    plt.grid(True)
    plt.legend()

    plt.show(block=final_plot)


def plot_harmonics_on_spectrum(freqs, mag, harmonics, title="Spectrum with Harmonics", final_plot=False):
    """
    Plottet das Spektrum und zeichnet alle gefundenen Harmonischen ein.
    harmonics = Liste von Dicts: {"k": int, "freq": float, "mag": float}
    """

    plt.figure(figsize=(12,6))

    # Spektrum
    plt.plot(freqs, mag, label="Spectrum", color="blue")

    # Harmonische einzeichnen
    for h in harmonics:
        k = h["k"]
        f = h["freq"]
        m = h["mag"]

        # Vertikale Linie
        plt.axvline(f, color="red", linestyle="--", alpha=0.6)

        # Marker
        plt.plot(f, m, "ro")

        # Beschriftung
        plt.text(f, m, f"k={k}\n{f:.2f} Hz", color="red", fontsize=9,
                 ha="center", va="bottom")

    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.xlim(0, 1000)
    plt.grid(True)
    plt.legend()
    plt.show(block=final_plot)
