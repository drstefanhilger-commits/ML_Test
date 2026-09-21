import numpy as np
import soundfile as sf
import scipy.signal as sig



# ============================================================
# Rauhigkeit zu einem 1/r-Abfall hinzufügen
# ============================================================
def add_roughness_to_decay(D, amount=0.1, filtered=True):
    """
    Fügt dem 1/r-Abfall Rauhigkeit hinzu.
    - D: Originalspektrum (z.B. 1/r-Abfall)
    - amount: Stärke der Rauhigkeit (0.05 bis 0.2 sinnvoll)
    - filtered: True = realistisch, False = weißes Noise

    Rückgabe:
    - D_rough: Spektrum mit Rauhigkeit
    """

    N = len(D)

    if filtered:
        # Gefiltertes Noise (realistischer)
        noise = np.random.randn(N)
        sos = sig.butter(2, 0.5, output='sos')
        noise = sig.sosfilt(sos, noise)
    else:
        # Weißes Noise
        noise = np.random.randn(N)

    roughness = 1.0 + amount * noise
    return D * roughness

# ============================================================
# Harmonisches Spektrum mit Abfall erzeugen
# ============================================================
def synth_freq_domain_harmonic_decay(f0, freqs, K=10, decay="1/k", base_mag=1.0):
    """
    Erzeugt ein harmonisches Spektrum mit künstlichem Abfall.
    - f0: Grundfrequenz
    - freqs: Frequenzachse
    - K: Anzahl Harmonischer
    - decay: '1/k', '1/k2', 'exp'
    - base_mag: Grundamplitude

    Rückgabe:
    - S_total: komplettes Spektrum
    - S_harm: nur Harmonische
    """

    S = np.zeros_like(freqs)

    for k in range(1, K + 1):
        fk = k * f0

        # Magnitude-Modell
        if decay == "1/k":
            Ak = base_mag / k
        elif decay == "1/k2":
            Ak = base_mag / (k * k)
        elif decay == "exp":
            Ak = base_mag * np.exp(-0.5 * k)
        else:
            Ak = base_mag / k  # fallback

        # Peak setzen
        idx = np.argmin(np.abs(freqs - fk))
        S[idx] += Ak

    return S, S.copy()

# ============================================================
# WAV laden
# ============================================================
def synth_freq_domain_decay_1_over_x(f0, freqs, mag=0.5, limit=5.0):
    D = np.zeros_like(freqs)
    f1 = f0

    for i, f in enumerate(freqs):
        if f > f1:
            val = mag / (f - f1)
            D[i] = min(val, limit)   # Begrenzung
    return D

# ============================================================
# Synthese im Frequenzbereich
# ============================================================
def synth_freq_domain_harmonic(f0, freqs, K=10, mag_model=None):
    """
    Erzeugt ein reines harmonisches Frequenzspektrum:
    - f0: Grundfrequenz
    - freqs: Frequenzachse (z.B. FFT-Frequenzen)
    - K: Anzahl Harmonischer
    - mag_model: Funktion A(k) -> Amplitude, optional

    Rückgabe:
    - S_total: komplettes Spektrum
    - S_harm: nur Harmonische
    """

    S = np.zeros_like(freqs)

    # Default-Amplitudenmodell: 1/k
    if mag_model is None:
        mag_model = lambda k: 1.0 / k

    for k in range(1, K + 1):
        fk = k * f0
        idx = np.argmin(np.abs(freqs - fk))
        S[idx] += mag_model(k)

    return S, S.copy()

# ============================================================
# WAV laden
# ============================================================
def load_wav(path):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio[:,0]
    return audio, sr

# ============================================================
# Spektrum zu WAV
# ============================================================
def spectrum_to_wav(freqs, mag, sr=48000, duration=1.0, phase_type="random"):
    """
    Erzeugt ein Zeitbereichssignal aus einem Magnitude-Spektrum.
    - freqs: Frequenzachse (linear)
    - mag: Magnitude-Spektrum (real)
    - sr: Samplingrate (48 kHz)
    - duration: Länge des Signals in Sekunden
    - phase_type: 'random' oder 'zero'

    Rückgabe:
    - audio: Zeitbereichssignal
    """

    # Anzahl Samples
    N = int(sr * duration)

    # Frequenzachse der FFT
    freqs_fft = np.fft.rfftfreq(N, 1/sr)

    # Magnitude auf FFT-Achse interpolieren
    mag_interp = np.interp(freqs_fft, freqs, mag)

    # Phase erzeugen
    if phase_type == "random":
        phase = np.exp(1j * 2 * np.pi * np.random.rand(len(freqs_fft)))
    else:
        phase = np.ones(len(freqs_fft), dtype=complex)

    # Komplexes Spektrum
    spectrum = mag_interp * phase

    # Inverse FFT
    audio = np.fft.irfft(spectrum, n=N)

    # Normalisieren
    audio /= np.max(np.abs(audio)) + 1e-12

    return audio

# ============================================================
# FFT
# ============================================================
def compute_fft(audio, sr):
    n = len(audio)
    window = np.hanning(n)
    fft = np.fft.rfft(audio * window)
    freqs = np.fft.rfftfreq(n, 1/sr)
    mag = np.abs(fft)
    return freqs, mag

# ============================================================
# f0 aus realer Drohne bestimmen
# ============================================================
def detect_f0(freqs, mag, fmin=40, fmax=200):
    mask = (freqs >= fmin) & (freqs <= fmax)
    idx = np.argmax(mag[mask])
    return freqs[mask][idx]

# ============================================================
# f0 der Simulation bestimmen (Peak-Interpolation)
# ============================================================
def detect_f0_sim(freqs, mag, f0_real):
    fmin = 0.5 * f0_real
    fmax = 1.5 * f0_real
    mask = (freqs >= fmin) & (freqs <= fmax)

    mag_win = mag[mask]
    freqs_win = freqs[mask]

    i = np.argmax(mag_win)

    if 1 <= i < len(mag_win)-1:
        alpha = mag_win[i-1]
        beta  = mag_win[i]
        gamma = mag_win[i+1]
        p = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma)
        f0_est = freqs_win[i] + p * (freqs_win[1] - freqs_win[0])
    else:
        f0_est = freqs_win[i]

    return f0_est

# ============================================================
# Magnitude bei f0 bestimmen
# ============================================================
def get_mag_at_f0(freqs, mag, f0, bw=2.0):
    mask = (freqs >= f0 - bw) & (freqs <= f0 + bw)
    return np.max(mag[mask])

# ============================================================
# Reine Harmonische Simulation
# ============================================================
def simulate_harmonic_drone(f0, sr=48000, duration=2.0):
    N = int(sr * duration)
    t = np.linspace(0, duration, N, endpoint=False)

    K = 6
    s = np.zeros_like(t)

    for k in range(1, K+1):
        Ak = 1.0 / k
        s += Ak * np.sin(2 * np.pi * k * f0 * t)

    return s, sr

# ============================================================
# Frequenzbereich: Harmonische + Dreieck
# ============================================================
def synth_freq_domain_harmonics_and_triangle(f0, freqs, K=6, tri_mag=1.0):
    S = np.zeros_like(freqs)

    # Harmonische
    for k in range(1, K+1):
        fk = k * f0
        idx = np.argmin(np.abs(freqs - fk))
        S[idx] += 1.0 / k

    # Dreieck
    T = np.zeros_like(freqs)
    f1 = f0
    f2 = 1000.0

    for i, f in enumerate(freqs):
        if f1 <= f <= f2:
            T[i] = tri_mag * (1 - (f - f1) / (f2 - f1))

    return S + T, S, T

# ============================================================
# Bandpass-Filter
# ============================================================
def bandpass_filter(audio, sr, low=40, high=2000, order=4):
    """
    Wendet einen stabilen Butterworth-Bandpass im Bereich [40..2000] Hz an.
    - audio: Zeitbereichssignal
    - sr: Samplingrate
    - low/high: Grenzfrequenzen
    - order: Filterordnung
    """
    sos = sig.butter(order, [low/(sr/2), high/(sr/2)], btype='band', output='sos')
    return sig.sosfiltfilt(sos, audio)

# ============================================================
# Harmonische im Zeitsignal finden
# ============================================================
def find_harmonics_in_timesignal(audio, sr, max_harmonics=10, fmin=40, fmax=200):
    freqs, mag = compute_fft(audio, sr)
    f0 = detect_f0(freqs, mag, fmin=fmin, fmax=fmax)

    harmonics = []
    df = freqs[1] - freqs[0]

    for k in range(1, max_harmonics + 1):
        fk = k * f0
        bw = 3 * df
        mask = (freqs >= fk - bw) & (freqs <= fk + bw)

        if not np.any(mask):
            continue

        mag_win = mag[mask]
        freqs_win = freqs[mask]
        idx = np.argmax(mag_win)

        # Peak interpolation
        if 1 <= idx < len(mag_win) - 1:
            alpha = mag_win[idx - 1]
            beta = mag_win[idx]
            gamma = mag_win[idx + 1]
            p = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
            f_est = freqs_win[idx] + p * df
            m_est = beta
        else:
            f_est = freqs_win[idx]
            m_est = mag_win[idx]

        # Erweiterte DSP-Metriken
        width = estimate_spectral_width(freqs, mag, f_est)
        noise = estimate_noise_floor(freqs, mag, f_est)
        snr = estimate_snr(m_est, noise)

        harmonics.append({
            "k": k,
            "freq": f_est,
            "mag": m_est,
            "width": width,
            "noise_floor": noise,
            "snr": snr
        })

    return harmonics

# ============================================================
# Spektrale Breite eines Peaks schätzen
# ============================================================
def estimate_spectral_width(freqs, mag, peak_freq, window_hz=20):
    """
    Bestimmt die spektrale Breite (FWHM) eines Peaks um peak_freq.
    """
    mask = (freqs >= peak_freq - window_hz) & (freqs <= peak_freq + window_hz)
    f = freqs[mask]
    m = mag[mask]

    if len(m) < 3:
        return 0.0

    peak_mag = np.max(m)
    half_mag = peak_mag * 0.5

    # Punkte über der Halbwertsgrenze
    above = m >= half_mag
    if not np.any(above):
        return 0.0

    idx = np.where(above)[0]
    fwhm = f[idx[-1]] - f[idx[0]]
    return float(fwhm)

# ============================================================
# Lokalen Noise-Floor schätzen
# ============================================================
def estimate_noise_floor(freqs, mag, peak_freq, window_hz=50):
    """
    Schätzt den lokalen Noise-Floor um peak_freq.
    """
    mask = (freqs >= peak_freq - window_hz) & (freqs <= peak_freq + window_hz)
    f = freqs[mask]
    m = mag[mask]

    if len(m) < 10:
        return 0.0

    # Peak entfernen
    peak_idx = np.argmax(m)
    m_no_peak = np.delete(m, peak_idx)

    noise_floor = np.median(m_no_peak)
    return float(noise_floor)

# ============================================================
# Signal-zu-Rausch-Verhältnis (SNR) schätzen
# ============================================================
def estimate_snr(peak_mag, noise_floor):
    """
    SNR = Peak / NoiseFloor
    """
    if noise_floor <= 1e-12:
        return float("inf")
    return float(peak_mag / noise_floor)


# ============================================================
# Drohnen-Signal synthetisieren
# ============================================================
def synthesize_drone_signal(
    sr=48000,
    duration=1.0,
    f0=120.0,
    n_harmonics=10,
    base_mag=1.0,
    decay_type="1/k2",
    decay_mag=1.0,
    roughness=0.5,
    am_depth=0.3,
    fm_depth=0.01,
    noise_level=0.02,
):
    """
    Realistische Drohnen-Synthese:
    - Harmonische Rotorlinien
    - 1/r-Abfall + Rauhigkeit
    - AM/FM-Rotor-Modulation
    - Noise-Floor + Turbulenz
    """

    N = int(sr * duration)
    t = np.arange(N) / sr

    # -----------------------------
    # 1. Harmonische im Zeitbereich
    # -----------------------------
    signal = np.zeros_like(t)

    for k in range(1, n_harmonics + 1):
        fk = f0 * k

        # Amplituden- und Frequenzmodulation
        am = 1.0 + am_depth * np.sin(2 * np.pi * f0 * t)
        fm = fk * (1.0 + fm_depth * np.sin(2 * np.pi * 0.5 * t))

        # leichte Instabilität (Jitter)
        jitter = 1.0 + 0.01 * np.random.randn()
        fk_eff = fm * jitter

        # harmonische Amplitude mit 1/k^2-Abfall
        if decay_type == "1/k2":
            ak = base_mag / (k ** 2)
        elif decay_type == "1/k":
            ak = base_mag / k
        else:
            ak = base_mag

        signal += ak * am * np.sin(2 * np.pi * fk_eff * t)

    # -----------------------------
    # 2. Breitbandiger 1/r-Abfall + Rauhigkeit
    # -----------------------------
    # Einfach als breitbandiges Noise mit leichtem Lowpass
    noise = np.random.randn(N)
    # einfacher 1st-order Lowpass
    alpha = 0.1
    for i in range(1, N):
        noise[i] = alpha * noise[i] + (1 - alpha) * noise[i - 1]

    signal += decay_mag * roughness * noise

    # -----------------------------
    # 3. Noise-Floor / Turbulenz
    # -----------------------------
    turbulence = noise_level * np.random.randn(N)
    signal += turbulence

    # -----------------------------
    # 4. Normalisieren
    # -----------------------------
    signal /= np.max(np.abs(signal) + 1e-12)

    return signal
