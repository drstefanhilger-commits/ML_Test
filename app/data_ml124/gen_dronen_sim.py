"""
Dronen/sim: synthetische, **saubere** Mehrrotor-Drohnensignale (ohne Umgebungsgeräusch) für
das Training von Modul 124 (docs/Trainingskonzept_ML124.md, Abschnitt 5). Das Störgeräusch wird
erst beim Mischen hinzugefügt; so lässt sich der Band-SNR der UAV-Komponente exakt berechnen.

Modell je Clip (alle Zufallswerte aus einem festen Seed -> reproduzierbar):
  - 4/6/8 Rotoren mit 2/3 Blättern; Blattfrequenz (BPF) 80–350 Hz, log-gleichverteilt
  - Drehzahl je Rotor: statischer Versatz (σ 1,5 %) + Regelschwankung (Ornstein-Uhlenbeck, σ 0,8 %)
  - Manöver: gemeinsame Drehzahländerung (bis ±8 %), Doppler durch Radialgeschwindigkeit (bis ±15 m/s)
  - Tonale Anteile je Rotor: BPF-Harmonische bis 8 kHz (Abfall k^-α, α 0,3–1,2, zufällige Hüllkurve
    ±3 dB), Wellen-Harmonische durch Unwucht (−30…−12 dB), Motor-Pfeifen (elektrische Harmonische
    bei Polpaarzahl 6–7 × Drehzahl, −25…−10 dB), AM einmal je Umdrehung
  - Breitbandiges Rotorrauschen (Steigung 1/f^β, β 0,3–1,0; 150–8000 Hz), mit der BPF moduliert
    (−12…+3 dB relativ zu den tonalen Anteilen)
  Abgestimmt auf die Spektralverteilung der echten DDS-Drohnenaufnahmen (docs/Daten_ML124.md).
  - Ausbreitung: Luftdämpfung ISO 9613-1 (20 °C, 70 % rF, Näherung) über die Distanz, Bodenreflexion
    (Zweistrahlmodell, Quellhöhe 10–120 m, Mikrofonhöhe 1,5 m, Reflexionsfaktor 0,5–0,9)
  - Ausgabe auf RMS −26 dBFS normiert (die Distanz wirkt nur spektral; der Pegel wird beim Mischen gesetzt)

⚠ PRÜFEN: Die Verwendung dieser Daten im Training ist noch nicht freigegeben und muss vorher
überprüft werden (Prüfpunkte: docs/Daten_ML124.md, Hinweis am Anfang).

Ausgabe: data/48kHz/{train,test}/Dronen/sim/drone_sim_XXXX.wav (48 kHz, mono, PCM_24) und .json
(Parameter, Drehzahlverläufe je Rotor mit 50 Hz); Fold = Index % 5 + 1, Test = Fold 5;
Liste data/48kHz/meta_dronen_sim.csv.

Aufruf (im Repo-Wurzelverzeichnis):
  python app/data_ml124/gen_dronen_sim.py [--n 300] [--seed 20260926] [--dur 10] [--jobs 8]
"""
import argparse, csv, json, os, sys
from multiprocessing import Pool
import numpy as np
from scipy.signal import butter, sosfilt
import soundfile as sf

sys.path.insert(0, os.path.dirname(__file__))
from common import DATA48, SR, split_of, write_meta

C = 343.0                     # Schallgeschwindigkeit m/s
F_MAX = 8000.0                # höchste erzeugte Teiltonfrequenz
RMS_DBFS = -26.0
H_MIC = 1.5
# ISO 9613-1, 20 °C, 70 % rel. Feuchte, Näherung: Frequenz [Hz] -> Dämpfung [dB/m]
AIR_F = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000], float)
AIR_DB_PER_M = np.array([0.0001, 0.0003, 0.0011, 0.0028, 0.0050, 0.0090, 0.0229, 0.0766, 0.26])


def ou_process(rng, n, dt, sigma, tau):
    """Ornstein-Uhlenbeck-Prozess (stationäre Streuung sigma, Zeitkonstante tau)."""
    x = np.empty(n); x[0] = rng.normal(0, sigma)
    a = np.exp(-dt / tau); b = sigma * np.sqrt(1 - a * a)
    w = rng.normal(0, 1, n)
    for i in range(1, n):
        x[i] = a * x[i - 1] + b * w[i]
    return x


def draw_params(rng):
    p = {}
    p["n_rotors"] = int(rng.choice([4, 6, 8], p=[0.6, 0.25, 0.15]))
    p["n_blades"] = int(rng.choice([2, 3], p=[0.7, 0.3]))
    p["f_bpf0_hz"] = float(np.exp(rng.uniform(np.log(80), np.log(350))))
    p["rotor_offset"] = [float(np.clip(rng.normal(0, 0.015), -0.04, 0.04)) for _ in range(p["n_rotors"])]
    p["rotor_weight"] = [float(rng.uniform(0.7, 1.0)) for _ in range(p["n_rotors"])]
    p["ou_sigma"] = float(rng.uniform(0.003, 0.012)); p["ou_tau_s"] = float(rng.uniform(0.3, 2.0))
    p["maneuver_amp"] = float(rng.uniform(0.0, 0.08)); p["maneuver_hz"] = float(rng.uniform(0.05, 0.4))
    p["v_radial0"] = float(rng.uniform(-10, 10)); p["v_radial_amp"] = float(rng.uniform(0, 5))
    p["harm_alpha"] = float(rng.uniform(0.3, 1.2))
    p["shaft_level_db"] = float(rng.uniform(-30, -12))
    p["am_depth"] = float(rng.uniform(0.05, 0.3))
    p["broadband_db"] = float(rng.uniform(-12, 3))
    p["broadband_slope"] = float(rng.uniform(0.3, 1.0))
    p["pole_pairs"] = int(rng.choice([6, 7]))
    p["motor_level_db"] = float(rng.uniform(-25, -10))
    p["distance_m"] = float(np.exp(rng.uniform(np.log(15), np.log(300))))
    p["height_m"] = float(min(rng.uniform(10, 120), 0.95 * p["distance_m"]))
    p["refl_coeff"] = float(rng.uniform(0.5, 0.9))
    return p


def synth(idx, seed, dur):
    rng = np.random.default_rng(seed)
    p = draw_params(rng)
    n = int(dur * SR); t = np.arange(n) / SR
    # Steuergrößen mit 1 kHz, dann auf Abtastrate interpoliert
    nc = int(dur * 1000) + 1; tc = np.arange(nc) / 1000.0
    maneuver = 1 + p["maneuver_amp"] * np.sin(2 * np.pi * p["maneuver_hz"] * tc + rng.uniform(0, 2 * np.pi))
    v_r = p["v_radial0"] + p["v_radial_amp"] * np.sin(2 * np.pi * rng.uniform(0.02, 0.2) * tc + rng.uniform(0, 2 * np.pi))
    doppler = C / (C - v_r)                                   # v_r > 0: Quelle nähert sich
    rps0 = p["f_bpf0_hz"] / p["n_blades"]
    nb = p["n_blades"]

    tonal = np.zeros(n); tracks = []
    env_db = rng.uniform(-3, 3, 400)                           # zufällige spektrale Hüllkurve je Harmonische
    for i in range(p["n_rotors"]):
        ou = ou_process(rng, nc, 1e-3, p["ou_sigma"], p["ou_tau_s"])
        rps_c = rps0 * (1 + p["rotor_offset"][i]) * np.exp(ou) * maneuver * doppler
        tracks.append(rps_c[::20])                             # 50 Hz für die Metadaten
        rps = np.interp(t, tc, rps_c)
        phase = 2 * np.pi * np.cumsum(rps) / SR + rng.uniform(0, 2 * np.pi)
        am = 1 + p["am_depth"] * np.sin(phase + rng.uniform(0, 2 * np.pi))
        f_hi = rps.max() * nb
        rotor = np.zeros(n)
        k = 1
        while k * f_hi <= F_MAX:                               # BPF-Harmonische
            a = k ** -p["harm_alpha"] * 10 ** (env_db[k] / 20)
            rotor += a * np.sin(k * nb * phase + rng.uniform(0, 2 * np.pi))
            k += 1
        m = 1                                                  # Wellen-Harmonische (Unwucht), keine BPF-Vielfachen
        while m * rps.max() <= 1500:
            if m % nb:
                rotor += 10 ** (p["shaft_level_db"] / 20) * m ** -1.5 * np.sin(m * phase + rng.uniform(0, 2 * np.pi))
            m += 1
        e = 1                                                  # Motor-Pfeifen: elektrische Harmonische
        while e * p["pole_pairs"] * rps.max() <= F_MAX:
            rotor += 10 ** (p["motor_level_db"] / 20) * e ** -1.0 * np.sin(e * p["pole_pairs"] * phase + rng.uniform(0, 2 * np.pi))
            e += 1
        tonal += p["rotor_weight"][i] * am * rotor
        if i == 0:
            bpf_phase = nb * phase
    # Breitbandiges Rotorrauschen (Leistung ∝ 1/f^β), BPF-moduliert
    w = rng.normal(0, 1, n)
    spec = np.fft.rfft(w); f = np.fft.rfftfreq(n, 1 / SR); spec[1:] /= f[1:] ** (p["broadband_slope"] / 2); spec[0] = 0
    pink = np.fft.irfft(spec, n)
    pink = sosfilt(butter(4, [150, 8000], btype="bandpass", fs=SR, output="sos"), pink)
    pink *= (1 + 0.5 * np.sin(bpf_phase))
    pink *= np.sqrt(np.mean(tonal ** 2) / (np.mean(pink ** 2) + 1e-20)) * 10 ** (p["broadband_db"] / 20)
    x = tonal + pink
    # Ausbreitung: Luftdämpfung + Bodenreflexion im Frequenzbereich (mit Nachlauf gegen Zirkularität)
    r = p["distance_m"]; hs = p["height_m"]
    dh = np.sqrt(max(r ** 2 - (hs - H_MIC) ** 2, 1.0)); r2 = np.sqrt(dh ** 2 + (hs + H_MIC) ** 2)
    tau = (r2 - r) / C
    pad = int(0.1 * SR)
    X = np.fft.rfft(np.concatenate([x, np.zeros(pad)])); fq = np.fft.rfftfreq(n + pad, 1 / SR)
    air = 10 ** (-np.interp(fq, AIR_F, AIR_DB_PER_M) * r / 20)
    air_r = 10 ** (-np.interp(fq, AIR_F, AIR_DB_PER_M) * r2 / 20)
    H = air + p["refl_coeff"] * (r / r2) * air_r * np.exp(-2j * np.pi * fq * tau)
    y = np.fft.irfft(X * H, n + pad)[:n]
    y *= 10 ** (RMS_DBFS / 20) / np.sqrt(np.mean(y ** 2))
    peak = float(np.max(np.abs(y)))
    if peak > 0.99:
        y *= 0.99 / peak

    name = f"drone_sim_{idx:04d}"
    fold = idx % 5 + 1; sp = split_of(fold)
    rel = os.path.join(sp, "Dronen", "sim", name)
    os.makedirs(os.path.dirname(os.path.join(OUT_ROOT, rel)), exist_ok=True)
    sf.write(os.path.join(OUT_ROOT, rel + ".wav"), y, SR, subtype="PCM_24")
    meta = dict(file=rel + ".wav", seed=int(seed), duration_s=dur, rms_dbfs=RMS_DBFS,
                peak_dbfs=round(20 * np.log10(np.max(np.abs(y))), 2), refl_delay_ms=round(tau * 1e3, 3),
                params=p, track_rate_hz=50,
                rotor_rps=[[round(float(v), 3) for v in tr] for tr in tracks])
    with open(os.path.join(OUT_ROOT, rel + ".json"), "w") as fh:
        json.dump(meta, fh)
    return dict(file=meta["file"], split=sp, group="Dronen", subset="sim", source="synthetisch", category="drone_sim",
                fold=fold, duration_s=dur, seed=seed, n_rotors=p["n_rotors"], n_blades=p["n_blades"],
                f_bpf0_hz=round(p["f_bpf0_hz"], 2), distance_m=round(p["distance_m"], 1),
                height_m=round(p["height_m"], 1), broadband_db=round(p["broadband_db"], 1),
                peak_dbfs=meta["peak_dbfs"], license="eigene Erzeugung")


OUT_ROOT = DATA48


def _job(a):
    return synth(*a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument("--dur", type=float, default=10.0)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    with Pool(a.jobs) as pool:
        rows = pool.map(_job, [(i, a.seed + i, a.dur) for i in range(a.n)])
    write_meta(os.path.join(OUT_ROOT, "meta_dronen_sim.csv"), rows)
    print(f"{len(rows)} Clips nach {OUT_ROOT}/{{train,test}}/Dronen/sim")


if __name__ == "__main__":
    main()
