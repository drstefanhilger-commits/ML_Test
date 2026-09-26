# Analyse: BPP-Modell und Transfer nach SDS_110

Stand 26.09.2026, Repository-Stand `f15c676` („Start der SDS Integration“).
Untersucht: `app/train_bpp` (Training), `app/sds/transfere` (Code für das Board),
`models/bpp` (Modell); zum Vergleich `app/train_hbo`, `app/sds_transfere`, `docs/`.
Alle Befunde wurden nachgerechnet (Abschnitt 5 enthält die Prüfskripte).

---

## 1. Kurzfassung

Das BPP-Modell ist ein binärer Klassifikator (Drohne ja/nein) auf 10 Merkmalen je Aufnahme.
Der Transfer-Code in `app/sds/transfere` ist in seinem jetzigen Zustand **nicht einsatzfähig**:

| # | Befund | Schwere |
|---|---|---|
| 1 | Merkmale im C++ haben andere Reihenfolge und Bedeutung als im Training | kritisch |
| 2 | GBM-Export ohne Startwert und Lernrate: log-odds ~20× zu groß | kritisch |
| 3 | `SDS_Bandpass`-Koeffizienten instabil (NaN nach 0,1 s), falsches Band | kritisch |
| 4 | Wichtigstes Merkmal `harmonic_ratio` (88 %) hängt von der Aufnahmedauer ab | hoch |
| 5 | f0-Schätzung liefert 188 Hz bei einem 118-Hz-Signal | hoch |
| 6 | C++-Merkmalsextraktor nur Gerüst (f0 fest 120 Hz, Spektrum 0), Modell nicht angeschlossen | hoch |
| 7 | Merkmale pro Aufnahme (Mittel über die Datei, Spitzenwert-Normierung) vs. Datenstrom auf dem Board | mittel |
| 8 | `app/sds_transfere`: älterer Entwurf mit Fehlern (u. a. Feldüberlauf) | niedrig |

Verhältnis zu SDS_110: Das dort abgelegte `HBD_ML_Model_Data.hpp` (Keras, 171 Merkmale →
64 Band-Wahrscheinlichkeiten) stammt **nicht** aus diesem Repository; das zugehörige
`bpp_training/export_model.py` fehlt hier. Das BPP-Modell liefert nur eine Wahrscheinlichkeit
je Aufnahme und könnte in SDS_110 höchstens die Drohnen-Entscheidung des HBD ersetzen,
nicht den Bandvektor s(t) des Moduls 124.

---

## 2. Aufbau der Pipeline

### Training (`app/train_bpp`)
- `bpp_bandpass.py`: Butterworth-Bandpass 200–3000 Hz, `butter(4, …, btype="bandpass")`
  → Ordnung 8, angewendet mit `filtfilt` (vorwärts und rückwärts, Nullphase).
- `bpp_dataset.py`: WAV (48 kHz) laden → Bandpass → Normierung auf `max(|x|)` der Datei →
  `harmonic_features()` aus `app/train_hbo` mit `n_fft = 2048`, `hop = 512`, 6 Harmonische.
- `bpp_train.py`: `GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
  max_depth=3)`; Daten aus `./data/train_48k/{drone_real,no_drone}_{train,test}`
  (nicht im Repository).
- `bpp_optimizer.py`, `bpp_test.py`: Parametersuche für den Drohnen-Synthesizer, Einzeltest.

### Merkmale (`app/train_hbo/features_harmonic.py`), je **Datei**
`S = |STFT|` (librosa, Hann), f0 über `estimate_f0_v3` (80–200 Hz), dann

| Index | Name | Definition |
|---|---|---|
| 0 | f0 | Grundfrequenz (auf FFT-Bin gerundet, 23,4 Hz Raster) |
| 1 | harmonic_ratio | Σ E_h / Σ S (Nenner: Summe über **alle** Bins und Frames) |
| 2 | harmonic_spread | std(E_1…E_6) |
| 3 | harmonic_stability | mean(diff(E_1…E_6)) = (E_6 − E_1)/5 – eigentlich eine Steigung |
| 4–9 | E_1…E_6 | Mittel über alle Frames von S am nächstgelegenen Bin zu h·f0 |

### Modell (`models/bpp/uav_bandpass_model.joblib`)
`GradientBoostingClassifier`, 10 Merkmale, 200 Bäume, Tiefe 3, Lernrate 0,05,
Startwert (log-odds) −0,407. Gespeichert mit numpy ≥ 2 (lädt nicht mit numpy 1.x).
Merkmalswichtigkeit: harmonic_ratio 0,879 · E5 0,041 · E4 0,025 · spread 0,020 ·
E2 0,017 · E6 0,008 · E1 0,006 · stability 0,002 · E3 0,001 · f0 0,000.

### Transfer (`app/sds/transfere`)
`SDS_Bandpass` (IIR 4. Ordnung, Direktform II transponiert), `SDS_FeatureExtractor`,
`SDS_ModelInference` (sigmoid(evalTrees)), `export_gbm_to_cpp.py` → `generated_gbm.cpp`
(200 Funktionen `tree_i`, `evalTrees`), `SDS_SimDrone` (identisch zu SDS_110).

---

## 3. Befunde im Einzelnen

### 3.1 Merkmale im C++ passen nicht zum Modell (kritisch)
| Index | Training (`harmonic_features`) | C++ (`SDS_ModelInference::toVector`) |
|---|---|---|
| 0 | f0 | f0 |
| 1 | harmonic_ratio | f0_energy (Σ\|X\|² um f0) |
| 2 | harmonic_spread | harm2_energy |
| 3 | harmonic_stability | harm3_energy |
| 4–6 | E1, E2, E3 | harm4…harm6_energy |
| 7 | E4 | spectral_centroid |
| 8 | E5 | spectral_flatness |
| 9 | E6 | broadband_energy |

Beleg aus dem Modell: Die Schwellen von `x[8]` liegen bei 1,3…40 (Energie E5), die spektrale
Flachheit liegt immer in 0…1; `x[1]` wird bei 1,5e-4…1,05e-3 geteilt (Verhältnis, keine Energie).
Auch die Energien sind anders definiert (Training: mittlerer Betrag an einem Bin, C++: Summe der
Betragsquadrate über ±10 Hz).

### 3.2 GBM-Export ohne Startwert und Lernrate (kritisch)
scikit-learn: `log-odds = init + learning_rate · Σ tree_i(x)`, `p = sigmoid(log-odds)`.
`export_gbm_to_cpp.py` exportiert nur `Σ tree_i(x)`; `SDS_ModelInference` wendet darauf direkt
die Sigmoidfunktion an.

Messung (2000 Eingaben im Wertebereich der Split-Schwellen):
- C++ `evalTrees`: −195 … +142 (Mittel −8,2); scikit-learn log-odds: −10,2 … +6,7 (Mittel −0,8).
- sigmoid(evalTrees) vs. `predict_proba`: max. Abweichung 0,53, Klasse verschieden bei 4,9 %.
- Mit `−0,40713317 + 0,05 · evalTrees`: identisch bis auf 1 Eingabe (Abweichung 0,2) – Ursache
  ist die Rundung der Schwellen auf `%.8f` (bei Werten um 1e-4 nur 4–5 gültige Stellen).

Korrektur: im Export `init` (`clf._raw_predict_init`) und `learning_rate` einsetzen, Schwellen
mit `%.9g` (float-genau) schreiben, `evalTrees` an `SDS_ModelInference` anbinden.

### 3.3 `SDS_Bandpass` instabil und falsches Band (kritisch)
Die Koeffizienten in `SDS_Bandpass.cpp` entsprechen keinem Butterworth-Bandpass 200–3000 Hz:
- Pole |z| = 0,49 / 0,99 / **1,035 / 1,035** → instabil; Impulsantwort in float32 nach 0,1 s NaN.
- Frequenzgang (wäre er stabil): −3-dB-Band 3,1–3,6 kHz, +4,4 dB Spitze.

| f [Hz] | Training (filtfilt, Ordnung 8) | C++ SDS_Bandpass |
|---|---|---|
| 100 | −51,8 dB | −41,3 dB |
| 200 | −6,0 dB | −34,6 dB |
| 1000 | 0,0 dB | −19,8 dB |
| 3000 | −6,0 dB | +0,1 dB |
| 4000 | −23,5 dB | −4,1 dB |

`butter(4, bandpass)` hat 9 Koeffizienten (Ordnung 8), nicht 5. Der korrekte Filter divergiert in
float32 als Direktform ebenfalls; stabil ist er nur als Biquad-Kaskade (SOS, z. B. CMSIS
`arm_biquad_cascade_df2T_f32`, wie `Pre_Processor_118` in SDS_110). `filtfilt` ist nicht kausal und
auf dem Board nicht nachbildbar; für Übereinstimmung mit dem Board muss mit kausalem `sosfilt`
trainiert werden.

### 3.4 `harmonic_ratio` hängt von der Aufnahmedauer ab (hoch)
Zähler: Mittel über Frames (6 Werte); Nenner: Summe über **alle** Bins und Frames → ∝ 1/Dauer.
Gemessen mit dem Code dieses Repositories am Signal `synthesize_parrot_drone` (f0 = 118 Hz):

| Dauer | f0 | harmonic_ratio | E1 | E5 |
|---|---|---|---|---|
| 1 s | 188 | 7,73e-4 | 18,3 | 12,6 |
| 2 s | 188 | 3,83e-4 | 14,0 | 8,5 |
| 5 s | 188 | 1,64e-4 | 13,2 | 10,2 |
| 10 s | 188 | 7,87e-5 | 9,9 | 12,3 |

Das Modell trifft 88 % seiner Entscheidung über dieses Merkmal (Schwellen 1,5e-4…1,05e-3). Waren
Drohnen- und Nicht-Drohnen-Aufnahmen unterschiedlich lang, kann das Modell die Länge gelernt haben
statt der Drohne. Mit den Trainingsdaten prüfen (Längenverteilung je Klasse). Korrektur: Zähler und
Nenner gleich normieren (z. B. beide als Mittel je Frame).

### 3.5 f0-Schätzung (hoch)
Beim Testsignal mit 118 Hz liefert `estimate_f0_v3` 187,5 Hz (bei jeder Länge). Das Raster ist bei
`n_fft = 2048` 23,4 Hz; im Bereich 80–200 Hz gibt es nur 5 mögliche Werte (93,8 / 117,2 / 140,6 /
164,1 / 187,5 Hz). Die Harmonischen-Energien werden am nächstgelegenen Bin gelesen – bei falschem f0
an den falschen Stellen. Das Modell nutzt f0 praktisch nicht (Wichtigkeit 0,000; einzige Schwelle
175,8 Hz = Mitte zwischen zwei Rasterwerten). SDS_110 sucht f0 in 80–350 Hz.

### 3.6 C++-Merkmalsextraktor und Inferenz unvollständig (hoch)
- `estimateF0()` liefert fest 120 Hz, `computeSpectrum()` setzt alle Beträge auf 0.
- `SDS_ModelInference::evalTrees()` ist ein Platzhalter (Bias −0,2); `generated_gbm.cpp` definiert
  eine freie Funktion `evalTrees`, die nicht aufgerufen wird.
- `std::vector` im Extraktor: Heap-Zugriff je Aufruf (auf dem Board vermeiden).

### 3.7 Merkmale pro Aufnahme vs. Datenstrom (mittel)
Die Merkmale sind Mittelwerte über eine ganze Datei, normiert auf deren Spitzenwert. Auf dem Board
gibt es einen kontinuierlichen Datenstrom: Es braucht ein festes Analysefenster, das zur Länge der
Trainingsaufnahmen passt, und eine kausal nachbildbare Normierung (keine Spitzenwert-Normierung
über die Zukunft). SDS_110 rechnet mit 3072-Sample-Frames, Hann, 4096er FFT (11,7 Hz Raster),
AGC nach Bandpass 80–8000 Hz – andere Voraussetzungen als `n_fft = 2048`, Bandpass 200–3000 Hz.

### 3.8 `app/sds_transfere` (niedrig)
Älterer Entwurf eines anderen Modells (MLP 12 → 128 → 64 → 1, Export über ONNX aus
`hbd_model.pkl`, das nicht im Repository liegt). `hbd_features.c`: Schleife bis Index 2000 über ein
Feld mit 1024 Einträgen, Logarithmus auf dem gepackten komplexen Spektrum statt auf Beträgen,
`const`-Eingabe wird von `arm_rfft_fast_f32` überschrieben. Nicht weiter verwendbar.

### 3.9 Hinweis zu `docs/` (V8)
Die V8-Berichte nennen für 1332 echte Drohnen-Dateien durchgehend Score 1,000 und 0 falsch
Negative. Solch perfekte Werte sprechen eher für Überschneidungen zwischen Trainings- und Testdaten
(z. B. Ausschnitte derselben Aufnahme) als für echte Generalisierung – vor einer Weiterverwendung
die Trennung der Datensätze prüfen.

---

## 4. Empfehlung für die Übertragung nach SDS_110

1. Rolle festlegen: 64 Band-Wahrscheinlichkeiten s(t) für Modul 124 (Keras-Modell, dessen
   Trainingsskript fehlt) oder eine Drohnen-Wahrscheinlichkeit als Ersatz der HBD-Entscheidung (BPP).
2. Für BPP vor jeder Einbindung:
   - Merkmale längenunabhängig definieren (3.4), f0-Schätzung verbessern (3.5).
   - Kausal trainieren: SOS-Bandpass mit `sosfilt` statt `filtfilt`, keine Spitzenwert-Normierung;
     möglichst mit derselben Vorverarbeitung wie auf dem Board (oder die Board-Kette in Python
     nachbilden) und festem Analysefenster.
   - Nach dem Neutraining die Längenverteilung je Klasse und die Trennung Train/Test prüfen.
   - Export korrigieren (3.2) und Extraktor exakt wie im Training umsetzen (3.1, 3.6).
   - Übereinstimmung Python ↔ C++ mit festen Referenzvektoren prüfen (Merkmale und
     Wahrscheinlichkeit), z. B. als Host-Test im SDS_110-Projekt (`test/host`).

---

## 5. Prüfskripte (Nachvollziehbarkeit)

Voraussetzung: Python mit numpy ≥ 2, scikit-learn, scipy (für das Modell); librosa für 3.4.
Aufruf jeweils im Wurzelverzeichnis des Repositories.

**Modellkenngrößen (2)**
```python
import joblib, numpy as np
clf = joblib.load("models/bpp/uav_bandpass_model.joblib")
print(clf.n_features_in_, clf.estimators_.shape, clf.learning_rate, clf.max_depth)
print("init log-odds:", clf._raw_predict_init(np.zeros((1, 10))).ravel())
print("Wichtigkeit:", np.round(clf.feature_importances_, 3))
```

**Export vs. scikit-learn (3.2)** – `generated_gbm.cpp` mit einer `main`, die je Zeile 10 Werte
einliest und `evalTrees(x)` ausgibt, übersetzen (`g++ -O2 main.cpp generated_gbm.cpp -o gbm`), dann:
```python
import joblib, numpy as np, subprocess
clf = joblib.load("models/bpp/uav_bandpass_model.joblib")
lo = np.array([90, 1e-4, 1, -15, .05, 1, .5, 2, 1, 1]); hi = np.array([200, 1.2e-3, 90, 8, 55, 190, 95, 30, 42, 28])
X = (lo + (hi - lo) * np.random.default_rng(1).random((2000, 10))).astype(np.float32)
out = subprocess.run(["./gbm"], input="\n".join(" ".join(f"{v:.9g}" for v in r) for r in X),
                     capture_output=True, text=True).stdout.split()
cpp = np.array(out, float); init = clf._raw_predict_init(X[:1]).ravel()[0]
print(np.abs(init + clf.learning_rate * cpp - clf.decision_function(X)).max())
```

**Bandpass (3.3)**
```python
import numpy as np, scipy.signal as sig
ca = [1, -3.36225606, 4.34856225, -2.50947845, 0.52379516]
print(np.abs(np.roots(ca)))                                   # Pole > 1 -> instabil
b, a = sig.butter(4, [200/24000, 3000/24000], btype="bandpass"); print(len(a) - 1)   # Ordnung 8
```

**Längenabhängigkeit (3.4)** – mit `PYTHONPATH=.`:
```python
import numpy as np
from simulation.drone_synth_parrot import synthesize_parrot_drone
from app.train_bpp.bpp_bandpass import apply_bandpass
from app.train_hbo.features_harmonic import harmonic_features
np.random.seed(3)                                   # Synthesizer hat Zufallsanteile
full = synthesize_parrot_drone(sr=48000, duration=10.0)
for d in (1, 2, 5, 10):
    x = apply_bandpass(full[:48000 * d], 48000); x = x / (np.max(np.abs(x)) + 1e-9)
    print(d, harmonic_features(x, 48000, 2048, 512, 6)[:2])
```
