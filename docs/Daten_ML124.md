# Daten für das Training von Modul 124

Stand 26.09.2026. Zugehörig: `docs/Trainingskonzept_ML124.md` (Abschnitt 5).
Die Audiodaten liegen unter `data/48kHz/` und sind **nicht** im Repository (`.gitignore`); sie
werden mit den Skripten in `app/data_ml124/` reproduzierbar erzeugt. Die vollständige Liste der
Trainings- und Testdaten steht in `docs/Datenliste_ML124.md` (und `.csv`).

> **⚠ PRÜFEN – Verwendung der Drohnendaten im Training**
> Die Verwendung der Drohnendaten (`Dronen/sim` und `Dronen/real_dds` unter `data/48kHz/`) im Training ist
> **noch nicht freigegeben** und muss vor dem Training überprüft werden, insbesondere:
> 1. Realitätsnähe des synthetischen Modells (bisher nur Spektralverteilung gegen 30 DDS-Aufnahmen
>    verglichen; Modulation, Drehzahldynamik, Drohnentypen und -größen nicht belegt).
> 2. Ob ein überwiegend synthetisch trainiertes Modell auf echte Drohnen übertragbar ist
>    (Validierung ausschließlich mit echten Aufnahmen).
> 3. Repräsentativität der echten Drohnen (30 Clips, eine Quelle, Drohnentypen unbekannt).
> 4. Rolle von `real_dds` (nur Validierung vs. Training) und der Datentrennung nach Aufnahmen.
> Status: offen.

---

## 1. Übersicht

Ablage: `data/48kHz/{train,test}/{Dronen,Umwelt}/<Teilmenge>/…`

| Teilmenge | Inhalt | Training | Test | Lizenz |
|---|---|---|---|---|
| `Dronen/sim/` | synthetische, saubere Mehrrotor-Drohnen (ohne Umgebungsgeräusch) + JSON je Clip, 10 s | 240 | 60 | eigene Erzeugung |
| `Dronen/real_dds/` | echte Drohnenaufnahmen (Drone-detection-dataset), 10 s | – | 30 | CC0 1.0 |
| `Umwelt/esc50/<Kategorie>/` | ESC-50, 50 Klassen Umweltgeräusche, 5 s | 1598 | 402 | **CC BY-NC 3.0** |
| `Umwelt/dds/background/`, `dds/helicopter/` | Hintergrund und Hubschrauber (Drone-detection-dataset), 10 s | 48 | 12 | CC0 1.0 |
| **gesamt** | | **1886 (181 min)** | **504 (51 min)** | |

Format: 48 kHz, mono, WAV (Umwelt und `real_dds` PCM_16, `sim` PCM_24), zusammen 1,4 GB.
Stereo-Aufnahmen (DDS) werden auf **Kanal 0** reduziert, nicht gemittelt: Das Board wertet ein
Referenzmikrofon aus; Mitteln zweier Mikrofone senkt diffusen Hintergrund um 1–2 dB (stärker bei
hohen Frequenzen) und wirkt als Kammfilter (gemessen: Korrelation der DDS-Kanäle bei Hintergrund
im Median 0,44, bei Drohnen 0,86; 16 der 90 Dateien sind Doppel-Mono).

Metadaten in `data/48kHz/`: `meta_umwelt.csv`, `meta_dronen_real.csv`, `meta_dronen_sim.csv` und
die zusammengeführte `datenliste.csv` (Spalten `split`, `group`, `subset`, `category`, `file`,
`fold`, `duration_s`, `source`, `license`) sowie `sim/*.json`.

**Aufteilung Training/Test** (`split_of()` in `common.py`): Test = Fold 5 (20 %), Training =
Folds 1–4 (Folds für Kreuzvalidierung erhalten).
- ESC-50: Folds aus dem Datensatz, aufgeteilt nach Originalaufnahme (`src_file`). Laut Datensatz
  liegen Clips derselben Aufnahme im selben Fold; 4 Aufnahmen verletzen das, 2 davon über die
  Grenze Training/Test – deren Clips liegen jetzt alle im Test.
- DDS: Fold = (laufende Nummer − 1) % 5 + 1 – der Aufnahmezusammenhang ist unbekannt, daher für
  die Endbewertung zusätzlich nach Quelle trennen.
- `sim`: Fold = Index % 5 + 1 (jeder Clip ist eine unabhängige Drohne).
- `real_dds`: vollständig im Test (Validierung der Übertragbarkeit nur an echten Drohnen;
  Rolle noch offen, siehe PRÜFEN-Punkt 4).
- Beim Mischen (Drohne + Umwelt) nur Dateien desselben Splits kombinieren.

---

## 2. Quellen und Lizenzen

- **ESC-50** – K. J. Piczak, „ESC: Dataset for Environmental Sound Classification“, ACM MM 2015;
  lokal aus `../audio_ml/datasets/esc50/ESC-50-master`. **CC BY-NC 3.0 (nicht kommerziell)**:
  Für Forschung unproblematisch; ob ein damit trainiertes Modell in einem kommerziell verwerteten
  Produkt eingesetzt werden darf, ist vor einer Verwertung zu klären (ggf. durch Daten unter
  freier Lizenz ersetzen).
- **Drone-detection-dataset (DDS)** – F. Svanström, F. Alonso-Fernandez, C. Englund, „A Dataset for
  Multi-Sensor Drone Detection“, Data in Brief 2021; lokal aus
  `../audio_ml/datasets/Drone-detection-dataset`. CC0 1.0. Außenaufnahmen 44,1 kHz Stereo.
- **Synthetisch** – `app/data_ml124/gen_dronen_sim.py`, Seed 20260926 + Index.

Hinweise zu den Klassen:
- ESC-50 enthält drohnenähnliche Klassen (u. a. `helicopter`, `airplane`, `engine`, `chainsaw`,
  `vacuum_cleaner`, `hand_saw`). Sie sind als Negativbeispiele (Label 0) gedacht und besonders
  wichtig gegen Fehlalarme; bei der Auswertung getrennt betrachten.
- `helicopter` kommt aus zwei Quellen (ESC-50 und DDS) – Unterscheidung über `source`.
- Nicht verwendet: DADS (Hugging Face, MIT, 180 000 Clips) – lokal nur Metadaten; bei Bedarf als
  großer realer Validierungsbestand nachladen.

---

## 3. Erzeugung

Im Repository-Wurzelverzeichnis, Python mit numpy, scipy, soundfile (z. B. `~/ml_env`):

Die Quelldatensätze (`esc50/`, `Drone-detection-dataset/`) werden in dieser Reihenfolge gesucht:
Umgebungsvariable `SDS_AUDIO_ML`, `../audio_ml/datasets` neben dem Repository,
`../../Copilot_Projekt/audio_ml/datasets`. Arbeitskopie mit den Daten: `Claude_Projekt/ML_Test`.

```bash
python app/data_ml124/prepare_umwelt.py          # Umwelt (≈ 35 s)
python app/data_ml124/prepare_dronen_real.py     # Dronen/real_dds
python app/data_ml124/gen_dronen_sim.py --n 300  # Dronen/sim (≈ 2 min mit 16 Prozessen)
python app/data_ml124/make_liste.py              # datenliste.csv, docs/Datenliste_ML124.{md,csv}
```

Die Umtastung 44,1 → 48 kHz erfolgt polyphas (`resample_poly`, 160/147). Wo sie über
Vollaussteuerung schwingt, wird die Datei minimal herunterskaliert; der Faktor steht in `gain`
(betrifft 559 ESC-50- und 4 DDS-Dateien). `gen_dronen_sim.py` ist bitgenau reproduzierbar (geprüft: Clips 0 und
177 erneut erzeugt, SHA-256 identisch).

---

## 4. Modell der synthetischen Drohnen

> ⚠ PRÜFEN: Verwendung im Training noch nicht freigegeben (siehe Hinweis am Anfang).

Je Clip (Parameter in `sim/*.json`, Drehzahlverläufe je Rotor mit 50 Hz in `rotor_rps`):

| Aspekt | Modell | Bereich |
|---|---|---|
| Aufbau | Rotoren / Blätter | 4, 6, 8 (60/25/15 %) / 2, 3 (70/30 %) |
| Grundfrequenz | Blattfrequenz (BPF) | 80–350 Hz, log-gleichverteilt (wie HBD-Suchbereich) |
| Drehzahl je Rotor | statischer Versatz + Regelschwankung (Ornstein-Uhlenbeck) | σ 1,5 % bzw. 0,3–1,2 %, τ 0,3–2 s |
| Manöver, Bewegung | gemeinsame Drehzahländerung; Doppler | ±0–8 %, 0,05–0,4 Hz; Radialgeschw. bis ±15 m/s |
| BPF-Harmonische | Abfall k^−α, Hüllkurve ±3 dB je Harmonische | α 0,3–1,2, bis 8 kHz |
| Wellen-Harmonische | Unwucht (keine BPF-Vielfachen) | −30…−12 dB, bis 1,5 kHz |
| Motor-Pfeifen | elektrische Harmonische, Polpaare × Drehzahl | 6–7 Polpaare, −25…−10 dB |
| Amplitudenmodulation | einmal je Umdrehung | Tiefe 0,05–0,3 |
| Breitbandrauschen | Leistung ∝ 1/f^β, 150–8000 Hz, BPF-moduliert | β 0,3–1,0, −12…+3 dB rel. tonal |
| Luftdämpfung | ISO 9613-1, 20 °C, 70 % rF (Näherung) | Distanz 15–300 m (log) |
| Bodenreflexion | Zweistrahlmodell, Mikrofon 1,5 m | Flughöhe 10–120 m, Reflexion 0,5–0,9 |
| Pegel | RMS −26 dBFS (Distanz wirkt nur spektral) | Pegel/SNR wird beim Mischen gesetzt |

Erzeugter Satz: 300 Clips; Rotoren 4/6/8 = 173/84/43, Blätter 2/3 = 221/79; BPF 80–347 Hz
(Quartile 119/164/254 Hz); Distanz 15–295 m (Median 69 m); höchste Spitze −10,1 dBFS.

---

## 5. Prüfungen

**Harmonische an der erwarteten Stelle** (Probe, 8 Clips): Spektralspitzen bei k·BPF (BPF aus den
Metadaten) 5–37 dB über dem lokalen Median; fehlende Spitzen fallen mit Auslöschungen durch die
Bodenreflexion zusammen.

**Spektralverteilung im Vergleich zu echten Drohnen** (Energieanteil in %, Median und 10–90 %):

| Band | synthetisch (40 Clips) | echt DDS (30 Clips) | DDS-Hintergrund |
|---|---|---|---|
| 0–80 Hz | 0,2 (0,0–0,8) | 3,9 (1,3–36) | 7,3 |
| 80–500 Hz | 43,9 (21–71) | 26,0 (8–55) | 33,8 |
| 500–1500 Hz | 26,9 (15–39) | 18,4 (7–36) | 9,4 |
| 1,5–4 kHz | 17,5 (10–30) | 12,0 (5–19) | 6,0 |
| 4–8 kHz | 7,5 (2–18) | 13,3 (2–44) | 2,8 |
| > 8 kHz | 0,2 (0,0–0,8) | 4,9 (0,5–19) | 0,6 |

Die echten Drohnenaufnahmen liegen im Median 12 dB über dem Hintergrund; ihr Anteil über 1,5 kHz
stammt überwiegend von der Drohne. Eine erste Modellfassung (α 0,7–1,6, Breitband −25…−8 dB, ohne
Motor-Pfeifen) hatte 82 % der Energie in 80–500 Hz und wurde daraufhin angepasst. Abweichungen
unter 80 Hz (Wind/Hintergrund in den echten Aufnahmen) und über 8 kHz liegen außerhalb des von
118/122 genutzten Bereichs (Bandpass 80–8000 Hz).

---

## 6. Nächste Schritte (Trainingskonzept)

1. Merkmalswerkzeug mit dem Board-Code (SDS_110, Arbeitspaket 1).
2. Mischung Drohne + Umwelt mit kontrolliertem SNR und Label-Erzeugung aus dem Band-SNR der
   getrennten Komponenten (Arbeitspaket 3).
3. `real_dds`: schwache Labels über f0-Verfolgung, nur für Validierung.
