# Trainingskonzept: Machine Learning Module 124 (SDS_110)

Stand 26.09.2026. Grundlage: SDS_110 Commit `caaa74d`, ML_Test Commit `c1ec90c`,
Analyse `docs/Analyse_BPP_Transfer.md`. Der Patenttext lag nicht vor; die Patentbezüge sind den
Verweisen im SDS_110-Code entnommen (Modul- und Abschnittsangaben in Klammern).

---

## 1. Ziel

Ein trainiertes Modell für Modul 124, das je Analyse-Frame den UAV-quellspezifischen akustischen
Zustand liefert und den klassischen Harmonic Band Detector (HBD) ersetzen oder ergänzen kann.

| Anforderung | Quelle |
|---|---|
| Ausgang s(t) = (p₁ … p_B), B = 64, ein Wert je Band | Abschnitt 3, FIG. 4 (`Machine_Learning_Module_124.hpp`, `Candidate_Report_140.hpp`) |
| Bänder: 64 × 62,5 Hz ab 80 Hz (bis 4080 Hz) | `SDS_110_Config.hpp` (`NUM_BANDS`, `BAND_WIDTH_HZ`, `BAND_LO_HZ`) |
| Eingang: Merkmale aus 122 des Referenzkanals | Claim 1 (b), Abschnitt 2 (`Feature_Extraction_Module_122.hpp`) |
| Frames 64 ms, 50 % Überlappung (31,25 Frames/s) | Abschnitt 2 (`Frame_Assembler.hpp`) |
| Verwendung: Selektion S(t) = {k : p_b(k) > θ_sel}, Gewicht w = p^γ, darauf quellkonditionierte GCC-PHAT | Abschnitte 4, 5 (`Correlation_Processing_Module_126`) |
| Rückkopplung ŝ der Tracking Unit senkt θ_sel für Referenzbänder | Abschnitt 10 |
| Keine Trajektorienbildung in SDS 110 | Claim 10 |

**Bedeutung von p_b** (abgeleitet aus der Verwendung in 126): Wahrscheinlichkeit, dass Band b im
aktuellen Frame von Energie der UAV-Quelle dominiert wird. Das Modell ist damit kein reiner
Drohnen-Detektor, sondern liefert eine **Bandmaske**: Die Qualität zeigt sich darin, ob die
selektierten Bänder die Peilung verbessern und ob ohne Drohne keine Bänder selektiert werden.

---

## 2. Lehren aus den bisherigen Modellen

| Problem (Analyse / Befunde) | Maßnahme in diesem Konzept |
|---|---|
| Merkmale im Training anders berechnet als auf dem Board (BPP, HBD_ML) | Merkmale **mit dem Board-Code** berechnen (Abschnitt 4) |
| Trainingsskript und Merkmalsdefinition nicht auffindbar (HBD_ML) | Alles versioniert in ML_Test; Merkmalsversion im Modell-Header |
| Export unvollständig (Startwert/Lernrate fehlten, BPP) | Export mit Referenzvektoren, automatischer Vergleich C++ ↔ Python |
| Nicht kausale Vorverarbeitung (`filtfilt`, Spitzenwert-Normierung) | Nur kausale Verarbeitung, identisch zum Board (118) |
| Merkmal abhängig von der Aufnahmedauer (`harmonic_ratio`) | Nur Frame-Merkmale; Längen je Klasse prüfen |
| Verdächtig perfekte Ergebnisse (V8) | Trennung nach Aufnahmen und Drohnentypen, Hold-out-Tests |
| Aktivierungsfunktionen nicht dokumentiert | Architektur vollständig im Header (Schichten, Aktivierungen, Normierung) |

---

## 3. Gesamtablauf

```
Rohdaten (WAV, mono oder 8-kanalig)          Simulator (SDS_110 Harness)
        │                                              │
        ▼                                              ▼
 [A] Merkmalswerkzeug (C++ aus SDS_110: 114 → 118 → Frame_Assembler → 122)
        │  Merkmale je Frame + Zeitstempel + Merkmalsversion
        ▼
 [B] Label-Erzeugung (Python, ML_Test): Band-SNR aus getrennten Komponenten → Zielwert je Band
        ▼
 [C] Training (Python, ML_Test): MLP, 64 Sigmoid-Ausgänge
        ▼
 [D] Export: C-Header (Gewichte, Aktivierungen, Normierung, Version, Referenzvektoren)
        ▼
 [E] SDS_110: Inferenz in 124 (umschaltbar HBD / ML), Host-Tests, Board-Messung
```

---

## 4. [A] Merkmale mit dem Board-Code

- Neues Host-Werkzeug im SDS_110-Repository (z. B. `tools/features/`), gebaut wie `test/host`
  aus den unveränderten Quellen 114, 118, `Frame_Assembler`, 122 (CMSIS-Stubs, FFT über kiss_fft).
- Eingabe: WAV mit 48 kHz, 1 Kanal (wird als Referenzkanal eingespeist) oder 8 Kanäle.
  Die Datei wird **am Stück** verarbeitet (AGC/NS und Filterzustände laufen durch wie auf dem Board);
  kein Zurücksetzen je Ausschnitt.
- Ausgabe je Frame: Zeit des ersten Samples, 169 Merkmale (band_log_power[64], mel[40],
  spectral_flux, band_am_depth[64]), optional HBD-Zustand (f0, Band-SNR) als Vergleich/Zusatzmerkmal.
  Format `.npy` + JSON-Metadaten.
- **Merkmalsversion**: Hash über die relevanten Konstanten aus `SDS_110_Config.hpp` (Frame, Hop,
  N_FFT, Bänder, Mel, AGC/NS-Zeitkonstanten, Bandpass) und den SDS_110-Commit. Wird in die Merkmale,
  das Modell und den exportierten Header geschrieben; die Firmware prüft ihn beim Übersetzen
  (`static_assert`). Jede Änderung an 118/122 erzwingt so ein Neutraining.
- Cepstrum-Merkmale (f0, Peak) nur, wenn sie sich im Training als nützlich erweisen – dann **zuerst**
  in 122 (C++) umsetzen und über das Werkzeug erzeugen, nie getrennt in Python.
- Abtastrate: Board 47 991 Hz (Befund 5), Aufnahmen 48 000 Hz – 186 ppm, vernachlässigbar;
  im Werkzeug mit 48 000 Hz rechnen.

Kontext über mehrere Frames (z. B. aktueller + 2 vorige) ist möglich; das Werkzeug liefert
Einzel-Frames, das Stapeln geschieht im Training und identisch in 124.

---

## 5. [B] Daten und Labels

### 5.1 Label-Definition (Bandmaske)
Zielwert je Frame und Band aus dem **Band-SNR der UAV-Komponente**:

  SNR_b = 10·log10( P_b(UAV) / P_b(Rest) ),  Ziel y_b = σ((SNR_b − SNR_0) / s)

mit P_b als Leistung im Band b nach derselben Kette (Fenster, FFT, Bandgrenzen wie 122).
Vorschlag SNR_0 = 0 dB (UAV dominiert), s = 3 dB (weicher Übergang). Alternativ hart
y_b = [SNR_b > 0 dB]. Die Komponenten werden **getrennt** durch das Merkmalswerkzeug geschickt,
damit Fenster und Vorverarbeitung übereinstimmen (Verstärkung von 118 aus dem Gemisch übernehmen).

### 5.2 Datenquellen
| Quelle | Label | Zweck |
|---|---|---|
| Simulator (`Signal_Simulator`, erweiterbar um `SDS_SimDrone`-Modell: Jitter, Drift, FM) | exakt (Komponenten bekannt) | Grundtraining, Randfälle (f0 80–350 Hz, AM, Distanz, SNR) |
| Echte Drohnenaufnahmen mit hohem SNR (nah, ruhige Umgebung) + unabhängige Störgeräusch-Aufnahmen, gemischt mit kontrolliertem SNR | aus den Komponenten | Hauptdatensatz, reale Spektren |
| Reine Störgeräusche (Wind, Verkehr, Sprache, Vögel, Maschinen, Generator) | alle Bänder 0 | Fehlalarm-Unterdrückung |
| Echte Drohnenaufnahmen im Feld (ungemischt) | schwach: Harmonische aus f0-Verfolgung, nur bei hohem SNR | Validierung, ggf. Feintuning |
| Aufnahmen mit dem Board (READ-Modus, IM69D130, 8 Kanäle) | wie oben | Anpassung an Mikrofon und Gehäuse, Endabnahme |

Vorhandene Bestände (laut `bpp_train.py`, V8-Berichten): `data/train_48k/…`, ca. 1332 Drohnen- und
2000 Nicht-Drohnen-Dateien – Umfang, Mikrofone, Längen und Herkunft sind vor Verwendung zu
inventarisieren (Abschnitt 9).

> **⚠ PRÜFEN:** Die Verwendung der Drohnendaten (synthetisch und echt) im Training ist noch nicht
> freigegeben und muss vorher überprüft werden – Prüfpunkte in `docs/Daten_ML124.md` (Hinweis am
> Anfang). Status: offen.

### 5.3 Datentrennung und Kontrollen
- Aufteilung **nach Aufnahme** (nie Ausschnitte derselben Aufnahme in Training und Test) und
  zusätzlich ein **Hold-out nach Drohnentyp** und nach Aufnahmeort.
- Je Klasse prüfen: Längen, Pegel, Aufnahmegerät, Abtastrate – keine Klasse darf an solchen
  Nebenmerkmalen erkennbar sein.
- Anteil positiver Bänder protokollieren (erwartet klein) und im Verlust gewichten.

---

## 6. [C] Modell und Training

- **Architektur (Startpunkt)**: MLP 169 (bzw. 169·Kontext) → 128 → 64 → 64, ReLU in den verdeckten
  Schichten, **Sigmoid** am Ausgang (unabhängig je Band). Kleiner als das bisherige HBD_ML
  (171 → 128 → 128 → 64), damit Flash und Rechenzeit Reserve haben; bei Bedarf vergrößern.
- **Normierung**: Mittelwert/Streuung je Merkmal aus dem Trainingsdatensatz, im Header abgelegt.
- **Verlust**: binäre Kreuzentropie je Band (weiche Ziele möglich), Gewichtung positiver Bänder.
- **Framework**: Keras oder PyTorch; entscheidend ist der eigene Export (Abschnitt 7), nicht das
  Framework. Zufallsstartwerte festhalten, Trainingslauf mit Konfiguration und Datensatz-Liste
  speichern.
- **Nachbearbeitung wie bisher in 124**: Glättung über `STATE_SMOOTH_FRAMES`; das HBD-Gate
  (Haltezeit, Befund 24) wird zunächst beibehalten und kann nach der Bewertung entfallen.
- **Später**: int8-Quantisierung nur, wenn Flash oder Rechenzeit es erfordern (dann Referenzvektoren
  für die quantisierte Fassung).

---

## 7. [D] Export

Ein Exportskript in ML_Test erzeugt einen Header für SDS_110 mit:
- Schichtgrößen, Gewichten, Biases (float32, `%.9g`), Aktivierung je Schicht (Enum),
- Mittelwert/Streuung der Merkmale, Kontextlänge, Merkmalsreihenfolge,
- Merkmalsversion (Abschnitt 4), Modellversion, Trainings-Commit, Datum,
- **Referenzvektoren**: N ≥ 32 Eingänge aus dem Testsatz mit den Python-Ausgaben.

Das Skript liegt versioniert in ML_Test (nicht in einer externen Sitzung) und prüft nach dem Export
selbst, dass eine Nachrechnung des Headers in NumPy die Python-Ausgaben reproduziert.

---

## 8. [E] Integration und Abnahme in SDS_110

- 124: generische kleine MLP-Inferenz (Dense + Aktivierung; CMSIS-DSP im Projekt ist V1.7.0 ohne
  `arm_mat_vec_mult_f32` → je Neuron `arm_dot_prod_f32` oder `arm_mat_mult_f32`),
  Umschalter HBD / ML in `SDS_110_Config.hpp`, HBD bleibt als Rückfall.
- Host-Test `t_ml124`: C++-Inferenz gegen die Referenzvektoren, max. Abweichung ≤ 1e-5.
- Messungen mit den vorhandenen Host-Tests, HBD und ML nebeneinander (`doc/Host_Tests.md`):

| Kriterium | Messung | Abnahme (Vorschlag) |
|---|---|---|
| Reports Drohne 3 dB / 0 dB | `m_selection` | ≥ HBD (100 % / 56 %) |
| Fehlalarm-Reports Wind, Stille, Einzelton | `m_selection`, Langlauf `noise` | 0 % wie HBD |
| Anteil selektierter Bänder mit Harmonischer | `m_selection` | ≥ 85 % |
| Peilfehler Median / 95 % bei 10 dB | `m_bearing_drone` | nicht schlechter als HBD (0,88° / 3,1°) |
| Band-Güte auf Hold-out (echte Daten) | Python | AUC je Band, F1 bei θ_sel = 0,5 |
| C++ ↔ Python | `t_ml124` | ≤ 1e-5 |
| Rechenzeit je Frame auf dem Board | Task-Statistik | ≤ 2 ms (Frame-Takt 32 ms) |
| Flash | Linker | ≤ 200 kB |

- Board: Rechenzeit und Ergebnisse mit echten Mikrofonen (setzt die zurückgestellten Punkte 1–4
  voraus).

---

## 9. Arbeitspakete

| # | Paket | Repository | Ergebnis |
|---|---|---|---|
| 1 | Merkmalswerkzeug mit Versionshash | SDS_110 `tools/features` | CLI `WAV → .npy/.json` |
| 2 | Inventur vorhandener Daten (Anzahl, Längen, Pegel, Geräte, Drohnentypen) | ML_Test | Datensatz-Übersicht |
| 3 | Label-Erzeugung (Simulator, Mischung) | ML_Test `app/train_ml124` | Merkmale + Bandziele |
| 4 | Training + Auswertung (AUC/F1 je Band, Hold-out) | ML_Test | Modell, Bericht |
| 5 | Export mit Referenzvektoren | ML_Test | Header für SDS_110 |
| 6 | Inferenz in 124, Umschalter, `t_ml124` | SDS_110 | Firmware + Host-Test |
| 7 | Vergleich HBD ↔ ML mit den Host-Messungen | SDS_110 | Bericht, Entscheidung |
| 8 | Board-Aufnahmen und Endabnahme | beide | Rechenzeit, Feldtest |

---

## 10. Offene Fragen

1. Patent Abschnitt 3: Gibt es Vorgaben zu Trainingsdaten, Labels oder Modelltyp, die über
   „s(t) mit B Bändern“ hinausgehen? (Hier nur aus dem Code erschlossen.)
2. Welche Drohnenaufnahmen stehen zur Verfügung (Typen, Abstände, SNR, Mikrofon, mono/mehrkanalig)?
   Gibt es saubere Aufnahmen nahe an der Drohne für die Mischung?
3. Können Aufnahmen mit dem Board selbst gemacht werden (READ-Modus), sobald die Punkte 1–4 gelöst
   sind?
4. Soll ML den HBD ersetzen oder ergänzen (HBD als Gate/zusätzliches Merkmal)?
5. Bevorzugtes Framework (Keras oder PyTorch)?
6. Wird die ursprüngliche Claude-Sitzung zu `hbd_ml_124.keras` noch gefunden? (Nur zum Vergleich;
   ein Neutraining ist wegen der Änderungen an 118/122 ohnehin nötig.)
