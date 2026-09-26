# Training und Auswertung ML124 (Arbeitspaket 4)

Stand 26.09.2026. Umsetzung der Abschnitte 6 und 8 („Band-Güte“) des Trainingskonzepts
(`docs/Trainingskonzept_ML124.md`) auf dem Datensatz aus AP3 (`docs/Datensatz_ML124.md`).
Merkmalsversion `837ff89cbda34b21`.

> **⚠ PRÜFEN:** Die Verwendung der Drohnendaten im Training ist noch nicht freigegeben
> (`docs/Daten_ML124.md`). Die Modelle erben diesen Vorbehalt.

---

## 1. Ablauf

```bash
# ~/ml_env (TensorFlow 2.13, CPU), im Repo-Wurzelverzeichnis
python app/train_ml124/train.py --context 5 --hidden 48,64 --dropout 0.3 --name k5_h48_d3
python app/train_ml124/evaluate.py model/ml124/k5_h48_d3
```

| Datei | Aufgabe |
|---|---|
| `app/train_ml124/ml124_data.py` | Laden der Merkmale, Kontext-Stapel, Ziel y_b = σ(SNR_b / 3 dB), erste 31 Frames verwerfen |
| `app/train_ml124/train.py` | Training (Keras), Normierung, Ablage in `model/ml124/<name>/` |
| `app/train_ml124/evaluate.py` | Testauswertung ML ↔ HBD, echte Aufnahmen; `eval.json` |

**Modell**: MLP mit 169·K Eingängen, ReLU in den verdeckten Schichten und 64 Sigmoid-Ausgängen
(ein Wert je Band). Die Eingänge sind die Merkmale von 122 für die Frames t, t−1, …, t−K+1,
neuester zuerst. Normiert wird mit Mittelwert und Streuung je Eingang aus den Trainingsframes
(`norm.npz`).

**Training**:

- Verlust: binäre Kreuzentropie mit weichen Zielen.
- Optimierer: Adam 10⁻³, Batch 512.
- Frühes Stoppen nach 10 Epochen ohne Besserung; Lernrate halbieren nach 4 Epochen.
- Seed `20260928`, deterministische Operationen.
- Keine Gewichtung der positiven Bänder, weil ihr Anteil etwa 50 % beträgt.
- Dropout wirkt nur im Training; die Inferenz besteht aus reinen Dense-Schichten.

**Validierung**: 15 % der Drohnen-Clips des Trainingssplits (mit all ihren Gemischen) und 15 %
der Umwelt-Beispiele. Das Modell wird nach dem Validierungsverlust ausgewählt. Der Testsplit
(Fold 5) wird nur zur Auswertung verwendet.

**Modellordner** `model/ml124/<name>/`:

| Datei | Inhalt |
|---|---|
| `config.json` | Architektur, Aktivierungen, Hyperparameter, Validierungs-Drohnen, Merkmalsversion, git-Stand |
| `history.csv` | Verlauf des Trainings |
| `eval.json` | Ergebnisse der Auswertung |
| `model.keras`, `norm.npz` | Nur für das ausgewählte Modell versioniert |

---

## 2. Modellauswahl

Vorgabe aus Konzept Abschnitt 8: Flash ≤ 200 kB, also höchstens etwa 50 000 float32-Parameter
einschließlich Normierung.

| Modell | Kontext K | verdeckt | Dropout / L2 | Parameter | Flash (float32) | beste Epoche | val_loss | Test AUC | Test F1 |
|---|---|---|---|---|---|---|---|---|---|
| ml124_k1 | 1 | 128-64 | 0 / 0 | 34 176 | 134 kB | 5 | 0,3664 | 0,900 | 0,872 |
| k1_d3 | 1 | 128-64 | 0,3 / 0 | 34 176 | 134 kB | 41 | 0,3664 | 0,900 | 0,864 |
| k1_d3_l2 | 1 | 128-64 | 0,3 / 10⁻⁴ | 34 176 | 134 kB | 17 | 0,3981¹ | – | – |
| k3 | 3 | 128-64 | 0 / 0 | 77 440 | 302 kB | 9 | 0,3496 | – | – |
| k3_d3 | 3 | 128-64 | 0,3 / 0 | 77 440 | 302 kB | 18 | 0,3429 | 0,901 | 0,864 |
| k3_d3_l2 | 3 | 128-64 | 0,3 / 10⁻⁴ | 77 440 | 302 kB | 31 | 0,3652¹ | – | – |
| k3_h64_d3 | 3 | 64-64 | 0,3 / 0 | 40 832 | 160 kB | 35 | 0,3465 | – | – |
| k5_d3 | 5 | 128-64 | 0,3 / 0 | 120 704 | 472 kB | 13 | 0,3389 | – | – |
| **k5_h48_d3** | 5 | 48-64 | 0,3 / 0 | 47 904 | 187 kB (+ 7 kB Normierung) | 53 | **0,3414** | 0,896 | 0,868 |
| k8_d3 | 8 | 128-64 | 0,3 / 0 | 185 600 | 725 kB | 23 | 0,3402 | – | – |
| k8_h32_d3 | 8 | 32-64 | 0,3 / 0 | 49 568 | 194 kB | 36 | 0,3526 | – | – |

¹ Einschließlich des L2-Strafterms, daher nicht direkt vergleichbar.

**Ausgewählt: `k5_h48_d3`**. Es hat den kleinsten Validierungsverlust unter den Modellen, die in
200 kB passen.

- Kontext über mehrere Frames hilft: Der Validierungsverlust sinkt von 0,366 (K = 1) auf 0,339
  (K = 5). Mehr als 5 Frames bringt nichts.
- L2 hilft nicht; Dropout 0,3 verbessert den Verlust nur wenig, stabilisiert aber das Training.
- Rechenaufwand: etwa 48 000 Multiplikationen je Frame, bei 216 MHz weit unter der Grenze von
  2 ms (Messung in AP8).

Die Testwerte der übrigen Modelle dienen nur zum Vergleich; ausgewählt wurde nicht nach ihnen.

---

## 3. Ergebnisse auf dem Testsplit (`k5_h48_d3`)

Ziel „Band wird von der Drohne dominiert“ bedeutet SNR_b > 0 dB. Selektiert ist ein Band bei
p_b > θ_sel = 0,5. ML wird wie in 124 über 6 Frames geglättet (`STATE_SMOOTH_FRAMES`). Als HBD
gilt das heutige s(t) aus 124 (Spalten `hbd_s_p_*`) mit Glättung und Gate.

### 3.1 Gemischte Beispiele (240 Gemische, 67 200 Frames)

| Teilmenge | ML AUC Mittel / Min | ML F1 | ML ungeglättet AUC / F1 | HBD AUC Mittel / Min | HBD F1 |
|---|---|---|---|---|---|
| alle Gemische | **0,896** / 0,795 | **0,868** | 0,901 / 0,872 | 0,544 / 0,505 | 0,051 |
| −30…−20 dB | 0,817 / 0,766 | 0,537 | 0,825 / 0,548 | 0,464 / 0,397 | 0,017 |
| −20…−10 dB | 0,814 / 0,672 | 0,648 | 0,827 / 0,664 | 0,492 / 0,448 | 0,028 |
| −10…0 dB | 0,819 / 0,586 | 0,817 | 0,836 / 0,830 | 0,504 / 0,443 | 0,052 |
| 0…10 dB | 0,709 / 0,523 | 0,909 | 0,739 / 0,914 | 0,467 / 0,404 | 0,053 |
| 10…20 dB | 0,670 / 0,441 | 0,971 | 0,686 / 0,968 | 0,453 / 0,270 | 0,062 |

- **Gesamt**: ML trennt die Bänder deutlich (AUC 0,90). Der HBD liegt nahe am Zufall (0,54),
  weil er nur wenige Harmonische markiert; die Labels zählen aber jedes von der Drohne dominierte
  Band, auch mit Breitbandanteil. Für den HBD ist dieser Vergleich daher streng. Seine
  eigentliche Aufgabe (Harmonische für die Peilung) wird in AP7 bewertet.
- **Innerhalb einer SNR-Klasse** ist die AUC niedriger (0,67–0,82). Das Gesamtergebnis profitiert
  auch davon, zu erkennen, *wie laut* die Drohne im Beispiel ist. Bei hohem SNR sind fast alle
  Bänder positiv, und die wenigen negativen Bänder (Umwelt-Spitzen) sind schwer zu treffen.
- Die Glättung kostet etwas AUC (0,901 → 0,896), stabilisiert aber die Selektion über die Zeit.

### 3.2 Drohnentypen und Hold-out nach Typ

| Drohnentyp (Rotoren×Blätter) | Test-Gemische | ML AUC / F1 | ML ohne 6-Rotor-Training: AUC / F1 | HBD AUC / F1 |
|---|---|---|---|---|
| 4x2 | 124 | 0,909 / 0,879 | 0,899 / 0,868 | 0,540 / 0,043 |
| 4x3 | 24 | 0,877 / 0,785 | 0,872 / 0,771 | 0,546 / 0,082 |
| 6x2 | 48 | 0,883 / 0,878 | **0,876 / 0,870** | 0,562 / 0,062 |
| 6x3 | 20 | 0,844 / 0,832 | **0,796 / 0,739** | 0,575 / 0,052 |
| 8x2 | 12 | 0,943 / 0,910 | 0,943 / 0,912 | 0,475 / 0,023 |
| 8x3 | 12 | 0,871 / 0,868 | 0,842 / 0,796 | 0,533 / 0,058 |

Für den Hold-out wurde `k5_h48_d3_ohne6rot` ohne 6-Rotor-Drohnen trainiert (67 von 240
Trainings-Clips entfallen). Unbekannte 6-Rotor-Drohnen erkennt es fast so gut wie das volle
Modell (6x2), bei 6x3 aber mit spürbarem Verlust (AUC −0,05). Die Übertragung auf neue Typen
funktioniert, mehr Vielfalt im Training ist aber nützlich.

### 3.3 Fehlalarme und echte Aufnahmen (ungemischt, nie im Training)

Da die Band-Wahrheit fehlt, wird nur die Selektion gezählt.

| Aufnahmen | Dateien | ML: Frames mit ≥ 1 / ≥ 3 Bändern, Bänder im Mittel | HBD: Frames mit ≥ 1 / ≥ 3 Bändern, Bänder im Mittel |
|---|---|---|---|
| Testsatz nur Umwelt (gemischt erzeugt) | 60 | 5,5 % / 4,9 %, 1,64 | 20,5 % / 12,4 %, 0,88 |
| ESC-50 (Fold 5) | 402 | 3,7 % / 3,1 %, 1,04 | 10,1 % / 6,4 %, 0,49 |
| DDS Hintergrund | 6 | 0,5 % / 0,2 %, 0,03 | 0,6 % / 0,3 %, 0,02 |
| DDS Hubschrauber | 6 | 3,2 % / 2,5 %, 0,74 | 0,7 % / 0,2 %, 0,02 |
| **Drohne (DDS, echt)** | 30 | 21,9 % / 20,5 %, 9,36 | 33,3 % / 22,4 %, 1,33 |

- **Fehlalarme**: Auf Umweltgeräuschen selektiert ML seltener als der HBD (ESC-50 3,7 % gegen
  10,1 % der Frames). Auf DDS-Hubschrauber liegt ML dagegen höher (3,2 % gegen 0,7 %). Beide
  erreichen das Ziel „0 %“ aus Abschnitt 8 nicht. Dort ist allerdings die Fehlalarm-Rate der
  *Reports* nach 126/128 gemeint; sie wird in AP7 mit `m_selection` gemessen.
- **Echte Drohnen**: ML reagiert „alles oder nichts“. Ist es aktiv, selektiert es meist 30–60 Bänder. In 9 von 30 Dateien (DRONE_004–006, 008–011, 015, 021) selektiert es gar
  nichts, während der HBD dort in 7–68 % der Frames Harmonische findet. Umgekehrt ist ML bei
  DRONE_017/018/024/029 in über 89 % der Frames aktiv.
- Die DDS-Aufnahmen sind Feldaufnahmen mit Umgebungsgeräusch. Sie sind weniger tonal als die
  synthetischen Drohnen: Spitze/Median innerhalb eines Bands 1,3 gegenüber 1,9. Ohne getrennte
  Komponenten ist offen, welches Verfahren dort richtig liegt.

---

## 4. Bewertung und nächste Schritte

1. **Auf synthetischen Gemischen erfüllt ML die Band-Güte klar** (AUC 0,90, F1 0,87 bei
   θ_sel = 0,5). Das Modell passt in Flash und Rechenzeit. → Export (AP5) mit `k5_h48_d3`.
2. **Echte Drohnen sind der Schwachpunkt**, weil alle Trainingsdrohnen synthetisch sind. Nötig
   sind saubere echte Drohnenaufnahmen für die Mischung (AP2, Konzept 5.2), nach Freigabe
   (⚠ PRÜFEN).
3. **Breitband-Selektion**: Die Labels markieren auch breitbandige Drohnenenergie. Für die Peilung
   in 126 kann das nützlich (mehr Bänder, mehr Energie) oder schädlich (Bänder ohne kohärente
   Phase) sein. Das entscheidet der Vergleich mit dem HBD in AP7 (`m_selection`,
   `m_bearing_drone`). Falls nötig, lässt sich das Label auf tonale Anteile beschränken.
4. **Fehlalarme**: Hubschrauber und einzelne ESC-Klassen sind die harten Negativbeispiele. Falls
   AP7 zu viele Fehl-Reports zeigt: θ_sel anheben, weitere Umweltdaten ohne Drohne oder das
   HBD-Gate beibehalten (Konzept Abschnitt 6).
