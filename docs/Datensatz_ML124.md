# Datensatz ML124 – Merkmale und Band-SNR-Labels (Arbeitspaket 3)

Stand 26.09.2026. Umsetzung von Abschnitt 5 des Trainingskonzepts (`docs/Trainingskonzept_ML124.md`):
Drohne und Umwelt werden mit kontrolliertem SNR gemischt. Merkmale und Band-SNR berechnet das
Merkmalswerkzeug von SDS_110 im Label-Modus (`sds_features --hbd --label`, Board-Code).

> **⚠ PRÜFEN:** Die Verwendung der Drohnendaten im Training ist noch nicht freigegeben
> (`docs/Daten_ML124.md`). Der Datensatz erbt diesen Vorbehalt.

---

## 1. Erzeugen und prüfen

```bash
make -C ../SDS_110/tools/features            # Werkzeug bauen (oder SDS_FEATURES=<pfad> setzen)
python app/train_ml124/make_dataset.py --jobs 16
python app/train_ml124/check_dataset.py
```

Die Laufzeit beträgt etwa 1–2 min bei 16 Prozessen. Der Datensatz ist reproduzierbar, weil jedes
Beispiel einen eigenen Seed hat (Basis-Seed `20260927`). Die Ausgabe liegt unter
`data/48kHz_features/` (558 MB) und wird nicht versioniert (`.gitignore`).

---

## 2. Aufbau eines Beispiels

| Schritt | Festlegung |
|---|---|
| Drohne | Synthetische Drohne (`Dronen/sim`) desselben Splits, 10 s, auf RMS 1 normiert. Jeder Clip wird 4-mal mit anderem Umweltanteil verwendet. |
| Umwelt-Spur | Zufällige Umwelt-Clips desselben Splits (ESC-50 sowie DDS-Hintergrund und -Hubschrauber). Jeder Clip wird auf RMS 1 normiert; die Clips werden mit 20 ms Überblendung zu 10 s aneinandergehängt. |
| Umgebungs-Grundpegel | DDS-Hintergrund desselben Splits, 10–25 dB unter der Umwelt-Spur und zu ihr addiert. Danach wird die Summe auf RMS 1 normiert. |
| Misch-SNR | 10·log10(P(Drohne)/P(Umwelt)) über den ganzen Ausschnitt, gleichverteilt zwischen **−30 und +20 dB** |
| Pegel | Der RMS des Gemischs ist gleichverteilt zwischen −45 und −15 dBFS. Liegt die Spitze über −1 dBFS, wird der Pegel gesenkt. |
| Nur Umwelt | 25 % zusätzliche Beispiele ohne Drohne; alle Labels sind 0. |
| Merkmale | Aus dem Gemisch: 169 Merkmale sowie 69 HBD-Spalten zum Vergleich |
| Labels | `snr_db_00…63` = 10·log10(P_b(Drohne)/P_b(Umwelt)) je Band nach der Kette (±80 dB). Das Ziel y_b = σ(SNR_b / 3 dB) wird erst im Training gebildet (5.1). |

**Label-Modus des Werkzeugs** (SDS_110 `doc/Merkmalswerkzeug.md`):

- Beide Anteile werden als 24-bit-Werte addiert.
- Für die Merkmale läuft das Gemisch durch die Board-Kette.
- Jeder Anteil läuft durch einen eigenen 118 (nur Bandpass). Darauf wird die NS·AGC-Rampe
  angewendet, die 118 im Gemisch berechnet hat. Danach folgen eigener Frame_Assembler und eigene 122.
- Selbstkontrolle: Die Abweichung |Gemisch − (Drohne + Umwelt)| nach 118 wird je Beispiel im JSON
  festgehalten.

**Aufteilung**: Das Beispiel übernimmt den Split seiner Quellen (Test = Fold 5,
`docs/Daten_ML124.md`). Drohne, Umwelt-Spur und Grundpegel stammen immer aus demselben Split.
Echte DDS-Drohnen sind nur Testdaten; sie werden hier nicht gemischt, sondern in AP4 separat
ausgewertet.

---

## 3. Ausgabe

| Datei | Inhalt |
|---|---|
| `data/48kHz_features/{train,test}/<split>_<nr>.npy` | float32 [311 Frames, 303 Spalten]: `t_s`, 169 Merkmale, 69 HBD-Spalten, `snr_db_00…63` |
| `…/<split>_<nr>.json` | Metadaten des Werkzeugs (Spaltennamen, `feature_version`, Parameter) sowie `recipe` |
| `data/48kHz_features/index.csv` | Je Beispiel: Split, nur Umwelt, Drohnen-Datei, Umwelt- und Grundpegel-Dateien, Grundpegel [dB], Misch-SNR, Pegel, Seed, Frames, übersteuerte Samples, Rekonstruktionsfehler, Merkmalsversion |

**Umfang**: 1500 Beispiele (4,2 h Audio, 466 500 Frames).

| Split | Gemische | Nur Umwelt | Summe |
|---|---|---|---|
| Training | 960 | 240 | 1200 |
| Test | 240 | 60 | 300 |

Kontrollen:

- Merkmalsversion bei allen Beispielen `837ff89cbda34b21`.
- 0 übersteuerte Samples.
- Größter Rekonstruktionsfehler 7,3·10⁻⁴. Er entsteht durch die Rundung der float32-Biquads im
  Bandpass und ist ohne Bandpass exakt 0.

---

## 4. Verteilung der Labels (`check_dataset.py`)

| Misch-SNR | Gemische | positive Bänder je Frame: Median (10 %…90 %) | mittleres Label y | Frames ohne Umwelt |
|---|---|---|---|---|
| −30…−20 dB | 215 | 2 (0…41) | 0,19 | 0,0 % |
| −20…−10 dB | 247 | 24 (1…60) | 0,43 | 0,0 % |
| −10…0 dB | 246 | 50 (10…64) | 0,67 | 0,0 % |
| 0…10 dB | 252 | 61 (43…64) | 0,87 | 0,0 % |
| 10…20 dB | 240 | 64 (59…64) | 0,96 | 0,1 % |

- **Alle Gemische**: Im Median sind 53 Bänder je Frame positiv; das mittlere Label beträgt 0,63.
- **Nur Umwelt**: 0 von 5 971 200 Band-Werten sind positiv.
- **Anteil positiver Bänder insgesamt**: Training 50,5 %, Test 53,2 %. Er steigt leicht von den
  tiefen zu den hohen Bändern (Training: 46 % in Band 0–15, 54 % in Band 48–63).

„Frames ohne Umwelt“ zählt Frames, in denen mindestens 60 Bänder ein SNR_b ≥ 70 dB haben, etwa
wegen digitaler Stille in der Umwelt-Spur.

Folgerung für AP4: Der Anteil positiver Bänder ist über alle Beispiele ausgewogen und nicht
„klein“, wie in Konzept 5.3 erwartet. Eine Gewichtung im Verlust ist daher nicht zwingend. Die
Auswertung sollte aber je Misch-SNR-Klasse erfolgen.

---

## 5. Entwurfsentscheidungen

Ein erster Lauf mit Misch-SNR −10…+30 dB und ohne Grundpegel ergab zu viele positive Bänder:
71 % insgesamt, im Median 51–64 von 64 Bändern je Frame. Das Werkzeug rechnet dabei richtig: Eine
unabhängige STFT in Python auf denselben Anteilen stimmt auf etwa 1 dB mit ihm überein. Die
Ursachen liegen im Entwurf des Datensatzes:

| Ursache | Befund | Maßnahme |
|---|---|---|
| Digitale Stille in ESC-50 | Die Clips sind auf 5 s aufgefüllt, im Mittel zu 10 % still. 12 % der Frames enthielten keine Umwelt, sodass fast alle Bänder > +70 dB lagen. | Umgebungs-Grundpegel (DDS-Hintergrund, −25…−10 dB) unter jede Umwelt-Spur legen; im Freien gibt es keine digitale Stille. |
| Umwelt spektral konzentriert | 90 % der Leistung liegen bei der Umwelt in 19 Bändern, bei den Drohnen in 32. Bei gleichem Breitband-SNR gewinnt die Drohne daher in den meisten Bändern. | Misch-SNR um 20 dB nach unten verschoben: −30…+20 dB |
| Harmonischen-Kennzahl | Bei 2–6 Rotoren und ±1 Band Toleranz decken die k·BPF-Bänder fast alle Bänder ab; die Kennzahl trennt daher nicht. | Ersetzt durch die Kennzahlen je Misch-SNR-Klasse (Abschnitt 4). |

---

## 6. Grenzen und offene Punkte

- Als Drohnen werden nur synthetische verwendet. Echte Drohnen mit hohem SNR für die Mischung
  fehlen (AP2, Inventur). Die echten DDS-Drohnen dienen nur dem Test.
- ESC-50 steht unter CC BY-NC 3.0, ist also nur für nichtkommerzielle Nutzung zulässig
  (`docs/Daten_ML124.md`).
- Die ersten Frames jedes Beispiels enthalten das Einschwingen von AGC, Rauschboden, AM-Historie
  und HBD (3 s). Das Training sollte etwa die erste Sekunde (31 Frames) verwerfen oder geringer
  gewichten.
- Es gibt keinen Raumhall und keine Mikrofon-/Gehäuseübertragung; beides kommt mit
  Board-Aufnahmen in AP8.
