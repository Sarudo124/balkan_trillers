# Balkan Trillers – synthetische Daten für Ornament-Erkennung

Ziel: ein Modell, das eine Akkordeon-Aufnahme hört und aufschreibt, welche Noten
und Ornamente (Triller, Mordent …) gespielt wurden.

## Konzept

Aus einem zufällig erzeugten „Notenblatt“ (einer Liste von Noten in Python)
entstehen **zwei Dinge**: die Frage und die richtige Antwort.

```
                         Notenblatt
                  G5 · Pause · G5 mit Triller · A#5
                   /                          \
        Tokens (Antwort)                  Audio (Frage)
  NOTE_G5 TIME_400 REST TIME_200 …        Akkordeon-Klang + Handy-Effekte
                   |                            |
        Token-IDs  [1, 39, 69, …]         Spektrogramm-Vektoren
                   \                          /
                    ───── Training: ─────────
               Modell hört das Audio und soll die Tokens schreiben
```

---

## Wie Noten als Tokens dargestellt werden

| Was | Tokens | Beispiel |
|---|---|---|
| normale Note | Tonhöhe + Dauer | `NOTE_G5 TIME_400` |
| Note mit Ornament | Ornament + Hauptnote + Abstand + Dauer | `ORN_TRILL_UD NOTE_G5 UP_1 DOWN_2 TIME_1550` |
| Pause | Pause + Dauer | `REST TIME_200` |

- `TIME_x` = Dauer in Millisekunden, auf 50 ms gerundet.
- `UP_x` / `DOWN_x` = Abstand der Hilfsnote nach oben / unten in Halbtönen.
- Eine Note mit Ornament hat **eine** Dauer für die ganze Note. Die einzelnen schnellen Triller-Töne stehen nicht in den Tokens.
- Jeder Satz beginnt mit `<bos>` und endet mit `<eos>`.

Beispiel:

```
<bos> NOTE_D5 TIME_200  REST TIME_350  ORN_PRALL NOTE_A#4 UP_2 TIME_650  NOTE_D5 TIME_450 <eos>
      └─ Note ──────┘   └─ Pause ───┘  └──── Note mit Pralltriller ───┘  └─ Note ──────┘
```

## Welche Ornamente es gibt

M = Hauptnote, U = Hilfsnote oben, L = Hilfsnote unten. Alle beginnen auf der Hauptnote.

| Ornament | Muster | Beispiel auf D5 | Abstände | Varianten |
|---|---|---|---|---|
| `PRALL` (Pralltriller) | M U M | D5 E5 D5 | oben 1, 2 oder 3 Halbtöne | 3 |
| `MORDENT` | M L M | D5 C5 D5 | unten 1 oder 2 | 2 |
| `TRILL` (Triller) | M U M U M | D5 E5 D5 E5 D5 | oben 1 oder 2 | 2 |
| `TRILL_UD` (Triller oben + unten) | M U M L M | D5 E5 D5 C#5 D5 | oben 1/2 × unten 1/2 | 4 |

→ 11 Varianten + „kein Ornament“

## Tokens → Zahlen

Jedes Token bekommt eine feste Nummer (ID) = seine Zeile in `vocab.txt` (ab 0 gezählt).
Das Modell rechnet nur mit diesen Nummern.

**Anzahl der Tokens: 142**

| Gruppe | Tokens | Anzahl |
|---|---|---|
| Steuerzeichen | `<pad>` `<bos>` `<eos>` | 3 |
| Pause | `REST` | 1 |
| Ornamente | `ORN_PRALL` `ORN_MORDENT` `ORN_TRILL` `ORN_TRILL_UD` | 4 |
| Tonhöhen | `NOTE_C3` … `NOTE_C7` (4 Oktaven) | 49 |
| Abstand oben | `UP_1` `UP_2` `UP_3` | 3 |
| Abstand unten | `DOWN_1` `DOWN_2` | 2 |
| Dauer | `TIME_50` … `TIME_4000` (alle 50 ms) | 80 |
| **Summe** | | **142** |

---

## Wie aus Audio Vektoren werden

```
Audio  ▕▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▕
        [── 85 ms ──]
          [── 85 ms ──]          alle 10 ms ein neuer Block
            [── 85 ms ──]
              …
                 ↓ pro Block: welche Frequenzen wie stark? (FFT)
                 ↓ auf 128 Frequenzbänder zusammenfassen
Vektor  [0.8, 0.7, 0.1, … 128 Zahlen]   ← ein Vektor pro 10 ms
```

- Pro 10 ms entsteht ein Vektor mit 128 Zahlen. Jede Zahl gibt an, wie laut gerade ein Frequenzband ist.
- 2 Sekunden Audio ergeben 200 Vektoren. Das ist die Eingabe für den Encoder.
- Die Anzahl der Vektoren (Audio) und die Anzahl der Tokens (Antwort) hängen nicht zusammen: z. B. 280 Vektoren rein, 11 Tokens raus.

**Konflikt Zeit- vs. Frequenzauflösung:**

| | langer Block | kurzer Block |
|---|---|---|
| Tonhöhe | genau, Halbtöne gut trennbar | ungenau, tiefe Halbtöne verschwimmen |
| Zeit | ungenau, schnelle Triller-Töne verschmieren | genau |

Wir nutzen 85 ms als Kompromiss: Halbtöne sind trennbar, aber Triller-Töne
(25–125 ms) werden etwas verschmiert. Einstellbar in `audio_to_vectors.py` (`N_FFT`, `HOP`).

---

## Erzeugte Daten

### Showcase (`data/showcase`) – zum Anhören und Vorführen

Jede Variante einmal auf D5, 90 BPM, 70 ms pro Triller-Ton, ohne Effekte.

| Datei | Ornament | Töne |
|---|---|---|
| `00000.wav` | `PRALL` oben 1 | D5 D#5 D5 |
| `00001.wav` | `PRALL` oben 2 | D5 E5 D5 |
| `00002.wav` | `PRALL` oben 3 | D5 F5 D5 |
| `00003.wav` | `MORDENT` unten 1 | D5 C#5 D5 |
| `00004.wav` | `MORDENT` unten 2 | D5 C5 D5 |
| `00005.wav` | `TRILL` oben 1 | D5 D#5 D5 D#5 D5 |
| `00006.wav` | `TRILL` oben 2 | D5 E5 D5 E5 D5 |
| `00007.wav` | `TRILL_UD` oben 1, unten 1 | D5 D#5 D5 C#5 D5 |
| `00008.wav` | `TRILL_UD` oben 1, unten 2 | D5 D#5 D5 C5 D5 |
| `00009.wav` | `TRILL_UD` oben 2, unten 1 | D5 E5 D5 C#5 D5 |
| `00010.wav` | `TRILL_UD` oben 2, unten 2 | D5 E5 D5 C5 D5 |

Spektrogramm-Bilder dazu: `data/showcase/features/000xx.png`

### Zufallsdaten (`data/random_test`, 300 Beispiele) – zum Trainieren

Jedes Beispiel: **normale Note · (Pause) · Note mit Ornament · (Pause) · normale Note**

- Ornament: eine der 11 Varianten (alle gleich oft), in 15 % der Fälle keins
- Pausen: je 30 % Wahrscheinlichkeit, 50–400 ms
- Hauptnote G3–E6, Notenlängen aus Tempo 70–160 BPM
- Triller-Töne 25–125 ms, jeder einzeln unterschiedlich lang
- Klang: Musette (2–3 leicht verstimmte Zungen) oder eine Zunge, Klangfarbe und Stimmung zufällig

Nicht im Repo (79 MB), aber exakt reproduzierbar, siehe Befehle unten.

### Augmentation – damit es wie eine Handy-Aufnahme klingt

| Effekt | Stärke | warum |
|---|---|---|
| Hochpass | 80–250 Hz | Handy-Mikros nehmen Bässe schlecht auf |
| Hall | 3 kurze Echos | Raumklang |
| Rauschen | 20–40 dB unter dem Signal | Hintergrundgeräusche |
| Lautstärke | 30–100 % | nah oder fern aufgenommen |

### Prüfung

`check_pitches.py` misst die Tonhöhe jedes Tons im Audio nach: Showcase 100 %, Zufallsdaten 99,8 % korrekt
(die wenigen Abweichungen sind sehr kurze, tiefe Töne, bei denen die Messung an ihre Grenze kommt).

---

## Befehle

Benötigt nur Python und `numpy`.

```bash
python make_synthetic_data.py showcase --out data/showcase                          # 11 Beispiele
python make_synthetic_data.py random --n 300 --context --out data/random_test       # Zufallsdaten
python audio_to_vectors.py data/showcase --png                                      # Vektoren (+ Bilder)
python check_pitches.py data/showcase                                               # Prüfung
```

| Datei | macht |
|---|---|
| `ornaments.py` | Ornamente definieren, Noten → Tokens |
| `synth.py` | Noten → Akkordeon-Audio, Augmentation |
| `make_synthetic_data.py` | Notenblätter auswürfeln, alles zusammen erzeugen |
| `audio_to_vectors.py` | Audio → Vektoren |
| `check_pitches.py` | Kontrolle: passt das Audio zu den Labels? |
