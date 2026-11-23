# Ornament-Erkennung: synthetische Daten und Audio-Features

Dieser Ordner erzeugt Trainingsdaten für ein Sequence-to-Sequence-Modell, das
Ornamente (Pralltriller, Mordent, Triller) in Akkordeon-Aufnahmen erkennt.
Jedes Beispiel besteht aus

- **Eingabe:** Audio, umgerechnet in eine Folge von Spektrogramm-Vektoren (für den Encoder)
- **Ziel:** eine Token-Folge, die die gespielten Noten und Ornamente beschreibt (für den Decoder)

Beides wird aus demselben synthetischen Notenblatt erzeugt; die Labels sind
damit exakt bekannt.

---

## 1. Setup

Benötigt nur Python ≥ 3.10 und `numpy` (im Cluster-venv aus `LLM_Practical_KIT` bereits enthalten).
Keine weiteren Bibliotheken; WAV- und PNG-Dateien werden mit der Standardbibliothek geschrieben.

```bash
cd ornaments
python ornaments.py                                              # Demo: Ornament -> Noten -> Tokens
python make_synthetic_data.py showcase --out data/showcase       # je 1 Beispiel pro Ornament-Variante
python make_synthetic_data.py random --n 2000 --context --out data/train   # Trainingsdaten
python audio_to_vectors.py data/train                            # Spektrogramm-Vektoren (+ --png für Bilder)
python check_pitches.py data/train                               # Test: passen Audio und Labels zusammen?
```

| Datei | Inhalt |
|---|---|
| `ornaments.py` | Definition der Ornamente, Umrechnung Ornament → gespielte Noten, Tokenisierung, Vokabular |
| `synth.py` | Akkordeon-Synthesizer (additiv, numpy), Augmentation, WAV lesen/schreiben |
| `make_synthetic_data.py` | erzeugt Datensätze (`showcase` / `random`) |
| `audio_to_vectors.py` | WAV → Log-Mel-Spektrogramm → Vektorfolge (`.npy`) |
| `check_pitches.py` | misst die Tonhöhe jeder Note im Audio nach und vergleicht mit den Labels |

---

## 2. Pipeline

```
                         Notenblatt (zufällig erzeugt)
                         z. B. G5 · G5 mit TRILL_UD · A#5
                                       │
               ┌───────────────────────┴───────────────────────┐
               ▼                                               ▼
   to_tokens()                                     expand_ornament()
   ZIEL (Decoder)                                  gespielte Einzeltöne
   <bos> NOTE_G5 TIME_400 ORN_TRILL_UD ...         G5 G#5 G5 F5 G5 ... (mit Start/Dauer)
               │                                               │
               ▼                                               ▼
   vocab.txt → Token-IDs                           render() + augment()   (synth.py)
   [1, 38, 68, 6, ...]                             WAV, 24 kHz mono
                                                               │
                                                               ▼
                                                   log_mel() + to_vectors()
                                                   EINGABE (Encoder)
                                                   Matrix (Frames × 128), 1 Frame = 10 ms
```

### Audio-Synthese (`synth.py`)

- Jeder Ton ist eine Summe von Sinus-Obertönen (Amplitude ~ 1/k^b, b = Klangfarbe), bis zur Nyquist-Grenze.
- Musette: 1–3 Zungen pro Ton, gegeneinander um 5–20 Cent verstimmt (Schwebung).
- Hüllkurve 8 ms Attack / 15 ms Release, Töne überlappen leicht (legato), leichtes Balg-Tremolo.
- Augmentation (nur `random`): Hochpass 80–250 Hz, drei kurze Echos (Raumklang), Rauschen mit 20–40 dB SNR, zufällige Lautstärke.
- Abtastrate 24 kHz (gleich wie MERT, damit beide Encoder auf denselben Daten verglichen werden können).

### Audio → Vektoren (`audio_to_vectors.py`)

| Schritt | Parameter | Ergebnis |
|---|---|---|
| STFT (Hann-Fenster) | `N_FFT = 2048` (85 ms), `HOP = 240` (10 ms) | 1025 Frequenzbins × 100 Frames/s |
| Mel-Filterbank | 128 Bänder, 50–8000 Hz | 128 × Frames |
| Log + Normierung | log10, max. 80 dB Dynamik, `(x + 4) / 4` (wie Whisper) | Werte ca. in [−1, 1] |
| Transponieren | optional `--stack k` Frames zusammenfassen | **(Frames, 128)** als `.npy` |

Im Modell wird jeder 128-dim Frame-Vektor mit `nn.Linear(128, d_model)` auf die
Modellbreite projiziert (Gegenstück zu `nn.Embedding` auf der Textseite).

**Bekannter Zielkonflikt:** Das 85-ms-Fenster trennt Halbtöne gut, verschmiert
aber Ornament-Töne von 25–60 ms. Kleineres `N_FFT` (z. B. 1024) ist zeitlich
schärfer, frequenzseitig gröber. Alternative: Constant-Q-Spektrogramm (Bins pro Halbton).

---

## 3. Token-Repräsentation

Jede Eigenschaft einer Note ist ein eigenes Token. Eine Note mit Ornament ist
**ein** Ereignis mit **einer** Dauer; die einzelnen schnellen Ornament-Töne
erscheinen nicht in den Tokens.

```
normale Note:        NOTE_<Tonhöhe> TIME_<ms>
Note mit Ornament:   ORN_<Typ> NOTE_<Hauptnote> [UP_<n>] [DOWN_<n>] TIME_<ms>
```

| Token | Bedeutung |
|---|---|
| `<bos>`, `<eos>`, `<pad>` | Anfang, Ende, Auffüllen |
| `ORN_PRALL`, `ORN_MORDENT`, `ORN_TRILL`, `ORN_TRILL_UD` | Ornament-Typ (steht vor der Note) |
| `NOTE_C3` … `NOTE_C7` | Tonhöhe (bei Ornamenten: die Hauptnote) |
| `UP_1`, `UP_2`, `UP_3` | Abstand der oberen Hilfsnote in Halbtönen (nur wenn das Ornament sie nutzt) |
| `DOWN_1`, `DOWN_2` | Abstand der unteren Hilfsnote in Halbtönen (nur wenn das Ornament sie nutzt) |
| `TIME_50` … `TIME_4000` | Dauer der gesamten Note in ms, auf 50 ms gerundet |

Vokabular: **141 Tokens** (3 Spezial + 4 Ornamente + 49 Tonhöhen + 3 UP + 2 DOWN + 80 TIME),
gespeichert in `vocab.txt` (Zeilennummer = Token-ID). Kein BPE nötig.

### Ornamente

Festgelegt für kroatisch-serbische Spielweise (Abgleich mit dem
[Bulgarian Accordion Tutorial](https://goldov.com/butterw/tutorial/tutorial.old0/definitions.html)
und eigener Spielpraxis). M = Hauptnote, U/L = obere/untere Hilfsnote.

| Token | Muster | Töne | obere Hilfsnote | untere Hilfsnote | Varianten |
|---|---|---|---|---|---|
| `PRALL` | M U M | 3 | 1, 2 oder 3 Halbtöne | – | 3 |
| `MORDENT` | M L M | 3 | – | 1 oder 2 Halbtöne | 2 |
| `TRILL` | M U M U M | 5 | 1 oder 2 | – | 2 |
| `TRILL_UD` | M U M L M | 5 | 1 oder 2 | 1 oder 2 | 4 |

→ 11 Ornament-Varianten + „kein Ornament“. Alle Ornamente beginnen auf der Hauptnote;
die letzte Hauptnote des Musters klingt für den Rest der Notendauer weiter.
Nicht enthalten: Prall+ / Mordent+ (in dieser Stilistik selten), Doppelschläge.
Änderungen nur in `ORNAMENTS` und `INTERVALS` in `ornaments.py`.

### Beispiel (`data/random_test`, Beispiel 00000)

```
<bos> NOTE_G5 TIME_400  ORN_TRILL_UD NOTE_G5 UP_1 DOWN_2 TIME_1550  NOTE_A#5 TIME_400 <eos>
```

Tatsächlich erklingende Töne (aus `labels.jsonl` → `played_notes`):

```
G5 385 ms │ G5 96 · G#5 56 · G5 114 · F5 100 · G5 1172 ms │ A#5 385 ms
          └──────────────── eine Note, TIME_1550 ─────────┘
```

### Designentscheidungen

- **Dauer in Millisekunden statt Notenwerten.** Notenwerte (Viertel, Achtel) setzen ein bekanntes
  Tempo voraus. Aus kurzen Aufnahmen lässt es sich nicht bestimmen (Viertel bei 80 BPM = Halbe bei
  160 BPM = 0,75 s), das ergäbe widersprüchliche Labels für identisches Audio.
- **Eine Dauer pro Note, auch bei Ornamenten.** Die Längen der einzelnen Ornament-Töne schwanken
  von Spieler zu Spieler; Ziel ist der Ornament-Typ, nicht die Mikro-Rhythmik. Die exakten
  Einzeltöne bleiben in `played_notes` für Auswertungen erhalten.
- **Getrennte Tokens statt eines Tokens pro Note.** Hält das Vokabular klein (141 statt
  Tausender Kombinationen) und erlaubt eine getrennte Auswertung von Typ, Tonhöhe und Abstand.

---

## 4. Bisher erzeugte Daten

| Datensatz | Beispiele | Inhalt | Zweck |
|---|---|---|---|
| `data/showcase` | 22 | jede der 11 Varianten auf D5, je mit 70 ms und 180 ms pro Ornament-Ton; 90 BPM, Note 1,35 s, Musette, keine Augmentation | Anhören, Vorführen, Sichtprüfung |
| `data/random_test` | 300 | je 3 Noten (normal · ornamentiert · normal), 6,5 min Audio, 128–393 Frames pro Beispiel, 8–11 Tokens; alle Zufallsparameter + Augmentation | Testlauf der Pipeline, kein finaler Trainingssatz |

Im Repo ist nur `data/showcase` enthalten. `data/random_test` (79 MB) ist über den festen Seed
exakt reproduzierbar:

```bash
python make_synthetic_data.py random --n 300 --context --out data/random_test
python audio_to_vectors.py data/random_test
```

Zufallsparameter im `random`-Modus:

- Hauptnote G3–E6, Tempo 70–160 BPM (nur zur Erzeugung plausibler Notenlängen)
- Ornament-Töne je 25–125 ms, einzeln variiert (±40 %)
- 11 Varianten gleichverteilt, ca. 15 % Beispiele ohne Ornament (`--p-plain`)
- Klang: Musette oder eine Zunge, Klangfarbe, Gesamtverstimmung ±20 Cent
- mit `--context`: unverzierte Nachbarnoten vor und nach der Ornament-Note (±4 Halbtöne)

Dateien pro Datensatz:

```
wav/00000.wav          Audio, 24 kHz, mono, 16 bit
labels.jsonl           pro Zeile: id, wav, tokens, bpm, played_notes (Tonhöhe, Start, Dauer), Klangparameter
vocab.txt              Vokabular (Zeilennummer = Token-ID)
features/00000.npy     float32-Matrix (Frames, 128)
features/00000.png     Spektrogramm-Bild (nur mit --png)
dataset.jsonl          labels.jsonl + Pfad zur .npy + Anzahl Frames → Eingabe für das Training
```

### Prüfung der Labels (`check_pitches.py`)

Die Tonhöhe jeder gespielten Note wird im Audio per Obertonsummen-Spektrum geschätzt und mit dem Label verglichen:

| Datensatz | korrekt |
|---|---|
| `showcase` | 90 / 90 Töne (100 %) |
| `random_test` | 1697 / 1702 Töne (99,7 %) |

Die Abweichungen betreffen ausschließlich Töne ≤ 50 ms in tiefer Lage (G3–A3), wo die
Frequenzauflösung der Prüfung nicht ausreicht.

---

## 5. Offene Punkte

- Echte Aufnahmen: reale Ornament-Geschwindigkeiten messen, Testset aufnehmen.
- Pausen kommen in den Daten noch nicht vor (alles legato); für echte Aufnahmen ggf. `REST`-Token.
- Begleitung der linken Hand (Bass/Akkorde) fehlt im Synthesizer.
- Spektrogramm-Parameter (`N_FFT`, Mel vs. Constant-Q) für kurze Ornament-Töne optimieren.
- Vergleich eigener Encoder vs. MERT (beide erhalten dieselben 24-kHz-Daten).
- Nächster Schritt: Seq2Seq-Modell (Encoder + Decoder mit Cross-Attention) auf Basis von `LLM_Practical_KIT/model.py`.
