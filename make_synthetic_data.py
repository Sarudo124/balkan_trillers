"""
Erzeugt synthetische Trainingsdaten: immer EIN Ornament pro Beispiel.

Zwei Modi:
  showcase  -> je ein sauberes Beispiel für jedes Ornament x jeden Abstand
               (11 Beispiele, zum Anhören / Zeigen im Meeting)
               python make_synthetic_data.py showcase --out data/showcase

  random    -> viele zufällige Beispiele (Tonhöhe, Tempo, Geschwindigkeit des
               Ornaments, Klang, Augmentation) zum Trainieren
               python make_synthetic_data.py random --n 2000 --out data/train

Ausgabe:
  <out>/wav/00000.wav ...      das Audio
  <out>/labels.jsonl           pro Zeile ein Beispiel: Tokens + gespielte Noten
  <out>/vocab.txt              alle Tokens (Zeilennummer = Token-ID)
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from ornaments import (INTERVALS, ORNAMENTS, OrnamentedNote, Rest, build_vocab,
                       expand_ornament, midi_to_name, n_fast_notes, to_tokens)
from synth import augment, render, write_wav

SR = 24000          # MERT arbeitet auch mit 24 kHz -> später direkt vergleichbar


def score_to_notes(score: list[OrnamentedNote | Rest], start: float = 0.2):
    """Notenblatt -> gespielte Noten (mit Startzeiten). Pausen = nur Zeit vergeht."""
    played, t = [], start
    for n in score:
        if not isinstance(n, Rest):
            played += expand_ornament(n, t)
        t += n.duration
    return played


def save_example(i, out, score, bpm, rng, sound, do_augment, labels_file):
    played = score_to_notes(score)
    audio = render(played, SR, rng=rng, **sound)
    if do_augment:
        audio = augment(audio, SR, rng)
    wav_path = out / "wav" / f"{i:05d}.wav"
    write_wav(wav_path, audio, SR)

    record = {
        "id": f"{i:05d}",
        "wav": str(wav_path.relative_to(out)),
        "tokens": " ".join(to_tokens(score)),        # Ziel für den Decoder
        "bpm": bpm,
        "played_notes": [                                 # für Auswertung/Debug
            {"pitch": midi_to_name(p.midi), "midi": p.midi,
             "onset": round(p.onset, 4), "duration": round(p.duration, 4)}
            for p in played
        ],
        "sound": {k: list(v) if isinstance(v, tuple) else v for k, v in sound.items()},
        "augmented": do_augment,
    }
    labels_file.write(json.dumps(record) + "\n")
    return record


def showcase(out: Path, main_midi: int = 74, bpm: float = 90):
    """Jedes Ornament x jeder Abstand auf D5, 70 ms pro schneller Note."""
    rng = np.random.default_rng(0)
    beat = 60 / bpm
    sound = {"reeds_cents": (0.0, -12.0, 12.0)}
    i = 0
    with open(out / "labels.jsonl", "w") as f:
        for orn in ORNAMENTS:
            ups, downs = INTERVALS[orn]
            ups, downs = ups or (0,), downs or (0,)   # 0 = wird nicht benutzt
            for up, down in itertools.product(ups, downs):
                score = [OrnamentedNote(main_midi, 2 * beat, orn, up=up, down=down,
                                        orn_note_dur=0.07)]
                r = save_example(i, out, score, bpm, rng, sound, False, f)
                print(f"{r['id']}  {r['tokens']}")
                i += 1


def random_examples(out: Path, n: int, seed: int, context: bool, p_plain: float):
    rng = np.random.default_rng(seed)
    # alle Varianten (Ornament, oben, unten) gleich oft ziehen -- sonst wäre
    # z. B. jede der 4 TRILL_UD-Kombis viel seltener als jeder Mordent
    variants = [(orn, up, down)
                for orn, (ups, downs) in INTERVALS.items()
                for up in (ups or (0,)) for down in (downs or (0,))]
    with open(out / "labels.jsonl", "w") as f:
        for i in range(n):
            bpm = float(rng.integers(70, 160))
            sixteenth = 60 / bpm / 4

            # die ornamentierte Note
            orn, up, down = None, 0, 0
            if rng.random() >= p_plain:
                orn, up, down = variants[rng.integers(len(variants))]
            # schnelle Noten sind real NICHT gleich lang: Grundtempo 40-90 ms,
            # jede einzelne Note +-40 % (deckt auch Antemordent-artiges Timing ab)
            base = rng.uniform(0.04, 0.09)
            fast = ([float(base * rng.uniform(0.6, 1.4)) for _ in range(n_fast_notes(orn))]
                    if orn else None)
            note = OrnamentedNote(
                main_midi=int(rng.integers(55, 89)),          # G3 .. E6
                duration=int(rng.choice([4, 6, 8])) * sixteenth,
                ornament=orn,
                up=up,
                down=down,
                orn_note_durs=fast,
            )
            # Ornament muss in die Note passen
            # (die Ornament-Noten dürfen höchstens 80 % der Notendauer belegen)
            if orn:
                needed = sum(fast) / 0.8
                if needed > note.duration:   # auf ganze Sechzehntel aufrunden
                    note.duration = np.ceil(needed / sixteenth) * sixteenth

            score = [note]
            if context:  # unverzierte Nachbarnoten davor/danach
                before = OrnamentedNote(note.main_midi + int(rng.integers(-4, 5)),
                                        int(rng.choice([2, 4])) * sixteenth)
                after = OrnamentedNote(note.main_midi + int(rng.integers(-4, 5)),
                                       int(rng.choice([2, 4])) * sixteenth)
                # mit 30 % Wahrscheinlichkeit eine Pause (50-400 ms) zwischen zwei Noten
                pause = lambda: ([Rest(float(rng.uniform(0.05, 0.4)))]
                                 if rng.random() < 0.3 else [])
                score = [before] + pause() + [note] + pause() + [after]

            sound = {
                "reeds_cents": (0.0,) if rng.random() < 0.3
                else (0.0, -float(rng.uniform(5, 20)), float(rng.uniform(5, 20))),
                "brightness": float(rng.uniform(0.8, 1.5)),
                "tuning_cents": float(rng.uniform(-20, 20)),
            }
            save_example(i, out, score, bpm, rng, sound, True, f)
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{n}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["showcase", "random"])
    ap.add_argument("--out", default="data/showcase")
    ap.add_argument("--n", type=int, default=1000, help="Anzahl (nur random)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--context", action="store_true",
                    help="unverzierte Noten vor/nach dem Ornament (nur random)")
    ap.add_argument("--p-plain", type=float, default=0.15,
                    help="Anteil Beispiele ganz ohne Ornament (nur random)")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "wav").mkdir(parents=True, exist_ok=True)
    (out / "vocab.txt").write_text("\n".join(build_vocab()) + "\n")

    if args.mode == "showcase":
        showcase(out)
    else:
        random_examples(out, args.n, args.seed, args.context, args.p_plain)
    print(f"fertig -> {out}")
