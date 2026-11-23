"""
Prüft, ob im erzeugten Audio wirklich die Noten erklingen, die in den Labels
stehen. (Test für Synthesizer + Labels -- nicht Teil des Modells.)

Für jede gespielte Note aus labels.jsonl:
  - das Stück Audio dieser Note herausschneiden
  - Spektrum berechnen
  - für jeden Kandidaten-Ton (MIDI 45..100) die Energie an den ersten
    Obertönen aufsummieren -> der Ton mit der meisten Energie gewinnt
  - mit dem Label vergleichen

  python check_pitches.py data/showcase
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from ornaments import midi_to_hz, midi_to_name
from synth import read_wav

CANDIDATES = np.arange(45, 101)


def estimate_midi(segment: np.ndarray, sr: int, n_harm: int = 6) -> int:
    n_fft = 16384                                         # viel Zero-Padding
    spec = np.abs(np.fft.rfft(segment * np.hanning(len(segment)), n_fft))
    logspec = np.log(spec + 1e-6)
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    # Pro Oberton das Maximum in einem Band von +-40 Cent: das Instrument kann
    # verstimmt sein, und bei langen Noten sind die Spitzen sehr schmal.
    band = 2 ** (0.4 / 12)
    scores = []
    for m in CANDIDATES:
        f = midi_to_hz(m) * np.arange(1, n_harm + 1)
        f = f[f * band < sr / 2]
        lo = np.searchsorted(freqs, f / band)
        hi = np.searchsorted(freqs, f * band) + 1
        scores.append(np.mean([logspec[a:b].max() for a, b in zip(lo, hi)]))
    return int(CANDIDATES[int(np.argmax(scores))])


def check(dataset: Path) -> None:
    with open(dataset / "labels.jsonl") as f:
        records = [json.loads(line) for line in f]

    by_len = defaultdict(lambda: [0, 0])                  # Notenlänge -> [ok, total]
    errors = []
    for r in records:
        x, sr = read_wav(dataset / r["wav"])
        for n in r["played_notes"]:
            # nur die Mitte der Note (Ränder überlappen mit Nachbarnoten)
            margin = min(0.25 * n["duration"], 0.03)
            a = int((n["onset"] + margin) * sr)
            b = int((n["onset"] + n["duration"] - margin) * sr)
            est = estimate_midi(x[a:b], sr)
            bucket = f"{int(n['duration'] * 1000) // 20 * 20:>4d} ms"
            if n["duration"] >= 0.3:
                bucket = "lang"
            by_len[bucket][1] += 1
            if est == n["midi"]:
                by_len[bucket][0] += 1
            else:
                errors.append(f"{r['id']}  {r['tokens']:45s} Note {n['pitch']} "
                              f"({n['duration'] * 1000:.0f} ms) erkannt als {midi_to_name(est)}")

    total_ok = sum(v[0] for v in by_len.values())
    total = sum(v[1] for v in by_len.values())
    print(f"{dataset}: {total_ok}/{total} Noten korrekt ({100 * total_ok / total:.1f} %)")
    print("  nach Notenlänge:")
    for k in sorted(by_len, key=lambda k: (k == "lang", k)):
        ok, n = by_len[k]
        print(f"    {k:>8s}: {ok:4d}/{n:<4d} ({100 * ok / n:5.1f} %)")
    for e in errors[:15]:
        print("  FEHLER", e)
    if len(errors) > 15:
        print(f"  ... und {len(errors) - 15} weitere")


if __name__ == "__main__":
    for path in sys.argv[1:]:
        check(Path(path))
