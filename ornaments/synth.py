"""
Ganz einfacher Akkordeon-Synthesizer (nur numpy).

Warum selbst bauen statt FluidSynth/SoundFont?
  - keine Installation nötig, läuft überall (auch auf dem Cluster)
  - wir kontrollieren jeden Parameter -> perfekte Labels
  - später kann man ihn durch einen echten Synthesizer ersetzen, das Format
    (Liste von Note-Objekten rein, Audio raus) bleibt gleich.

Akkordeon-Klang, stark vereinfacht:
  - Zungen (reeds) erzeugen einen obertonreichen, fast sägezahnartigen Ton
  - "Musette": 2-3 Zungen pro Ton, leicht gegeneinander verstimmt -> Schwebung
  - Balg: der Luftstrom läuft weiter, wenn man Knöpfe wechselt -> legato
"""

import wave
import numpy as np

from ornaments import Note, midi_to_hz


def render(
    notes: list[Note],
    sr: int = 24000,
    reeds_cents: tuple[float, ...] = (0.0, -12.0, +12.0),  # Musette; (0.0,) = trocken
    brightness: float = 1.1,      # größer = dumpfer (Obertöne fallen schneller ab)
    tuning_cents: float = 0.0,    # Verstimmung des ganzen Instruments
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    total = max(n.onset + n.duration for n in notes) + 0.3   # + Ausklingen
    out = np.zeros(int(total * sr), dtype=np.float64)

    attack, release = 0.008, 0.015  # sehr kurz -> schnelle Ornamente bleiben klar

    for n in notes:
        start = int(n.onset * sr)
        length = int((n.duration + release) * sr)  # leicht überlappen = legato
        t = np.arange(length) / sr

        f0 = midi_to_hz(n.midi + tuning_cents / 100)
        n_harm = int((sr / 2 * 0.9) // f0)           # nur Obertöne unter Nyquist
        k = np.arange(1, n_harm + 1)[:, None]        # (Harmonische, 1)
        amp = 1.0 / k ** brightness                  # Obertonspektrum

        tone = np.zeros(length)
        for cents in reeds_cents:
            f = f0 * 2 ** (cents / 1200)
            phase = rng.uniform(0, 2 * np.pi, size=(n_harm, 1))
            tone += (amp * np.sin(2 * np.pi * f * k * t + phase)).sum(axis=0)

        env = np.ones(length)
        a, r = int(attack * sr), int(release * sr)
        env[:a] = np.linspace(0, 1, a)
        env[-r:] = np.linspace(1, 0, r)

        end = min(start + length, len(out))
        out[start:end] += (tone * env)[: end - start]

    # Balg-Hüllkurve über die ganze Phrase + leichtes Zittern
    t_all = np.arange(len(out)) / sr
    bellows = 1 + 0.03 * np.sin(2 * np.pi * rng.uniform(3, 6) * t_all)
    out *= bellows
    out /= np.max(np.abs(out)) + 1e-9
    return (0.8 * out).astype(np.float32)


# ---------------------------------------------------------------------------
# Augmentation: Audio soll ein bisschen wie eine Handy-Aufnahme klingen
# ---------------------------------------------------------------------------
def augment(x: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    x = x.astype(np.float64)

    # 1. Handy-Mikro: tiefe Frequenzen weg (einfacher Hochpass, 1. Ordnung)
    cutoff = rng.uniform(80, 250)
    alpha = 1 / (1 + 2 * np.pi * cutoff / sr)
    y = np.zeros_like(x)
    for i in range(1, len(x)):                   # y[i] = a*(y[i-1] + x[i] - x[i-1])
        y[i] = alpha * (y[i - 1] + x[i] - x[i - 1])
    x = y

    # 2. etwas Hall: ein paar abklingende Echos
    for delay_ms, gain in [(23, 0.25), (41, 0.15), (67, 0.08)]:
        d = int(delay_ms / 1000 * sr)
        x[d:] += rng.uniform(0.5, 1.0) * gain * x[:-d]

    # 3. Rauschen mit zufälligem Signal-Rausch-Abstand (20-40 dB)
    snr_db = rng.uniform(20, 40)
    noise_power = np.mean(x ** 2) / 10 ** (snr_db / 10)
    x += rng.normal(0, np.sqrt(noise_power), size=len(x))

    # 4. zufällige Lautstärke
    x *= rng.uniform(0.3, 1.0) / (np.max(np.abs(x)) + 1e-9)
    return x.astype(np.float32)


# ---------------------------------------------------------------------------
# WAV lesen / schreiben (Standardbibliothek, 16 bit mono)
# ---------------------------------------------------------------------------
def write_wav(path, x: np.ndarray, sr: int) -> None:
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(pcm.tobytes())


def read_wav(path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as f:
        sr, ch, width = f.getframerate(), f.getnchannels(), f.getsampwidth()
        data = f.readframes(f.getnframes())
    if width != 2:
        raise ValueError(f"{path}: nur 16-bit WAV unterstützt")
    x = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768
    if ch > 1:                                   # Stereo -> Mono
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr
