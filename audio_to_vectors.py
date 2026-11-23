"""
Audio -> Spektrogramm -> Vektoren   (das ist der Input für den Encoder)

Schritte:
  1. Audio      : 1D-Array mit 24 000 Zahlen pro Sekunde (Luftdruck)
  2. STFT       : Audio in kurze, überlappende Fenster schneiden; pro Fenster
                  per FFT ausrechnen, wie stark jede Frequenz vorkommt
  3. Mel-Skala  : die ~1000 FFT-Frequenzen auf 128 "gehörgerechte" Bänder
                  zusammenfassen (unten fein, oben grob -- wie das Ohr)
  4. Log        : Lautstärke logarithmisch (dB) -- wie das Ohr
  5. Vektoren   : jede Spalte (= 10 ms) ist ein Vektor mit 128 Zahlen.
                  Optional k Spalten zusammenkleben -> kürzere Sequenz.

Danach (im Modell, PyTorch):  nn.Linear(k*128, d_model) macht aus jedem
Vektor ein Embedding -- genau wie die Token-Embeddings bei Text.

Benutzung:
  python audio_to_vectors.py data/showcase              # ganzer Datensatz
  python audio_to_vectors.py data/showcase/wav/00000.wav --png
"""

import argparse
import json
import struct
import zlib
from pathlib import Path

import numpy as np

from synth import read_wav


# --------------------------------------------------------------- Parameter
SR = 24000
N_FFT = 2048        # Fenster 85 ms  -> Frequenzauflösung 11.7 Hz
HOP = 240           # Schritt 10 ms  -> 100 Vektoren pro Sekunde
N_MELS = 128
FMIN, FMAX = 50.0, 8000.0
# Zielkonflikt: längeres Fenster = Halbtöne besser trennbar (tiefe Noten liegen
# nur ~10 Hz auseinander), aber kurze Ornament-Noten (40-100 ms) verschmieren.
# -> Ausprobieren! N_FFT=1024 ist schärfer in der Zeit, unschärfer in Tonhöhe.


# --------------------------------------------------------------- 2. STFT
def stft_power(x: np.ndarray, n_fft: int = N_FFT, hop: int = HOP) -> np.ndarray:
    """Gibt (Frequenzen, Zeit) zurück: Leistung |FFT|^2 pro Fenster."""
    x = np.pad(x, n_fft // 2)                              # zentrierte Fenster
    n_frames = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = x[idx] * np.hanning(n_fft)[None, :]           # (Zeit, n_fft)
    spec = np.fft.rfft(frames, axis=1)                     # (Zeit, n_fft/2+1)
    return (np.abs(spec) ** 2).T                           # (Freq, Zeit)


# --------------------------------------------------------------- 3. Mel
def hz_to_mel(f):
    return 2595 * np.log10(1 + np.asarray(f) / 700)


def mel_to_hz(m):
    return 700 * (10 ** (np.asarray(m) / 2595) - 1)


def mel_filterbank(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX):
    """Matrix (n_mels, n_fft/2+1): jede Zeile ist ein Dreiecksfilter."""
    fft_freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    edges = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    fb = np.zeros((n_mels, len(fft_freqs)))
    for m in range(n_mels):
        left, center, right = edges[m], edges[m + 1], edges[m + 2]
        rising = (fft_freqs - left) / (center - left)
        falling = (right - fft_freqs) / (right - center)
        fb[m] = np.maximum(0, np.minimum(rising, falling))
    return fb


# --------------------------------------------------------------- 1-4 zusammen
def log_mel(x: np.ndarray, sr: int) -> np.ndarray:
    if sr != SR:
        raise ValueError(f"Erwarte {SR} Hz, bekommen {sr} Hz (bitte resamplen)")
    mel = mel_filterbank() @ stft_power(x)                 # (n_mels, Zeit)
    logmel = np.log10(np.maximum(mel, 1e-10))
    logmel = np.maximum(logmel, logmel.max() - 8.0)        # max. 80 dB Dynamik
    return ((logmel + 4.0) / 4.0).astype(np.float32)       # grob auf [-1, 1]


# --------------------------------------------------------------- 5. Vektoren
def to_vectors(logmel: np.ndarray, stack: int = 1) -> np.ndarray:
    """(n_mels, Zeit) -> (Zeit/stack, stack*n_mels): eine Zeile = ein Vektor.

    stack=1: 100 Vektoren/s à 128 Zahlen
    stack=4:  25 Vektoren/s à 512 Zahlen (kürzere Sequenz -> schnellere
              Attention, aber 40 ms pro Vektor -- Ornament-Noten sind ~60 ms!)
    """
    frames = logmel.T                                      # (Zeit, n_mels)
    T = (len(frames) // stack) * stack
    return frames[:T].reshape(T // stack, stack * frames.shape[1])


# --------------------------------------------------------------- Bild (PNG)
def save_png(path, logmel: np.ndarray, scale: int = 3) -> None:
    """Spektrogramm als Bild (tiefe Frequenzen unten). Nur Standardbibliothek."""
    img = np.flipud(logmel)
    img = (img - img.min()) / (img.max() - img.min() + 1e-9)
    # einfache Farbskala: schwarz -> lila -> orange -> gelb
    stops = np.array([[0, 0, 0], [80, 20, 120], [230, 90, 40], [255, 240, 150]])
    pos = img * (len(stops) - 1)
    lo = np.floor(pos).astype(int).clip(0, len(stops) - 2)
    frac = (pos - lo)[..., None]
    rgb = (stops[lo] * (1 - frac) + stops[lo + 1] * frac).astype(np.uint8)
    rgb = rgb.repeat(scale, axis=0).repeat(scale, axis=1)
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c))

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
    Path(path).write_bytes(png)


# --------------------------------------------------------------- main
def process_file(wav: Path, out_dir: Path, stack: int, png: bool) -> np.ndarray:
    x, sr = read_wav(wav)
    lm = log_mel(x, sr)
    vec = to_vectors(lm, stack)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{wav.stem}.npy", vec)
    if png:
        save_png(out_dir / f"{wav.stem}.png", lm)
    return vec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="WAV-Datei oder Datensatz-Ordner (mit labels.jsonl)")
    ap.add_argument("--stack", type=int, default=1, help="Frames pro Vektor")
    ap.add_argument("--png", action="store_true", help="Spektrogramm-Bilder speichern")
    args = ap.parse_args()

    path = Path(args.path)
    if path.suffix == ".wav":
        vec = process_file(path, path.parent.parent / "features", args.stack, True)
        print(f"{path.name}: {vec.shape[0]} Vektoren à {vec.shape[1]} Zahlen")
    else:
        out_dir = path / "features"
        with open(path / "labels.jsonl") as f:
            records = [json.loads(line) for line in f]
        for r in records:
            vec = process_file(path / r["wav"], out_dir, args.stack, args.png)
            r["features"] = f"features/{r['id']}.npy"
            r["n_vectors"] = int(vec.shape[0])
        with open(path / "dataset.jsonl", "w") as f:   # labels + Feature-Pfad
            f.writelines(json.dumps(r) + "\n" for r in records)
        lens = [r["n_vectors"] for r in records]
        print(f"{len(records)} Dateien -> {out_dir}  "
              f"(Vektoren pro Beispiel: min {min(lens)}, max {max(lens)}, "
              f"Dimension {vec.shape[1]})")
