"""
Definition der Ornamente + Umwandlung in Tokens.

Hier steht das "Musik-Wissen" des Projekts an EINER Stelle:
  1. Welche Ornamente gibt es? (ORNAMENTS)
  2. Wie wird aus einem Ornament eine Folge echt gespielter Noten? (expand_ornament)
  3. Wie wird daraus eine Token-Folge, die der Decoder lernen soll? (to_tokens)

Pitch-Konvention: MIDI-Nummern (60 = C4, 62 = D4, 69 = A4 = 440 Hz).
Ein Halbton = +1, ein Ganzton = +2.
"""

from dataclasses import dataclass

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_name(midi: int) -> str:
    """62 -> 'D4'"""
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def midi_to_hz(midi: float) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


# ---------------------------------------------------------------------------
# 1. Die Ornamente
# ---------------------------------------------------------------------------
# Jedes Ornament ist ein Muster aus Hilfsnoten relativ zur Hauptnote:
#   "M" = Hauptnote, "U" = obere Hilfsnote, "L" = untere Hilfsnote
# Nach dem Muster bleibt die Hauptnote liegen (Rest der Notendauer).
#
# Festgelegt mit Sandro (kroatisch-serbische Spielweise), Stand 23.11.2026.
# Alle Ornamente beginnen auf der HAUPTNOTE (wie im Bulgarian Accordion
# Tutorial, goldov.com/butterw -- nicht wie im Barock oben).
# Hier zu ändern reicht, alles andere (Daten, Tokens, Vokabular) passt sich an.
ORNAMENTS: dict[str, list[str]] = {
    # --- 3 Noten ---
    "PRALL":    ["M", "U", "M"],            # Pralltriller (oberer Mordent)
    "MORDENT":  ["M", "L", "M"],            # unterer Mordent
    # --- 5 Noten ---
    "TRILL":    ["M", "U", "M", "U", "M"],  # Triller = verlängerter Prall
    "TRILL_UD": ["M", "U", "M", "L", "M"],  # Triller oben UND unten
}

# erlaubte Abstände der Hilfsnoten in Halbtönen: (oben, unten)
# 1 = Halbton, 2 = Ganzton, 3 = kleine Terz
INTERVALS: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {
    "PRALL":    ((1, 2, 3), ()),
    "MORDENT":  ((), (1, 2)),
    "TRILL":    ((1, 2), ()),
    "TRILL_UD": ((1, 2), (1, 2)),     # 2 x 2 Kombinationen
}
assert INTERVALS.keys() == ORNAMENTS.keys(), "jedes Ornament braucht Intervalle"

# alle vorkommenden Abstände (fürs Vokabular)
UP_INTERVALS = tuple(sorted({i for up, _ in INTERVALS.values() for i in up}))
DOWN_INTERVALS = tuple(sorted({i for _, down in INTERVALS.values() for i in down}))


@dataclass
class Note:
    """Eine tatsächlich erklingende Note (das, was im Audio zu hören ist)."""
    midi: int
    onset: float      # Startzeit in Sekunden
    duration: float   # Dauer in Sekunden


@dataclass
class OrnamentedNote:
    """Eine Note aus dem Notenblatt -- optional mit Ornament."""
    main_midi: int
    duration: float              # Gesamtdauer der Note inkl. Ornament (s)
    ornament: str | None = None  # Schlüssel aus ORNAMENTS oder None
    up: int = 2                  # Abstand obere Hilfsnote (Halbtöne)
    down: int = 2                # Abstand untere Hilfsnote (Halbtöne)
    orn_note_dur: float = 0.06   # Dauer jeder schnellen Ornament-Note (s)
    # optional: eigene Dauer pro schneller Note (real sind sie NICHT gleich
    # lang, z. B. Antemordent: 1. Note sehr kurz, Hilfsnote länger)
    orn_note_durs: list[float] | None = None


def uses(ornament: str) -> tuple[bool, bool]:
    """Benutzt das Ornament eine obere / untere Hilfsnote?"""
    pattern = ORNAMENTS[ornament]
    return "U" in pattern, "L" in pattern


def n_fast_notes(ornament: str) -> int:
    """Anzahl schneller Noten = alle außer der letzten Hauptnote (die klingt aus)."""
    return len(ORNAMENTS[ornament]) - 1


# ---------------------------------------------------------------------------
# 2. Ornament -> echt gespielte Noten
# ---------------------------------------------------------------------------
def expand_ornament(n: OrnamentedNote, onset: float) -> list[Note]:
    """Macht aus einer (ornamentierten) Notenblatt-Note die gespielten Noten.

    Die LETZTE Hauptnote "M" im Muster bekommt die restliche Zeit, alle anderen
    Noten sind schnell. (Stünde danach noch eine Hilfsnote, wäre sie ein kurzer
    Vorschlag ganz am Ende der Note.)

    Beispiel PRALL auf D5 (74), up=1, Dauer 0.5 s:
        D5 60ms, D#5 60ms, D5 380ms
    Beispiel TRILL_UD auf D5, up=2, down=1, Dauer 0.5 s:
        D5 60ms, E5 60ms, D5 60ms, C#5 60ms, D5 260ms
    """
    if n.ornament is None:
        return [Note(n.main_midi, onset, n.duration)]

    offset = {"M": 0, "U": +n.up, "L": -n.down}
    pattern = ORNAMENTS[n.ornament]
    sustain = max(i for i, s in enumerate(pattern) if s == "M")
    fast = list(n.orn_note_durs or [n.orn_note_dur] * (len(pattern) - 1))
    assert len(fast) == len(pattern) - 1, "eine Dauer pro schneller Note"

    durs = fast[:sustain] + [n.duration - sum(fast)] + fast[sustain:]
    notes, t = [], onset
    for symbol, dur in zip(pattern, durs):
        notes.append(Note(n.main_midi + offset[symbol], t, dur))
        t += dur
    return notes


# ---------------------------------------------------------------------------
# 3. Tokens (Antwort auf "How do we represent each note as a token?")
# ---------------------------------------------------------------------------
# Idee: Jede Eigenschaft einer Note ist EIN Token (wie ein Wort im Satz).
#   normale Note:       NOTE_D5 TIME_400
#   Note mit Ornament:  ORN_PRALL NOTE_D5 UP_1 TIME_400
#
# TIME_x = Dauer der ganzen Note in Millisekunden, gerundet auf 50 ms.
# Bei einer Note mit Ornament zählt die GESAMTE Note (alle Ornament-Töne +
# ausgehaltene Hauptnote) -- die einzelnen schnellen Töne stehen nicht drin.
# Bewusst echte Zeit statt Notenwerten (Viertel, Achtel ...): Notenwerte
# setzen ein bekanntes Tempo voraus, das man einer kurzen Aufnahme nicht
# anhören kann (Viertel bei 80 BPM = Halbe bei 160 BPM = 0,75 s).
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>"]
TIME_STEP_MS = 50
TIME_MAX_MS = 4000


def time_token(seconds: float) -> str:
    """0.385 s -> 'TIME_400'  (auf 50 ms gerundet, 50 ... 4000)"""
    ms = round(seconds * 1000 / TIME_STEP_MS) * TIME_STEP_MS
    ms = min(max(ms, TIME_STEP_MS), TIME_MAX_MS)
    return f"TIME_{ms}"


def to_tokens(score: list[OrnamentedNote]) -> list[str]:
    tokens = ["<bos>"]
    for n in score:
        if n.ornament is not None:
            has_up, has_down = uses(n.ornament)
            tokens.append(f"ORN_{n.ornament}")
            tokens.append(f"NOTE_{midi_to_name(n.main_midi)}")
            if has_up:
                tokens.append(f"UP_{n.up}")
            if has_down:
                tokens.append(f"DOWN_{n.down}")
        else:
            tokens.append(f"NOTE_{midi_to_name(n.main_midi)}")
        tokens.append(time_token(n.duration))
    tokens.append("<eos>")
    return tokens


def build_vocab(midi_range=range(48, 97)) -> list[str]:
    """Alle möglichen Tokens -> Token-ID = Position in dieser Liste."""
    vocab = list(SPECIAL_TOKENS)
    vocab += [f"ORN_{o}" for o in ORNAMENTS]
    vocab += [f"NOTE_{midi_to_name(m)}" for m in midi_range]
    vocab += [f"UP_{i}" for i in UP_INTERVALS]
    vocab += [f"DOWN_{i}" for i in DOWN_INTERVALS]
    vocab += [f"TIME_{ms}" for ms in range(TIME_STEP_MS, TIME_MAX_MS + 1, TIME_STEP_MS)]
    return vocab


if __name__ == "__main__":
    # kleine Demo:  python ornaments.py
    score = [
        OrnamentedNote(72, 0.3),                                 # C5 plain
        OrnamentedNote(74, 0.6, "PRALL", up=1),                  # D5 Prall
        OrnamentedNote(76, 0.6, "MORDENT", down=2),              # E5 Mordent
        OrnamentedNote(77, 0.6, "PRALL", up=3),                  # F5 Prall, kl. Terz
        OrnamentedNote(79, 0.6, "TRILL", up=2,                   # G5 Triller mit
                       orn_note_durs=[0.03, 0.07, 0.05, 0.06]),  # ungleichen Noten
        OrnamentedNote(81, 0.6, "TRILL_UD", up=2, down=1),       # A5 oben+unten
    ]
    t = 0.0
    for n in score:
        played = expand_ornament(n, t)
        print(f"{n.ornament or 'plain':10s} {midi_to_name(n.main_midi):4s} ->",
              ", ".join(f"{midi_to_name(p.midi)}({p.duration*1000:.0f}ms)" for p in played))
        t += n.duration
    print("Tokens:", " ".join(to_tokens(score)))
    print("Vokabulargröße:", len(build_vocab()))
