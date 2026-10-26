"""Parse the simai (maimai) chart in maidata.txt into timed note events.

The chart is only used as a timing map: every tap, hold, slide and break
becomes a cue for an attack. Times are in seconds from the start of the audio.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from pathlib import Path


@dataclass
class Slide:
    start_key: int
    segments: list[tuple[str, int]]   # (shape, end key)
    t_star: float                     # star tap time
    t_start: float                    # star begins to move (one beat later)
    duration: float
    brk: bool = False

    @property
    def t_end(self) -> float:
        return self.t_start + self.duration

    @property
    def end_key(self) -> int:
        return self.segments[-1][1]


@dataclass
class Note:
    t: float
    kind: str            # "tap" | "hold" | "slide"
    key: int
    brk: bool = False
    dur: float = 0.0     # hold length
    slides: list[Slide] = field(default_factory=list)
    group: int = 0       # notes sharing a group are simultaneous ("each")
    group_size: int = 1


def _bracket_seconds(text: str, bpm: float) -> float:
    text = text.strip("[]")
    if text.startswith("#"):
        return float(text[1:])
    if "#" in text:
        left, right = text.split("#", 1)
        if right.startswith("#"):
            return float(right[1:])
        bpm = float(left)
        text = right
    a, b = text.split(":")
    return 240.0 / bpm / float(a) * float(b)


SHAPES = ("pp", "qq", "-", "^", "<", ">", "v", "p", "q", "s", "z", "w")


def _parse_note(token: str, t: float, bpm: float) -> Note | None:
    token = token.strip()
    if not token or not token[0].isdigit():
        return None
    key = int(token[0])
    rest = token[1:]
    # modifiers directly after the key
    mods = re.match(r"[bx$!?@]*", rest).group(0)
    brk = "b" in mods
    rest = rest[len(mods):]
    if rest.startswith("h"):
        dur = 0.0
        m = re.match(r"h(\[[^\]]*\])?", rest)
        if m.group(1):
            dur = _bracket_seconds(m.group(1), bpm)
        tail = rest[m.end():]
        brk = brk or "b" in tail
        if dur <= 1e-6:
            return Note(t, "tap", key, brk)
        return Note(t, "hold", key, brk, dur)
    if not rest:
        return Note(t, "tap", key, brk)
    # slide: one path (this chart never uses '*')
    segments: list[tuple[str, int]] = []
    duration = 0.0
    slide_brk = False
    pos = 0
    while pos < len(rest):
        shape = next((s for s in SHAPES if rest.startswith(s, pos)), None)
        if shape is None:
            if rest[pos] == "b":
                slide_brk = True
                pos += 1
                continue
            if rest[pos] == "[":
                end = rest.index("]", pos)
                duration += _bracket_seconds(rest[pos:end + 1], bpm)
                pos = end + 1
                continue
            pos += 1
            continue
        pos += len(shape)
        end_key = int(rest[pos])
        pos += 1
        segments.append((shape, end_key))
    beat = 60.0 / bpm
    slide = Slide(key, segments, t, t + beat, duration, slide_brk or brk)
    return Note(t, "slide", key, brk, slides=[slide])


def parse_chart(path: Path, offset: float = 0.0) -> tuple[list[Note], float]:
    text = Path(path).read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if not line.startswith("&")]
    body = "".join(lines)
    body = body[:body.rindex("E")] if "E" in body else body
    bpm, div, t = 120.0, 4, offset
    notes: list[Note] = []
    group = 0
    token = ""
    pos = 0
    while pos < len(body):
        c = body[pos]
        if c == "(":
            end = body.index(")", pos)
            bpm = float(body[pos + 1:end])
            pos = end + 1
            continue
        if c == "{":
            end = body.index("}", pos)
            div = int(body[pos + 1:end])
            pos = end + 1
            continue
        if c == ",":
            parts = [p for p in re.split(r"[/`]", token) if p.strip()]
            made = [n for n in (_parse_note(p, t, bpm) for p in parts) if n]
            if made:
                group += 1
                for n in made:
                    n.group, n.group_size = group, len(made)
                notes.extend(made)
            token = ""
            t += 240.0 / bpm / div
            pos += 1
            continue
        token += c
        pos += 1
    return notes, t


if __name__ == "__main__":
    import sys
    ns, end = parse_chart(Path(sys.argv[1] if len(sys.argv) > 1 else "maidata.txt"))
    print(len(ns), "notes, chart ends", round(end, 3))
    for n in ns[:40]:
        print(round(n.t, 3), n.kind, n.key, "B" if n.brk else "", round(n.dur, 3),
              [(s.segments, round(s.t_start, 3), round(s.duration, 3), s.brk) for s in n.slides])
