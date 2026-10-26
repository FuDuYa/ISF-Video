"""Ink Sans' speech bubble, UT battle style: a white bubble to his right with a
tail pointing at him, black text typed out character by character, no voice.

Chinese text uses Fusion Pixel 12px (SIL OFL, assets/fonts/), drawn at 2x so it
sits on the same 2-px grain as the Ink Sans sprites.
"""

from __future__ import annotations

import bisect

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import choreo as C
from common import FONTS

tm = C.tm
FALLBACK_FONT = FONTS / "fusion-pixel-12px.ttf"
# Ink Sans speaks in FZShaoEr (方正少儿) when its file is placed in assets/fonts/
PREFERRED = ("fzshaoer", "方正少儿", "fzsejw", "shaoer")
SCALE = 1                        # text drawn at native pixel size


def font_path():
    for p in sorted(FONTS.glob("*")):
        if p.suffix.lower() in (".ttf", ".otf", ".ttc") and any(k in p.name.lower() for k in PREFERRED):
            return p, True
    return FALLBACK_FONT, False
CPS = 16.0                       # characters per second while typing
MAX_CHARS = 9                    # per line inside the bubble

# bubble geometry (640x480 world pixels)
BX, BY, BW, BH = 370, 70, 170, 66
TAIL_TIP = (350, 102)
PADDING = 12


def _times():
    from chart import parse_chart
    from common import ROOT
    notes, _ = parse_chart(ROOT / "maidata.txt", C.OFFSET)
    return sorted({round(n.t, 4) for n in notes})


_T = _times()


def _note(m0, idx=0):
    """The idx-th note time at or after measure m0."""
    i = bisect.bisect_left(_T, tm(m0) - 1e-3)
    return _T[i + idx]


# (start, end, text, style): "ink" lines are painted in by ink together with
# Ink Sans (intro.py), "type" lines pop in and type out character by character
LINES = [
    (tm(5), _note(6) - .02, "还是不够。", "ink"),                        # forms with Ink Sans
    (_note(6), tm(8.5) - .02, "你的决心力量在战斗中变得更强大了？", "type"),      # the brush is summoned
    # sixteenths: no bubble
    (tm(9.25), tm(11) - .03, "既然灵魂破碎了这么多次也没有消失……", "type"),
    (tm(11), tm(12.75) + .05, "那就让我亲手把你拉进地狱里吧。", "type"),
]
INK_HOOK = None      # set by render.py: draws the ink-forming bubble until it is done


class Dialogue:
    def __init__(self):
        path, self.vector = font_path()
        self.font = ImageFont.truetype(str(path), 16 if self.vector else 12)
        self.cell = 16 if self.vector else 12
        self.cache: dict[str, np.ndarray] = {}

    def glyph(self, ch: str) -> np.ndarray:
        if ch not in self.cache:
            img = Image.new("L", (20, 20))
            d = ImageDraw.Draw(img)
            d.fontmode = "1"                      # no antialiasing: a pixel bitmap
            d.text((0, 0), ch, font=self.font, fill=255)
            w = max(1, int(round(self.font.getlength(ch))))
            a = np.asarray(img)[:self.cell + 2, :min(w, 20)] > 0
            self.cache[ch] = a.repeat(SCALE, 0).repeat(SCALE, 1)
        return self.cache[ch]

    @staticmethod
    def bubble(frame: np.ndarray):
        img = Image.fromarray(frame)
        d = ImageDraw.Draw(img)
        x0, y0, x1, y1 = BX, BY, BX + BW - 1, BY + BH - 1
        d.rectangle((x0 + 4, y0, x1 - 4, y1), fill=255)
        d.rectangle((x0, y0 + 4, x1, y1 - 4), fill=255)
        d.rectangle((x0 + 2, y0 + 2, x1 - 2, y1 - 2), fill=255)
        tip = TAIL_TIP
        d.polygon([(x0 + 1, tip[1] - 10), tip, (x0 + 1, tip[1] + 2)], fill=255)
        frame[:] = np.asarray(img)

    @staticmethod
    def layout(text: str) -> list[int]:
        """Line index for every character; a line never starts with punctuation
        (the previous character moves down with it)."""
        PUNCT = "。，、？！…；：」』）"
        lines, cur = [], []
        for ch in text:
            if len(cur) == MAX_CHARS:
                if ch in PUNCT:
                    carry = cur.pop()
                    lines.append(cur)
                    cur = [carry]
                else:
                    lines.append(cur)
                    cur = []
            cur.append(ch)
        lines.append(cur)
        out = []
        for i, ln in enumerate(lines):
            out += [i] * len(ln)
        return out

    def render_text(self, frame: np.ndarray, text: str, shown: int | None = None):
        rows = self.layout(text)
        x, y = BX + PADDING, BY + PADDING
        row = 0
        for i, ch in enumerate(text):
            if shown is not None and i >= shown:
                break
            if rows[i] != row:
                row = rows[i]
                x, y = BX + PADDING, BY + PADDING + row * ((self.cell + 1) * SCALE + 2)
            g = self.glyph(ch)
            h, w = g.shape
            region = frame[y:y + h, x:x + w]
            region[g[:region.shape[0], :region.shape[1]]] = 0
            x += w

    def static(self, text: str) -> np.ndarray:
        """Bubble with the full text on an empty frame (used for the ink formation)."""
        frame = np.zeros((480, 640), np.uint8)
        self.bubble(frame)
        self.render_text(frame, text)
        return frame

    def draw(self, frame: np.ndarray, t: float):
        for t0, t1, text, style in LINES:
            if t0 <= t < t1:
                break
        else:
            return
        if style == "ink":
            if INK_HOOK is not None and INK_HOOK(frame, t):
                return
            self.bubble(frame)
            self.render_text(frame, text)
            return
        self.bubble(frame)
        self.render_text(frame, text, int((t - t0) * CPS) + 1)
