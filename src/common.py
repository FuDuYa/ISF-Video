"""Shared helpers: paths, greyscale sprites, bitmap font, easing, numpy pasting."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SPRITES = ROOT / "sprite"
FONTS = ROOT / "assets" / "fonts"
W, H = 640, 480
FPS = 60


# ---------------------------------------------------------------- easing
def clamp(v: float, a: float = 0.0, b: float = 1.0) -> float:
    return a if v < a else b if v > b else v


def smooth(v: float) -> float:
    v = clamp(v)
    return v * v * (3 - 2 * v)


def out_cubic(v: float) -> float:
    return 1 - (1 - clamp(v)) ** 3


def in_quad(v: float) -> float:
    v = clamp(v)
    return v * v


def lerp(a, b, u):
    return a + (b - a) * u


# ---------------------------------------------------------------- sprites
class Sprite:
    """Greyscale value + binary alpha, luminance converted from the source PNG."""

    def __init__(self, value: np.ndarray, alpha: np.ndarray):
        self.v = value
        self.a = alpha
        self.h, self.w = value.shape

    @classmethod
    def load(cls, name: str) -> "Sprite":
        with Image.open(SPRITES / name) as src:
            rgba = src.convert("RGBA")
        luma = np.asarray(rgba.convert("L"), dtype=np.uint8)
        alpha = np.asarray(rgba.getchannel("A")) > 127
        return cls(luma.copy(), alpha)


VIRTUAL: dict[str, Sprite] = {}     # derived sprites registered at runtime


@lru_cache(maxsize=None)
def _load(name: str) -> Sprite:
    return Sprite.load(name)


def sprite(name: str) -> Sprite:
    if name in VIRTUAL:
        return VIRTUAL[name]
    return _load(name)


def paste(frame: np.ndarray, spr: Sprite, x: int, y: int, gain: float = 1.0,
          mask: np.ndarray | None = None) -> None:
    """Paste a sprite with binary alpha. `mask` optionally limits which sprite pixels show."""
    x, y = int(round(x)), int(round(y))
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(frame.shape[1], x + spr.w), min(frame.shape[0], y + spr.h)
    if x0 >= x1 or y0 >= y1:
        return
    sv = spr.v[y0 - y:y1 - y, x0 - x:x1 - x]
    sa = spr.a[y0 - y:y1 - y, x0 - x:x1 - x]
    if mask is not None:
        sa = sa & mask[y0 - y:y1 - y, x0 - x:x1 - x]
    if gain < 1.0:
        sv = (sv.astype(np.float32) * gain).astype(np.uint8)
    region = frame[y0:y1, x0:x1]
    region[sa] = sv[sa]


# ---------------------------------------------------------------- bitmap font
class BitmapFont:
    """CYF's uibattlesmall atlas (UT battle HUD font), read glyph by glyph."""

    ALIASES = {" ": "space", "/": "slash"}

    def __init__(self, stem: str = "uibattlesmall"):
        with Image.open(FONTS / f"{stem}.png") as src:
            atlas = src.convert("RGBA")
        tree = ET.parse(FONTS / f"{stem}.xml")
        self.glyphs: dict[str, Sprite] = {}
        for item in tree.findall("./spritesheet/sprite"):
            rect = item.find("rect")
            if rect is None:
                continue
            x, y, w, h = (int(rect.attrib[k]) for k in ("x", "y", "w", "h"))
            top = atlas.height - y - h          # Unity rects use a bottom-left origin
            crop = atlas.crop((x, top, x + w, top + h))
            luma = np.asarray(crop.convert("L"), dtype=np.uint8)
            alpha = np.asarray(crop.getchannel("A")) > 127
            self.glyphs[item.attrib["name"]] = Sprite(np.where(alpha, 255, 0).astype(np.uint8), alpha)

    def draw(self, frame: np.ndarray, text: str, x: int, y: int, tracking: int = 2,
             gain: float = 1.0) -> int:
        for ch in text:
            g = self.glyphs[self.ALIASES.get(ch, ch)]
            if ch != " ":
                paste(frame, g, x, y + 15 - g.h, gain)
            x += g.w + tracking
        return x


def hp_label() -> Sprite:
    """Small two-letter HP tag drawn in 2x2 pixel blocks, like the UT battle HUD."""
    rows = ["X..X.XXX.",
            "X..X.X..X",
            "XXXX.XXX.",
            "X..X.X...",
            "X..X.X..."]
    small = np.array([[c == "X" for c in r] for r in rows])
    big = small.repeat(2, 0).repeat(2, 1)
    return Sprite(np.where(big, 255, 0).astype(np.uint8), big)


# ---------------------------------------------------------------- maimai geometry
def key_angle(k: int) -> float:
    """Screen angle (radians, clockwise from up) of maimai button k (1..8)."""
    return math.radians(22.5 + 45.0 * (k - 1))


def key_vec(k: int) -> tuple[float, float]:
    a = key_angle(k)
    return math.sin(a), -math.cos(a)


def save_atomic(path, arr) -> None:
    """np.save that never leaves a half-written file behind: render workers build
    the same caches at the same time, and one may read while another writes."""
    import os
    import numpy as _np
    from pathlib import Path as _P
    path = _P(path)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npy")
    _np.save(tmp, arr)
    os.replace(tmp, path)


def load_cached(path):
    """np.load, or None if the cache file is missing or unreadable (then rebuild it)."""
    import numpy as _np
    try:
        return _np.load(path)
    except (OSError, ValueError, EOFError):
        return None

