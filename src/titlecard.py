"""Rhythm-game BGA style title card on the eighth notes of M16.

Black screen; on each eighth note a piece pops in: a white slash line, the big
title, two brush swooshes painted along the pp/qq slides, then the subtitle.
Text uses Fusion Pixel 12px (pixel Latin), enlarged with nearest neighbour.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import choreo as C
from common import FONTS, H, W, clamp, out_cubic, smooth

tm = C.tm
TITLE = "Ink! Sans Fight"
SUBTITLE = "Bgm:Larva"

T0 = tm(16.125)              # 5/6   blackout + line (the canvas is set on 3/4 before it)
T_TITLE = tm(16.125)         # 5/6   big title slams in
T_SWOOSH = tm(16.25)         # 1pp4/8qq5: swooshes painted while the slides travel
SW_START, SW_END = tm(16.25) + 60 / 202, tm(16.25) + 60 / 202 + 240 / 202 / 8 * 3
T_SUB = tm(16.375)           # 2/7   subtitle
T_END = tm(17)               # back to the battle on the next downbeat

_font = ImageFont.truetype(str(FONTS / "fusion-pixel-12px.ttf"), 12)


def _text_bitmap(text: str) -> np.ndarray:
    w = int(_font.getlength(text)) + 2
    img = Image.new("L", (w, 16))
    d = ImageDraw.Draw(img)
    d.fontmode = "1"
    d.text((1, 1), text, font=_font, fill=255)
    a = np.asarray(img) > 0
    ys, xs = np.nonzero(a)
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _logo():
    """User-made title logo (black text on transparent) -> white bitmap mask."""
    from common import ROOT
    path = ROOT / "sprite" / "fd85f15c-6520-4dda-9d8c-30d103f3411e.png"
    if not path.exists():
        return None
    from scipy import ndimage as ndi
    a = np.asarray(Image.open(path).convert("RGBA").getchannel("A")) > 127
    # keep only the "Ink!Sans Fight" line: glyph shapes that start in its rows
    lab, n = ndi.label(a)
    rows = np.nonzero(a.any(1))[0]
    title_bottom = rows[0] + 50
    keep = np.zeros_like(a)
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        if sl[0].start < title_bottom:
            keep |= lab == i
    ys, xs = np.nonzero(keep)
    return keep[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


_LOGO = _logo()
_TITLE = _text_bitmap(TITLE)
_SUB = _text_bitmap(SUBTITLE)
LOGO_Y = 186


def _blit(frame, bmp, scale, cx, cy, val=255):
    big = bmp.repeat(scale, 0).repeat(scale, 1)
    h, w = big.shape
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    xa, ya = max(0, x0), max(0, y0)
    xb, yb = min(W, x0 + w), min(H, y0 + h)
    if xa >= xb or ya >= yb:
        return
    region = frame[ya:yb, xa:xb]
    region[big[ya - y0:yb - y0, xa - x0:xb - x0]] = val


def _bezier(p0, p1, p2, n=60):
    s = np.linspace(0, 1, n)[:, None]
    return (1 - s) ** 2 * np.array(p0) + 2 * (1 - s) * s * np.array(p1) + s ** 2 * np.array(p2)


# light trails traced by the two slides (1pp4 / 8qq5) around the title
TRAIL_A = _bezier((72, 234), (150, 318), (318, 300))
TRAIL_B = _bezier((568, 234), (490, 318), (322, 300))


def _glow_layers():
    """Stepped glow around the title, computed on a 2-px grid (pixel-art bloom)."""
    from scipy import ndimage as ndi
    logo = _LOGO if _LOGO is not None else _TITLE.repeat(4, 0).repeat(4, 1)
    pad = 24
    m = np.zeros((logo.shape[0] + 2 * pad, logo.shape[1] + 2 * pad), bool)
    m[pad:pad + logo.shape[0], pad:pad + logo.shape[1]] = logo
    small = m[::2, ::2]
    blur = ndi.gaussian_filter(small.astype(float), 2.4)
    glow = np.zeros_like(blur)
    glow[blur > .05] = 14
    glow[blur > .16] = 26
    glow[blur > .34] = 42
    glow = glow.repeat(2, 0).repeat(2, 1)[:m.shape[0], :m.shape[1]]
    return m, glow.astype(np.uint8), pad


_MASK, _GLOW, _PADG = _glow_layers()
_rng = np.random.default_rng(5)
_SPARK = _rng.random((12, 3))


def active(t: float) -> bool:
    return T0 <= t < T_END


def _sparkle(d, x, y, r, v):
    d.line((x - r, y, x + r, y), fill=v, width=2)
    d.line((x, y - r, x, y + r), fill=v, width=2)


def draw(t: float) -> np.ndarray:
    frame = np.zeros((H, W), np.uint8)
    cx, cy = 320, LOGO_Y
    mh, mw = _MASK.shape
    x0, y0 = cx - mw // 2, cy - mh // 2
    # 1) a fine line of light opens from the centre
    a = t - T0
    line_k = out_cubic(a / .35)
    line_g = 1 - smooth((t - T_SUB) / .5) * .6
    if line_k > 0:
        half = 250 * line_k
        ly = cy + mh // 2 - _PADG + 14
        v = int(150 * line_g)
        frame[ly:ly + 2, int(cx - half):int(cx + half)] = v
        for ex in (int(cx - half), int(cx + half) - 2):        # bright end points
            frame[ly - 2:ly + 4, max(0, ex - 2):ex + 4] = min(255, v + 90)
    # 2) the title fades up inside a soft stepped glow, then a light sweep crosses it
    if t >= T_TITLE:
        a = t - T_TITLE
        fade = smooth(a / .18)
        glow = (_GLOW.astype(float) * fade * (1 + .25 * math.sin(t * 6))).astype(np.uint8)
        region = frame[y0:y0 + mh, x0:x0 + mw]
        np.maximum(region, glow, out=region)
        core = int(205 * fade)
        region[_MASK] = np.maximum(region[_MASK], core)
        # diagonal shine band
        sweep = (t - T_TITLE - .12) / .55
        if 0 <= sweep <= 1:
            xx = np.arange(mw)[None, :] + np.arange(mh)[:, None] * .6
            band = np.abs(xx - (-80 + sweep * (mw + 160))) < 18
            hit = _MASK & band
            region[hit] = 255
            halo = band & ~_MASK & (_GLOW > 0)
            region[halo] = np.minimum(255, region[halo].astype(int) + 40).astype(np.uint8)
        if a < .06:                                             # soft flash as it appears
            frame = np.maximum(frame, int(40 * (1 - a / .06))).astype(np.uint8)
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    # 3) light trails traced by the two slides
    if t >= SW_START - .02:
        u = smooth((t - SW_START) / (SW_END - SW_START))
        fade_out = 1 - smooth((t - SW_END - .1) / .3)
        for path in (TRAIL_A, TRAIL_B):
            n = max(2, int(u * len(path)))
            pts = [tuple(p) for p in path[:n]]
            if len(pts) > 1:
                d.line(pts, fill=int(110 * fade_out), width=2)
                hx, hy = pts[-1]
                _sparkle(d, hx, hy, 5, int(240 * fade_out))
    # glints on the letters on the beats
    for i, (sx, sy, ph) in enumerate(_SPARK):
        tt = (T_SWOOSH, T_SUB, SW_START, SW_START + .15, SW_END)[i % 5] + ph * .1
        age = t - tt
        if 0 <= age < .25 and t >= T_TITLE:
            r = 2 + 4 * math.sin(age / .25 * math.pi)
            _sparkle(d, x0 + _PADG + sx * (mw - 2 * _PADG), y0 + _PADG + sy * (mh - 2 * _PADG), r, 255)
    frame = np.asarray(img).copy()
    # 4) subtitle, letter-spaced, fades in below between two short rules
    if t >= T_SUB:
        g = smooth((t - T_SUB) / .25)
        x = 320 - (len(SUBTITLE) * 14) // 2
        sy = cy + mh // 2 - _PADG + 34
        for ch in SUBTITLE:
            bmp = _text_bitmap(ch) if ch != " " else None
            if bmp is not None:
                _blit(frame, bmp, 2, x + 6, sy, int(185 * g))
            x += 14
        rv = int(90 * g)
        frame[sy:sy + 2, 320 - (len(SUBTITLE) * 14) // 2 - 46:320 - (len(SUBTITLE) * 14) // 2 - 10] = rv
        frame[sy:sy + 2, 320 + (len(SUBTITLE) * 14) // 2 + 10:320 + (len(SUBTITLE) * 14) // 2 + 46] = rv
    # exit: the card brightens into a flash on the last frames before the cut
    if t > T_END - .08:
        k = (t - (T_END - .08)) / .08
        frame = np.maximum(frame, int(255 * k * k)).astype(np.uint8)
    return frame


# ------------------------------------------------------------------ title fragments
# Before the title card, the quick close-up cuts show pieces of the title instead:
# black screen, part of the logo at 3x with its glow, a slow drift and a light sweep.
_FRAG_SCALE = 4
_FRAG = None


def _frag_source():
    global _FRAG
    if _FRAG is None:
        glow = _GLOW.astype(np.float32) * 1.4
        img = np.maximum(glow, np.where(_MASK, 215, 0)).astype(np.uint8)
        _FRAG = img.repeat(_FRAG_SCALE, 0).repeat(_FRAG_SCALE, 1), _MASK.repeat(_FRAG_SCALE, 0).repeat(
            _FRAG_SCALE, 1)
    return _FRAG


def _word_centres(groups=(4, 4, 5)):
    """Centres (mask x) of the words of the logo: its glyphs grouped as
    "Ink!" / "Sans" / "Fight" (letter gaps are too even to split by size)."""
    cols = _MASK.any(0)
    runs, x = [], 0
    while x < len(cols):
        if cols[x]:
            s = x
            while x < len(cols) and cols[x]:
                x += 1
            runs.append((s, x))
        x += 1
    if len(runs) != sum(groups):
        return [], []
    words, spans, k = [], [], 0
    for n in groups:
        a, b = runs[k][0], runs[k + n - 1][1]
        words.append((a + b) / 2)
        spans.append((a, b))
        k += n
    return words, spans


_WORDS, _SPANS = _word_centres()


def fragment(t: float, idx: int, count: int, t0: float, t1: float, roll: float = 0.0) -> np.ndarray:
    """Cut `idx` of `count`: a close shot of the title, left to right."""
    src, mask = _frag_source()
    sh, sw = src.shape
    u = clamp((t - t0) / max(1e-6, t1 - t0))
    pad = _PADG * _FRAG_SCALE
    span = sw - 2 * pad
    if count == len(_WORDS):                                      # one word per cut
        cx = _WORDS[idx] * _FRAG_SCALE + (u - .5) * 60
    else:
        cx = pad + span * (idx + .5) / count + (u - .5) * 60      # slow drift
    cy = sh / 2
    big = Image.fromarray(src)
    if roll:
        big = big.rotate(roll, resample=Image.Resampling.NEAREST, center=(cx, cy))
    x0, y0 = int(cx - W / 2), int(cy - H / 2)
    frame = np.zeros((H, W), np.uint8)
    crop = np.asarray(big.crop((x0, y0, x0 + W, y0 + H)))
    frame[:] = crop
    if count == len(_WORDS):                                      # only this word shows
        a, b = _SPANS[idx]
        xs = np.arange(W) + x0
        keep = (xs >= (a - 5) * _FRAG_SCALE) & (xs < (b + 5) * _FRAG_SCALE)
        frame[:, ~keep] = 0
    # light sweep across the letters
    xx = np.arange(W)[None, :] + np.arange(H)[:, None] * .5
    band = np.abs(xx - (-200 + u * (W + 400))) < 46
    lit = band & (frame > 150)
    frame[lit] = 255
    # pop in: one bright frame
    if t - t0 < 1 / 60:
        frame = np.maximum(frame, 70).astype(np.uint8)
    return frame
