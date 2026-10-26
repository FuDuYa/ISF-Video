"""Camera and full-screen effects applied after a battle frame is composed.

The battle frame is treated as a flat card in 3D space: the camera can push in,
pan, tilt (pitch / yaw) and roll it with perspective, plus short "kicks" on hits.
Afterwards screen effects run in screen space: vignette, speed lines, ghost
(double image), glitch bands and full-screen inversion. Everything is a pure
function of time; the schedule lives in cinema.py.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from common import W, H, clamp, smooth, out_cubic

PAD = 160
EASE = {
    "smooth": smooth,
    "linear": clamp,
    "out": out_cubic,
    "in": lambda u: clamp(u) ** 3,
}

IDENTITY = dict(focus=(320.0, 240.0), zoom=1.0, pitch=0.0, yaw=0.0, roll=0.0)


def camera_at(t: float, keys, kicks) -> dict:
    """keys: list of dicts with t, focus, zoom, pitch, yaw, roll, ease (ease into this key).
    A focus may also be a function of t (e.g. one that follows the soul)."""
    def focus(k):
        f = k["focus"]
        return f(t) if callable(f) else f

    if not keys or t <= keys[0]["t"]:
        p = dict(keys[0]) if keys and t >= keys[0]["t"] - 1e-9 else dict(IDENTITY)
    elif t >= keys[-1]["t"]:
        p = dict(keys[-1])
    else:
        for a, b in zip(keys, keys[1:]):
            if t < b["t"]:
                break
        u = EASE[b.get("ease", "smooth")]((t - a["t"]) / max(1e-6, b["t"] - a["t"]))
        p = {}
        for k in ("zoom", "pitch", "yaw", "roll"):
            p[k] = a[k] + (b[k] - a[k]) * u
        p["focus"] = tuple(np.array(focus(a), float) * (1 - u) + np.array(focus(b), float) * u)
    if callable(p.get("focus")):
        p["focus"] = p["focus"](t)
    p = {k: p.get(k, IDENTITY[k]) for k in IDENTITY}
    for k in kicks:
        tau = t - k["t"]
        if 0 <= tau < 1.0:
            e = math.exp(-tau * k.get("decay", 16))
            if e < .003:
                continue
            p["zoom"] += k.get("zoom", 0) * e
            p["roll"] += k.get("roll", 0) * e
            p["yaw"] += k.get("yaw", 0) * e
            p["pitch"] += k.get("pitch", 0) * e
    return p


def is_identity(p) -> bool:
    return (abs(p["zoom"] - 1) < 1e-4 and abs(p["pitch"]) < 1e-3 and abs(p["yaw"]) < 1e-3
            and abs(p["roll"]) < 1e-3 and abs(p["focus"][0] - 320) < .01
            and abs(p["focus"][1] - 240) < .01)


def warp(frame: np.ndarray, p, plane: np.ndarray | None = None) -> np.ndarray:
    """`plane`: the frame already inside a (H + 2 PAD, W + 2 PAD) plane with
    content past its edges (otherwise the padding is black)."""
    if is_identity(p):
        return frame
    focal = 800.0
    dist = focal / p["zoom"]
    pitch, yaw, roll = (math.radians(p[k]) for k in ("pitch", "yaw", "roll"))
    cp, sp, cy, sy, cr, sr = (math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw),
                              math.cos(roll), math.sin(roll))
    rot = (np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]]) @
           np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]) @
           np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]]))
    fx, fy = p["focus"]

    def project(x, y):
        q = rot @ np.array([x - fx, y - fy, 0.0])
        s = focal / max(100, dist - q[2])
        return W / 2 + q[0] * s, H / 2 + q[1] * s

    if plane is None:
        plane = np.zeros((H + 2 * PAD, W + 2 * PAD), np.uint8)
        plane[PAD:PAD + H, PAD:PAD + W] = frame
    corners = [(0, 0), (plane.shape[1], 0), (plane.shape[1], plane.shape[0]), (0, plane.shape[0])]
    eq, val = [], []
    for sx, sy in corners:
        dx, dy = project(sx - PAD, sy - PAD)
        eq += [[dx, dy, 1, 0, 0, 0, -sx * dx, -sx * dy], [0, 0, 0, dx, dy, 1, -sy * dx, -sy * dy]]
        val += [sx, sy]
    coef = np.linalg.solve(np.array(eq), np.array(val))
    out = Image.fromarray(plane).transform((W, H), Image.Transform.PERSPECTIVE, coef,
                                           Image.Resampling.NEAREST, fillcolor=0)
    return np.asarray(out).copy()


# ------------------------------------------------------------------ screen effects
_YY, _XX = np.mgrid[0:H, 0:W]


def project_point(p, x, y):
    """Where world point (x, y) lands on screen under camera p (same maths as warp)."""
    if is_identity(p):
        return float(x), float(y)
    focal = 800.0
    dist = focal / p["zoom"]
    pitch, yaw, roll = (math.radians(p[k]) for k in ("pitch", "yaw", "roll"))
    cp, sp, cy, sy, cr, sr = (math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw),
                              math.cos(roll), math.sin(roll))
    rot = (np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]]) @
           np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]) @
           np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]]))
    fx_, fy_ = p["focus"]
    q = rot @ np.array([x - fx_, y - fy_, 0.0])
    s = focal / max(100, dist - q[2])
    return W / 2 + q[0] * s, H / 2 + q[1] * s


def _noise_field(seed, cells, size=(H, W)):
    rng = np.random.default_rng(seed)
    small = (rng.random(cells) * 255).astype(np.uint8)
    big = Image.fromarray(small).resize((size[1] * 2, size[0] * 2), Image.Resampling.BICUBIC)
    return np.asarray(big, np.float32) / 255 - .5          # twice the screen, to scroll through


_INK_N1 = _noise_field(41, (14, 18))
_INK_N2 = _noise_field(42, (40, 52))
_INK_ANG = np.random.default_rng(43).uniform(-1, 1, 11)


def crisis_ink(frame, t, centre, radius, throb=0.0, splats=()):
    """Screen-wide ink closing in on `centre` (the soul on screen): smoky, ragged
    ink creeps in from every edge, leaving a clear hole of about `radius` px round
    the soul (pale at its rim, dark grey - never solid black - at the edges).
    `throb` darkens it for a heartbeat; `splats`: (x, y, r, age, seed) blots of
    ink thrown on the lens, running down and fading."""
    cx, cy = centre
    ox, oy = int(t * 23) % W, int(t * 17) % H                 # the smoke drifts
    n1 = _INK_N1[oy:oy + H, ox:ox + W]
    n2 = _INK_N2[(oy * 2) % H:(oy * 2) % H + H, (ox * 3) % W:(ox * 3) % W + W]
    dx, dy = (_XX - cx) / 1.2, (_YY - cy)
    d = np.hypot(dx, dy)
    th = np.arctan2(dy, dx)
    k = (th / math.tau + .5) * len(_INK_ANG) + t * .8       # a few broad lobes round the soul
    i0 = np.floor(k).astype(int)
    fr = .5 - .5 * np.cos(np.pi * (k - i0))
    ang = _INK_ANG[i0 % len(_INK_ANG)] * (1 - fr) + _INK_ANG[(i0 + 1) % len(_INK_ANG)] * fr
    rim = radius * (1 + .12 * ang) + 60 * n1 + 16 * n2
    e = d - rim
    body = np.clip(e / 28.0, 0, 1)                         # a short, wet edge...
    band = np.exp(-((e - 8) / 10.0) ** 2) * (e > 0)          # ...where the ink pools darker
    a = np.clip(body * (.8 + .12 * n2) + .18 * band + throb * .15 * (body > 0), 0, .9)
    for x, y, r, age, sd in splats:                          # blots on the lens
        fade = 1 - smooth(age / .7)
        if fade <= 0:
            continue
        rr = np.random.default_rng(int(sd))
        blob = np.hypot(_XX - x, _YY - y) < r * (1 + .3 * _INK_N2[(int(sd) * 7) % H:(int(sd) * 7) % H + H,
                                                                  (int(sd) * 13) % W:(int(sd) * 13) % W + W])
        for _ in range(3):                                   # it runs down
            hx = x + rr.uniform(-.7, .7) * r
            run = r * (.6 + 2.4 * out_cubic(min(1.0, age / .6))) * rr.uniform(.6, 1.2)
            blob |= (np.abs(_XX - hx) < max(2, r * .12)) & (_YY > y) & (_YY < y + run)
        a = np.maximum(a, blob * .82 * fade)
    out = frame.astype(np.float32) * (1 - a) + 14 * a
    return np.clip(out, 0, 255).astype(np.uint8)


def vignette(frame, strength, center=(320, 300)):
    if strength <= 0:
        return frame
    d = np.hypot((_XX - center[0]) / 330, (_YY - center[1]) / 260)
    m = 1 - strength * np.clip((d - .35) / .75, 0, 1) ** 1.5
    return (frame * m).astype(np.uint8)


def speed_lines(frame, t, strength, center=(320, 318), seed=0):
    if strength <= 0:
        return frame
    rng = np.random.default_rng(int(t * 60) * 31 + seed)
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    n = int(26 * strength)
    for _ in range(n):
        a = rng.uniform(0, math.tau)
        r0 = rng.uniform(200, 260)
        r1 = r0 + rng.uniform(80, 260)
        w = int(rng.integers(1, 3)) * 2
        p0 = (center[0] + math.cos(a) * r0, center[1] + math.sin(a) * r0)
        p1 = (center[0] + math.cos(a) * r1, center[1] + math.sin(a) * r1)
        d.line([p0, p1], fill=int(rng.uniform(90, 200)), width=w)
    return np.asarray(img).copy()


def run_lines(frame, t, t0, strength, seed=0, sides=((0, 205), (435, 640))):
    """Horizontal speed lines streaking left down both sides of the screen (a wall
    rushing past in the run). Pure function of t; lightens only."""
    if strength <= 0:
        return frame
    rng = np.random.default_rng(seed)
    img = Image.new("L", (W, H))
    d = ImageDraw.Draw(img)
    for x_lo, x_hi in sides:
        span = x_hi - x_lo
        for _ in range(11):
            y = rng.uniform(16, H - 12)
            L = rng.uniform(50, 190)
            v = rng.uniform(1400, 2300)
            x = x_lo + (rng.uniform(0, span + L) - v * (t - t0)) % (span + L) - L
            val = int(rng.uniform(95, 205) * strength)
            d.line([(x, y), (x + L, y)], fill=val, width=int(rng.integers(1, 3)) * 2)
    lines = np.asarray(img)
    keep = np.zeros_like(lines)
    for x_lo, x_hi in sides:
        keep[:, x_lo:x_hi] = lines[:, x_lo:x_hi]
    return np.maximum(frame, keep)


def ghost(frame, dx, amount):
    if amount <= 0 or dx == 0:
        return frame
    sh = np.zeros_like(frame)
    if dx > 0:
        sh[:, dx:] = frame[:, :-dx]
    else:
        sh[:, :dx] = frame[:, -dx:]
    return np.maximum(frame, (sh * amount).astype(np.uint8))


def glitch(frame, t, strength, seed=0):
    if strength <= 0:
        return frame
    rng = np.random.default_rng(int(t * 60) * 7 + seed)
    out = frame.copy()
    for _ in range(int(3 + 5 * strength)):
        y0 = int(rng.integers(0, H - 8))
        h = int(rng.integers(4, 28)) // 2 * 2
        dx = int(rng.integers(-24, 25) * strength) // 2 * 2
        out[y0:y0 + h] = np.roll(frame[y0:y0 + h], dx, axis=1)
    return out


def invert(frame):
    return 255 - frame
