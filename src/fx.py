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
