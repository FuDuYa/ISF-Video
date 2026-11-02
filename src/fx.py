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

PAD = 220
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


def page_turn(img: np.ndarray, phi: float, cx: float, cy: float, f: float = 1100.0) -> np.ndarray:
    """The picture as a page turned over about the vertical line x = cx by phi (rad):
    at pi it is mirrored; on the way it swings up at us (perspective) and darkens
    as it goes edge-on."""
    phi = phi % (2 * math.pi)
    if min(phi, 2 * math.pi - phi) < 1e-4:
        return img
    h, w = img.shape
    out = np.zeros_like(img)
    xs = np.arange(w, dtype=float)
    if abs(phi - math.pi) < 1e-4:                         # turned right over: mirrored
        src = np.round(2 * cx - xs).astype(int)
        ok = (src >= 0) & (src < w)
        out[:, ok] = img[:, src[ok]]
        return out
    c, s = math.cos(phi), abs(math.sin(phi))
    d = cx - xs                                           # from the hinge
    sc = f / (f - np.abs(d) * s)                          # (it comes up at us)
    xp = cx - d * c * sc
    order = np.argsort(xp)
    src = np.interp(xs, xp[order], xs[order])
    okx = (xs >= xp.min()) & (xs <= xp.max())
    si = np.clip(np.round(src).astype(int), 0, w - 1)
    sy = np.round(cy + (np.arange(h)[:, None] - cy) / sc[si][None, :]).astype(int)
    ok = okx[None, :] & (sy >= 0) & (sy < h)
    yy, xx = np.nonzero(ok)
    out[yy, xx] = img[sy[yy, xx], si[xx]]
    return (out * (.55 + .45 * abs(c))).astype(np.uint8)


def sheet_fold(img: np.ndarray, M, b, fold, pad: float) -> np.ndarray:
    """The page's folds (choreo.sheet_state) on a picture padded by `pad`: the
    finished ones (p -> M p + b on screen), then the one under way - turned over
    about its line (a page turn, the line stood upright for it) and carried along."""
    h, w = img.shape
    out = img
    if not (np.array_equal(M, np.eye(2)) and np.allclose(b, 0)):
        yy, xx = np.mgrid[0:h, 0:w]
        qx, qy = xx - pad - b[0], yy - pad - b[1]
        sx = np.round(M[0, 0] * qx + M[1, 0] * qy + pad).astype(int)     # (M is orthogonal: M^-1 = M^T)
        sy = np.round(M[0, 1] * qx + M[1, 1] * qy + pad).astype(int)
        ok = (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)
        out = np.zeros_like(img)
        out[ok] = img[sy[ok], sx[ok]]
    if fold is not None:
        (px, py), deg, phi, d = fold
        cx, cy = px + pad, py + pad
        rot = deg - 90.0                                  # (turns the fold line upright)
        im = Image.fromarray(out)
        if abs(rot) > 1e-6:
            im = im.rotate(rot, resample=Image.NEAREST, center=(cx, cy), fillcolor=0)
        a = page_turn(np.asarray(im), phi, cx, cy)
        im = Image.fromarray(a)
        if abs(rot) > 1e-6:
            im = im.rotate(-rot, resample=Image.NEAREST, center=(cx, cy), fillcolor=0)
        out = np.asarray(im)
        dx, dy = int(round(d[0])), int(round(d[1]))
        if dx or dy:
            sh = np.zeros_like(out)
            ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
            xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
            sh[yd, xd] = out[ys, xs]
            out = sh
    return out


def _quad_fill(out, q, pad, src=None, img=None, value=0, shade=1.0):
    """Fill the quad q (4 x 2, screen) on the padded picture `out`: with the rect
    `src` of `img` mapped onto it (perspective), or a flat value."""
    h, w = out.shape
    Q = np.asarray(q, float) + pad
    x0, y0 = np.floor(Q.min(0)).astype(int)
    x1, y1 = np.ceil(Q.max(0)).astype(int) + 1
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return
    # homography from the unit square to the quad, inverted
    U = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    A = []
    for (u, v), (x, y) in zip(U, Q):
        A.append([u, v, 1, 0, 0, 0, -u * x, -v * x, -x])
        A.append([0, 0, 0, u, v, 1, -u * y, -v * y, -y])
    _, _, Vt = np.linalg.svd(np.asarray(A))
    Hm = Vt[-1].reshape(3, 3)
    try:
        Hi = np.linalg.inv(Hm)
    except np.linalg.LinAlgError:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    den = Hi[2, 0] * xx + Hi[2, 1] * yy + Hi[2, 2]
    den = np.where(np.abs(den) < 1e-9, 1e-9, den)
    uu = (Hi[0, 0] * xx + Hi[0, 1] * yy + Hi[0, 2]) / den
    vv = (Hi[1, 0] * xx + Hi[1, 1] * yy + Hi[1, 2]) / den
    m = (uu >= 0) & (uu <= 1) & (vv >= 0) & (vv <= 1)
    if not m.any():
        return
    blk = out[y0:y1, x0:x1]
    if src is None:
        blk[m] = value
    else:
        sx0, sy0, sx1, sy1 = src
        sx = np.clip((sx0 + uu * (sx1 - sx0) + pad).astype(int), 0, img.shape[1] - 1)
        sy = np.clip((sy0 + vv * (sy1 - sy0) + pad).astype(int), 0, img.shape[0] - 1)
        v = img[sy[m], sx[m]]
        blk[m] = v if shade >= 1 else (v * shade).astype(out.dtype)


def paper_pack(img: np.ndarray, geo, pad: float, ghosts=()) -> np.ndarray:
    """M38's folding up (choreo.pack_geometry): the sheet as it is on screen (img) cut
    into cells and drawn where they lie now - face up showing what is on them, face
    down their back (blank paper, darker the more edge-on) - with a hairline at
    every fold; ghosts: the packet a moment before, dimmer, behind it (hopping)."""
    out = np.zeros_like(img)
    for g, dim in list(ghosts) + [(geo, 1.0)]:
        cells, _ = g
        layer = np.zeros_like(img) if dim < 1 else out
        for q, front, src, flat in cells:                  # (turned up off the table: in shadow)
            if front:
                _quad_fill(layer, q, pad, src, img, shade=.7 + .3 * flat)
            else:
                _quad_fill(layer, q, pad, value=int(140 + 70 * flat))
        im = Image.fromarray(layer)
        d = ImageDraw.Draw(im)
        for q, *_ in cells:                                # the folds: hairlines
            d.line([tuple(p + pad) for p in q] + [tuple(q[0] + pad)], fill=120, width=1)
        layer = np.asarray(im)
        if dim < 1:
            m = layer > 0
            out[m] = np.maximum(out[m], (layer[m] * dim).astype(out.dtype))
        else:
            out = layer.copy()
    return out


def pack_bursts(img: np.ndarray, knocks, pad: float) -> np.ndarray:
    """M38's packet landing hard (choreo.PACK_KNOCKS): its outline flashes outwards
    and ink is thrown off its edges - white off the paper. knocks: (how long ago,
    its box on screen then, how hard, seed)."""
    if not knocks:
        return img
    lay = Image.new("L", (img.shape[1], img.shape[0]), 0)
    d = ImageDraw.Draw(lay)
    for a, (x0, y0, x1, y1), pw, seed in knocks:
        life = .16 + .14 * pw
        q = a / life
        if q >= 1:
            continue
        x0, y0, x1, y1 = x0 + pad, y0 + pad, x1 + pad, y1 + pad
        if pw >= .9 and a < .07:                           # (a fold slammed down:) the outline, flung outwards
            g = 3 + 16 * pw * (a / .07) ** .6
            d.rectangle([x0 - g, y0 - g, x1 + g, y1 + g], outline=int(235 * (1 - a / .07)), width=2 + int(2 * pw))
        rng = np.random.default_rng(seed)
        per = 2 * (x1 - x0) + 2 * (y1 - y0)
        for r1, r2, r3, r4 in rng.uniform(0, 1, (int(4 + 20 * pw ** 1.5), 4)):
            s_ = r1 * per
            if s_ < x1 - x0:
                pt, n = (x0 + s_, y0), (0, -1)
            elif s_ < x1 - x0 + y1 - y0:
                pt, n = (x1, y0 + s_ - (x1 - x0)), (1, 0)
            elif s_ < 2 * (x1 - x0) + y1 - y0:
                pt, n = (x1 - (s_ - (x1 - x0) - (y1 - y0)), y1), (0, 1)
            else:
                pt, n = (x0, y1 - (s_ - 2 * (x1 - x0) - (y1 - y0))), (-1, 0)
            dist = 4 + (20 + 60 * pw) * (1 - (1 - q) ** 3) * (.4 + r2)
            ang = math.atan2(n[1], n[0]) + (r3 - .5) * 1.3
            r = (1.5 + 3 * r4) * (.6 + .4 * pw) * (1 - .7 * q)
            cx, cy = pt[0] + math.cos(ang) * dist, pt[1] + math.sin(ang) * dist
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=int(235 * (1 - q)))
    return np.maximum(img, np.asarray(lay))


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


def crisis_ink(frame, t, centre, radius, throb=0.0, splats=(), gain=1.0):
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
        fade = smooth(age / .35) * (1 - smooth((age - .5) / .7))    # it seeps in, then fades
        if fade <= 0:
            continue
        rr = np.random.default_rng(int(sd))
        r = r * (.3 + .7 * out_cubic(min(1.0, age / .4)))             # spreading out slowly
        blob = np.hypot(_XX - x, _YY - y) < r * (1 + .3 * _INK_N2[(int(sd) * 7) % H:(int(sd) * 7) % H + H,
                                                                  (int(sd) * 13) % W:(int(sd) * 13) % W + W])
        for _ in range(3):                                   # it runs down
            hx = x + rr.uniform(-.7, .7) * r
            run = r * (.6 + 2.4 * smooth(min(1.0, age / 1.0))) * rr.uniform(.6, 1.2)
            blob |= (np.abs(_XX - hx) < max(2, r * .12)) & (_YY > y) & (_YY < y + run)
        a = np.maximum(a, blob * .82 * fade)
    a = a * gain                                             # (it seeps in and out, never pops)
    out = frame.astype(np.float32) * (1 - a) + 14 * a
    return np.clip(out, 0, 255).astype(np.uint8)


def dismantle(frame, t, slices, region):
    """解 - the box cut by invisible blades: every slice (t_cut, (x, y), angle,
    shift, life) is a dead-straight hairline through the box; inside `region`
    (cx, cy, rx, ry: an ellipse round the box on screen, bigger than the box) the
    picture on one side of the cut slips along it by `shift` px and eases back,
    while the hairline flashes (inverted, so it shows on paper and on black
    alike). Nothing outside the region is touched."""
    cx, cy, rx, ry, y_max = region
    rr = ((_XX - cx) / rx) ** 2 + ((_YY - cy) / ry) ** 2
    rr = np.where(_YY < y_max, rr, 9.0)                    # (never into the status bar and buttons)
    inside = rr < 1.0
    out = frame
    lines = []
    for tc, (px, py), ang, shift, life in slices:
        a = t - tc
        if not (0 <= a < life):
            continue
        ux, uy = math.cos(ang), math.sin(ang)
        side = inside & (((_XX - px) * -uy + (_YY - py) * ux) > 0)
        d = shift * (1 - a / life) ** 2
        dx, dy = int(round(ux * d)), int(round(uy * d))
        if dx or dy:
            moved = np.roll(out, (dy, dx), axis=(0, 1))
            out = np.where(side, moved, out)
        if a < life * .55:
            lines.append((px, py, ux, uy, 1 - a / (life * .55)))
    if not lines:
        return out
    out = out.copy()
    for px, py, ux, uy, g in lines:
        dist = np.abs((_XX - px) * -uy + (_YY - py) * ux)
        reach = rr < (1.0 - .25 * (1 - g))                  # the cut line, tapering in from the ends as it fades
        core = (dist < 1.0) & reach
        glow = (dist < 3.0) & ~core & reach
        v = out[core].astype(np.int16)
        out[core] = np.where(v < 128, 255, 0).astype(np.uint8) if g > .5 else (255 - v).astype(np.uint8)
        if g > .4:
            gv = out[glow].astype(np.int16)
            out[glow] = np.clip(gv + (np.where(gv < 128, 90, -90) * g), 0, 255).astype(np.uint8)
    return out


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
