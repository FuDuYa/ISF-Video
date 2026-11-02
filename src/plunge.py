"""The end of the descent (M45.75-M48): the canvas soaks through with ink and lies
down into a pool, and the soul falls into it - the slow slide, from the top of the
screen to the bottom, falling ever faster - while drops of ink fall in round it on
the taps (right, left, right, left, right, closing in); on the break it goes under,
and from there on it is all in 3D.

M45.75 (the third pair): ink wells up through the sheet from the bottom, a ragged
front with blots going ahead of it, until it is all ink; the soul lights up on it.
M45.875 (the star): the soaked sheet lies down - tilted back by the same camera as
everything else (fx.warp on the frame's plane, past the screen's edges) - into a
pool, black, faintly mottled, a sheen far off; the soul stays hanging up top.
M46.125-M47 (the slide): it falls, from rest, faster and faster, down onto the
pool, its reflection coming up to meet it; on each tap a drop falls in where the
note is (ripples, a crown of drips). M47 (the break): it hits - a crown, rings -
and the camera follows it down through the surface: under it, the soul sinking
into the dark, turning, a trail of ink and bubbles behind it, the surface far
above it a ceiling of light.

Everything over the pool (the soul, its reflection, the drops, the drips) is a
point in 3D over the pool's plane, put on screen with the same maths as fx.warp
(project); what is on the pool (the soak, the ripples) is drawn on its plane
first."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

import choreo as C
import fx
from common import Sprite, W, H, clamp, smooth, paste

T_SOAK = C.tm(45.75)
T_TILT = C.tm(45.875)
T_LAID = C.tm(46.75)             # it has lain down (slowly: through most of the fall)
T_DROP = C.tm(46.125)
T_SPLASH = C.tm(47.0)
T_DIVE = T_SPLASH + .2           # the camera goes down after it
T_UNDER = T_SPLASH + .2          # under the surface
T_END = C.tm(49)                 # under it, sinking, until the rest (bga) takes over

SOAK_DUR = .15
RING = 175                       # px: the notes' ring on screen (round the middle)
TAPS = [(C.tm(46.125), 2), (C.tm(46.3125), 8), (C.tm(46.5), 3), (C.tm(46.6875), 7), (C.tm(46.875), 4)]
PITCH0, PITCH1 = 52.0, 57.0      # the pool's tilt once it has lain down; by the splash
ZOOM1 = 1.22                     # pushed in a little by the splash
FOCUS1 = (345.0, 300.0)
LAND_Y = 385                     # px: it goes in down here on screen, straight down from where it hung
NEAR = 2.3                       # x its size: it comes up off the sheet towards us, this near -
T_NEAR = C.tm(46.5)              # slowly, by here (then away, falling ever faster)
DIP = 14                         # px: (and a touch lower on screen, to stay in frame)
REFL_FADE = 190                  # its reflection shows as it comes down within this of the pool
FOCAL = 800.0
_SOUL_AT = None                  # (frame() sets it: render.soul_at)


def active(t: float) -> bool:
    return T_SOAK <= t < T_END


def key_screen(k, r=RING):
    a = math.radians((k - .5) * 45)
    return 320 + r * math.sin(a), 240 - r * math.cos(a)


# ------------------------------------------------------------- the camera (3D)
def _rot(p):
    pitch, yaw, roll = (math.radians(p[k]) for k in ("pitch", "yaw", "roll"))
    cp, sp, cy, sy, cr, sr = (math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw),
                              math.cos(roll), math.sin(roll))
    return (np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]]) @
            np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]) @
            np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]]))


def project(p, x, y, z=0.0):
    """Where (x, y) on the pool's plane, z up off it (towards the camera's side), lands
    on screen under camera p - and how many screen px a world px is there."""
    q = _rot(p) @ np.array([x - p["focus"][0], y - p["focus"][1], z])
    s = FOCAL / max(100.0, FOCAL / p["zoom"] - q[2])
    return W / 2 + q[0] * s, H / 2 + q[1] * s, s


def unproject(p, u, v, z=0.0):
    """The point (x, y) under screen point (u, v) at height z over the pool's plane."""
    R = _rot(p)
    D = FOCAL / p["zoom"]
    a, b = (u - W / 2) / FOCAL, (v - H / 2) / FOCAL
    A = np.array([[R[0, 0] + a * R[2, 0], R[0, 1] + a * R[2, 1]],
                  [R[1, 0] + b * R[2, 0], R[1, 1] + b * R[2, 1]]])
    rhs = np.array([a * D - (R[0, 2] + a * R[2, 2]) * z, b * D - (R[1, 2] + b * R[2, 2]) * z])
    d = np.linalg.solve(A, rhs)
    return np.array([d[0] + p["focus"][0], d[1] + p["focus"][1]])


_KEYS = None


def _keys():
    global _KEYS
    if _KEYS is None:
        import cinema
        p0 = fx.camera_at(T_TILT, cinema.KEYS, cinema.KICKS)
        k0 = dict(t=T_TILT, ease="linear", **p0)
        k1 = dict(t=T_LAID, focus=(330.0, 262.0), zoom=1.0, pitch=PITCH0, yaw=0.0, roll=0.0, ease="smooth")
        k2 = dict(t=T_SPLASH, focus=FOCUS1, zoom=ZOOM1, pitch=PITCH1, yaw=0.0, roll=0.0, ease="smooth")
        k3 = dict(k2, t=T_SPLASH + .08, zoom=ZOOM1 * 1.05, ease="out")      # (it lands: a breath)
        _KEYS = [k0, k1, k2, k3]
        L = landing()
        _KEYS.append(dict(t=T_DIVE, focus=(float(L[0]), float(L[1])), zoom=3.2, pitch=34.0, yaw=0.0, roll=0.0,
                          ease="in"))                                           # (down into it, after it)
    return _KEYS


def cam(t):
    if t < T_TILT:
        import cinema
        return fx.camera_at(t, cinema.KEYS, cinema.KICKS)
    return fx.camera_at(t, _keys(), [])


_LAND = None


def land_screen():
    """Where on screen it goes in: straight down from where it hung (a touch in)."""
    s0 = hang(_SOUL_AT)
    return np.array([s0[0] + .06 * (320 - s0[0]), LAND_Y])


def landing():
    """Where the soul goes in, on the pool's plane."""
    global _LAND
    if _LAND is None:
        k2 = dict(focus=FOCUS1, zoom=ZOOM1, pitch=PITCH1, yaw=0.0, roll=0.0)
        _LAND = unproject(k2, *land_screen())
    return _LAND


# --------------------------------------------------------------- the soul's fall
def soul_path(t):
    """The soul over the pool: (where it is on screen, how big it is there - its own
    size times). It comes up off the sheet towards us as the sheet lies down (hanging
    where it was, a touch lower, bobbing), then falls from rest, ever faster, almost
    straight down the screen - a little sway - and goes in on the break. Its size: it
    comes slowly nearer, growing to NEAR by T_NEAR (already falling), then falls away
    from us, ever faster. (Laid out on screen - how it reads; how far off it is from us
    goes as 1/size.)"""
    s0 = np.array(hang(_SOUL_AT), float)
    bob = 1.5 * math.sin((t - T_TILT) * 7)
    if t < T_NEAR:                                         # (nearer: growing evenly - in log - easing in and out)
        e = smooth((t - T_TILT) / (T_NEAR - T_TILT))
        k0 = _sheet_scale()
        size = k0 * (NEAR / k0) ** e
    else:                                                  # (away, gravity-like)
        w = (t - T_NEAR) / (T_SPLASH - T_NEAR)
        size = FOCAL / (FOCAL / NEAR + (FOCAL / _land_scale() - FOCAL / NEAR) * w * w)
    if t < T_DROP:
        e = smooth((t - T_TILT) / (T_DROP - T_TILT))
        return s0 + [0.0, (DIP + bob) * e], size
    u = clamp((t - T_DROP) / (T_SPLASH - T_DROP))
    sh, sl = s0 + [0.0, DIP], land_screen()
    x = sh[0] + (sl[0] - sh[0]) * u + 6 * math.sin(math.pi * u) * math.sin(u * 7.5)
    y = sh[1] + (sl[1] - sh[1]) * u ** 2.2 + bob * (1 - u) ** 2
    return np.array([x, y]), size


def _sheet_scale():
    s0 = hang(_SOUL_AT)
    p = cam(T_TILT)
    return project(p, *unproject(p, *s0))[2]


def _land_scale():
    return project(cam(T_SPLASH), *landing())[2]


def soul3d(t):
    """The soul as a point over the pool: (x, y on the plane, z up off it) - the point
    on the ray through its place on screen that is as far off as its size says."""
    (u, v), s = soul_path(t)
    p = cam(t)
    q = np.array([(u - W / 2) / s, (v - H / 2) / s, FOCAL / p["zoom"] - FOCAL / s])
    P = _rot(p).T @ q + [p["focus"][0], p["focus"][1], 0.0]
    return P[0], P[1], max(0.0, P[2])


def soul_turn(t):
    """The heart's own turn in 3D and how thick it is: it is the flat heart on the sheet
    to begin with (lying down with it), then it grows thick, rears up a little towards
    us and rocks and turns, slowly - a solid thing."""
    a = t - T_TILT
    g = smooth(a / .6)
    al = math.radians(30) * smooth((a - .15) / .7)       # (rearing up)
    be = math.radians(38) * math.sin(a * 2.0) * g        # (turning: its side shows)
    ga = math.radians(7) * math.sin(a * 2.9) * g         # (rocking)
    ca, sa = math.cos(-al), math.sin(-al)
    Rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    cb, sb = math.cos(be), math.sin(be)
    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    cg, sg = math.cos(ga), math.sin(ga)
    Rz = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])
    return Rx @ Ry @ Rz, 5.0 * smooth(a / .55)


_HANG = None


def hang(soul_at):
    """Where the soul is on screen as the sheet starts to lie down (it hangs about there)."""
    global _HANG
    if _HANG is None:
        _HANG = soul_screen(T_TILT, soul_at)
    return _HANG


def soul_screen(t, soul_at):
    """The soul on screen while it is still on the sheet (its frame, the folds, the camera)."""
    pos, _, _ = soul_at(int(round(t * 60)))
    off = np.array(C.arena_offset(t))
    M, b, _ = C.sheet_state(t)
    w = M @ (np.asarray(pos, float) + off) + b
    return fx.project_point(cam(t), w[0], w[1])


# ------------------------------------------------------------------ the pool
_TEX = {}


def _noise(shape, sigma, seed):
    from scipy import ndimage as ndi
    n = ndi.gaussian_filter(np.random.default_rng(seed).standard_normal(shape), sigma)
    return (n - n.min()) / (n.max() - n.min())


def textures(shape):
    """(the order the sheet soaks through in: 0 first; the ink's own look) over the plane."""
    if shape not in _TEX:
        h, w = shape
        Y = np.arange(h, dtype=float)[:, None] / h
        order = .62 * (1 - Y) + .26 * _noise(shape, 22, 4501) + .12 * _noise(shape, 5, 4502)
        blots = _noise(shape, 9, 4503) > .74                  # (soaking through ahead of the front)
        order = np.where(blots, order * .55, order)
        order = (order - order.min()) / (order.max() - order.min())
        mott = .6 * _noise(shape, 30, 4504) + .4 * _noise(shape, 6, 4505)
        sheen = 1 - smooth_arr(np.clip((Y - .02) / .5, 0, 1))  # (lighter far off)
        ink = np.clip(8 + 14 * mott + 46 * sheen ** 1.6, 0, 255).astype(np.float32)
        _TEX[shape] = (order, ink)
    return _TEX[shape]


def smooth_arr(u):
    return u * u * (3 - 2 * u)


_GRID = {}


def surface(shape, t):
    """The ink as a pool's surface: its mottle, a sheen far off, and slow swells - thin
    pale lines across it that drift and bend (squeezed together far off once it lies
    down: that is what makes it read as lying down), coming up as it does."""
    order, ink = textures(shape)
    g = smooth((t - T_TILT) / (T_LAID - T_TILT))
    if g <= 0:
        return ink
    if shape not in _GRID:
        h, w = shape
        Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
        bend = 9 * np.sin(X * .011 + .7) + 5 * np.sin(X * .027 + 2.1) + 18 * (_noise(shape, 40, 4506) - .5)
        far = smooth_arr(np.clip((Y - 10) / 150, 0, 1))     # (its far edge going off into the dark)
        _GRID[shape] = (Y, X, bend.astype(np.float32), far.astype(np.float32))
    Y, X, bend, far = _GRID[shape]
    w1 = np.sin((Y + bend) * .21 + t * 2.2)
    w2 = np.sin((Y * .9 - X * .18 + bend * .6) * .13 - t * 1.5)
    lines = np.clip((w1 - .82) / .18, 0, 1) * 30 + np.clip((w2 - .9) / .1, 0, 1) * 18
    return np.clip((ink + g * lines) * (1 - g * (1 - far)), 0, 255).astype(np.uint8)


def soak(plane, t):
    """The sheet soaking through: ink wells up from the bottom, blots ahead of it, a wet
    grey rim at its front; then it is all ink - a pool."""
    order, ink = textures(plane.shape)
    q = smooth((t - T_SOAK) / SOAK_DUR) * 1.06
    if q <= 0:
        return plane
    pool = surface(plane.shape, t)
    if q >= 1.06 - 1e-6:
        return pool.copy()
    out = plane.copy()
    rim = (order >= q) & (order < q + .05)
    out[rim] = np.minimum(out[rim], 150)
    wet = order < q
    out[wet] = pool[wet]
    return out


def ripples(plane, t, pad):
    """Rings spreading on the pool: where each drop went in, and where the soul did."""
    layer = Image.new("L", (plane.shape[1], plane.shape[0]), 0)
    d = ImageDraw.Draw(layer)
    for c, t0, strength, speed, life in _sources():
        a = t - t0
        if not (0 <= a < life):
            continue
        for j in range(3 if strength < 1.5 else 5):
            r = speed * a - 15 * j
            if r <= 1:
                continue
            v = int(min(200, 140 * strength * (1 - a / life) ** 1.5 * (1 - .18 * j)))
            if v < 8:
                continue
            x, y = c[0] + pad, c[1] + pad
            d.ellipse([x - r, y - r, x + r, y + r], outline=v, width=3 if j == 0 else 2)
    return np.maximum(plane, np.asarray(layer))


_SRC = None


def _sources():
    global _SRC
    if _SRC is None:
        _SRC = [(tap_point(i), t0, 1.0, 140.0, 1.1) for i, (t0, k) in enumerate(TAPS)]
        _SRC.append((landing(), T_SPLASH, 2.2, 240.0, 1.2))
    return _SRC


def tap_point(i):
    t0, k = TAPS[i]
    return unproject(cam(t0), *key_screen(k))


# ------------------------------------------------------------- over the pool
def _heart(value):
    from common import sprite
    src = sprite("player.png")
    return Sprite(np.where(src.a, value, 0).astype(np.uint8), src.a.copy())


_HEARTS = {}


def heart(k, value, flip=False):
    """The heart, k times its size, all one grey (flip: upside down - a reflection)."""
    n = max(4, int(round(16 * k / 2)) * 2)
    key = (n, value, flip)
    if key not in _HEARTS:
        src = _heart(value)
        idx = (np.arange(n) * 16 / n).astype(int)
        v, a = src.v[idx][:, idx], src.a[idx][:, idx]
        if flip:
            v, a = v[::-1], a[::-1]
        _HEARTS[key] = Sprite(v.copy(), a.copy())
    return _HEARTS[key]


_HEART_A = None


def fill_heart(out, quad, value):
    """The heart's shape filled in on the quad (screen corners for its picture's
    top left, top right, bottom right, bottom left) - in perspective. (value: a grey,
    or a function of where on the picture (u, v in 0..1) giving the greys.)"""
    global _HEART_A
    if _HEART_A is None:
        from common import sprite
        _HEART_A = sprite("player.png").a
    Q = np.asarray(quad, float)
    x0, y0 = np.floor(Q.min(0)).astype(int)
    x1, y1 = np.ceil(Q.max(0)).astype(int) + 1
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(out.shape[1], x1), min(out.shape[0], y1)
    if x1 <= x0 or y1 <= y0:
        return
    U = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    A = []
    for (u, v), (x, y) in zip(U, Q):
        A.append([u, v, 1, 0, 0, 0, -u * x, -v * x, -x])
        A.append([0, 0, 0, u, v, 1, -u * y, -v * y, -y])
    Hm = np.linalg.svd(np.asarray(A))[2][-1].reshape(3, 3)
    try:
        Hi = np.linalg.inv(Hm)
    except np.linalg.LinAlgError:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1] + .5
    den = Hi[2, 0] * xx + Hi[2, 1] * yy + Hi[2, 2]
    den = np.where(np.abs(den) < 1e-9, 1e-9, den)
    uu = (Hi[0, 0] * xx + Hi[0, 1] * yy + Hi[0, 2]) / den
    vv = (Hi[1, 0] * xx + Hi[1, 1] * yy + Hi[1, 2]) / den
    m = (uu >= 0) & (uu < 1) & (vv >= 0) & (vv < 1)
    if not m.any():
        return
    hh, ww = _HEART_A.shape
    a = np.zeros(m.shape, bool)
    a[m] = _HEART_A[(vv[m] * hh).astype(int), (uu[m] * ww).astype(int)]
    out[y0:y1, x0:x1][a] = value(uu[a], vv[a]) if callable(value) else value


def heart3d(out, P, R, thick, proj, face=228, side=150, dx=0.0):
    """A solid heart at P (16 px across, `thick` deep), turned R (its picture's right,
    down, and the way it faces), put on screen by proj (a 3D point -> x, y, px per
    px - bigger is nearer): slices through its thickness, the far ones first - the
    nearest its face, the rest its side."""
    U, V, N = R[:, 0], R[:, 1], R[:, 2]
    c = proj(P)
    if c is None:
        return
    n = 1 if thick < .3 else max(2, int(math.ceil(thick * c[2])) + 1)
    slices = []
    for j in range(n):
        d = 0.0 if n == 1 else -thick / 2 + thick * j / (n - 1)
        Cj = np.asarray(P, float) + N * d
        q = [proj(Cj + (u - .5) * 16 * U + (v - .5) * 16 * V) for u, v in ((0, 0), (1, 0), (1, 1), (0, 1))]
        if any(qq is None for qq in q):
            return
        slices.append((proj(Cj)[2], [(qq[0] + dx, qq[1]) for qq in q]))
    slices.sort(key=lambda sl: sl[0])
    for j, (_, quad) in enumerate(slices):                 # (its side darker towards the back)
        k = 1.0 if len(slices) < 3 else .72 + .28 * j / (len(slices) - 2)
        fill_heart(out, quad, face if j == len(slices) - 1 else int(side * min(1.0, k)))


def _crown(c, t0, n, speed, up, seed):
    rng = np.random.default_rng(seed)
    return [(c, t0, rng.uniform(0, math.tau), rng.uniform(.5, 1) * speed, rng.uniform(.6, 1) * up,
             rng.uniform(1.5, 3.0)) for _ in range(n)]


_DRIPS = None


def drips():
    global _DRIPS
    if _DRIPS is None:
        _DRIPS = []
        for i, (t0, k) in enumerate(TAPS):
            _DRIPS += _crown(tap_point(i), t0, 9, 90, 210, 4600 + i)
        _DRIPS += _crown(landing(), T_SPLASH, 64, 260, 560, 4690)
    return _DRIPS


def over(out, t, soul_at):
    """The soul (and its reflection), the drops falling in, the drips thrown up."""
    p = cam(t)
    g = 900.0
    im = Image.fromarray(out)
    d = ImageDraw.Draw(im)
    for c, t0, ang, v, up, r in drips():                  # drips thrown up, falling back
        a = t - t0
        if not (0 <= a < 2 * up / g):
            continue
        z = up * a - .5 * g * a * a
        x, y, s = project(p, c[0] + math.cos(ang) * v * a, c[1] + math.sin(ang) * v * a, max(0.0, z))
        rr = max(1.0, r * s)
        rr = min(rr, 5.0)
        d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=215)
    for i, (t0, k) in enumerate(TAPS):                    # each drop falling in on its note
        a = t - (t0 - .2)
        if not (0 <= a < .2):
            continue
        c = tap_point(i)
        z0, z1 = 900 * (1 - (a / .2) ** 2), 900 * (1 - (max(0.0, a - .02) / .2) ** 2)
        x0, y0, s0 = project(p, c[0], c[1], z0)
        x1, y1, _ = project(p, c[0], c[1], z1)
        d.line([(x1, y1), (x0, y0)], fill=170, width=2)
        rr = max(1.5, 3 * s0)
        d.ellipse([x0 - rr, y0 - rr, x0 + rr, y0 + rr], fill=225)
    a = t - T_SPLASH                                       # the break: it goes in - a bright ring racing out
    if 0 <= a < .2:                                        # on the surface, a crown of water thrown up round
        L = landing()                                      # the hole (thin sheets, leaning out, falling back)
        e = 1 - (1 - min(1.0, a / .18)) ** 2
        rr = 10 + 150 * e
        ring = [project(p, L[0] + rr * math.cos(v), L[1] + rr * math.sin(v))[:2] for v in np.linspace(0, math.tau, 48)]
        d.line(ring + ring[:1], fill=int(235 * (1 - a / .2)), width=3)
        rng = np.random.default_rng(4695)
        for j in range(16):
            v = j / 16 * math.tau + rng.uniform(-.15, .15)
            hj = rng.uniform(28, 62) * math.sin(math.pi * min(1.0, a / .2))
            r0 = 12 + 30 * e
            r1 = r0 + rng.uniform(12, 24) * e
            b0 = project(p, L[0] + r0 * math.cos(v), L[1] + r0 * math.sin(v))[:2]
            bm = project(p, L[0] + (r0 + r1) / 2 * math.cos(v), L[1] + (r0 + r1) / 2 * math.sin(v), hj * .75)[:2]
            b1 = project(p, L[0] + r1 * math.cos(v), L[1] + r1 * math.sin(v), hj)[:2]
            d.line([b0, bm], fill=225, width=3)        # (a sheet: thicker at its foot, curling out)
            d.line([bm, b1], fill=225, width=2)
    for c, t0, ang, v, up, r in drips():                  # (the drips over it all)
        b = t - t0
        if t0 != T_SPLASH or not (0 <= b < 2 * up / g):
            continue
        z = up * b - .5 * g * b * b
        x, y, s = project(p, c[0] + math.cos(ang) * v * b, c[1] + math.sin(ang) * v * b, max(0.0, z))
        rr = min(4.0, max(1.0, r * s))
        d.ellipse([x - rr, y - rr, x + rr, y + rr], fill=235)
    out = np.asarray(im).copy()
    if 0 <= a < .1:
        out = np.maximum(out, 70 * (1 - a / .1)).astype(np.uint8)
    if t >= T_SPLASH:
        return out
    lit = 76 + int((228 - 76) * smooth((t - T_SOAK) / SOAK_DUR))   # (it lights up as the ink comes)
    proj = lambda Q: project(p, Q[0], Q[1], Q[2])
    if t < T_TILT:                                         # still on the sheet: the flat heart, on it
        sx, sy = soul_screen(t, soul_at)
        F = unproject(p, sx, sy)
        heart3d(out, (F[0], F[1], 0.0), np.eye(3), 0.0, proj, face=lit)
        return out
    x, y, z = soul3d(t)
    R, thick = soul_turn(t)
    al = 1 - smooth(z / REFL_FADE)                         # its reflection: the same heart, under the surface,
    if z > .5 and al > .02:                                # dimmer, wavering - showing as it comes down near
        mirror = np.diag([1.0, 1.0, -1.0])                 # the pool (high up, it would be off by the middle)
        refl = np.zeros_like(out)
        heart3d(refl, (x, y, -z), mirror @ R, thick, proj, face=105, side=68,
                dx=2.5 * math.sin(t * 23) * min(1.0, z / 60 + .2))
        m = refl > 0
        out[m] = (out[m] * (1 - al) + refl[m] * al).astype(np.uint8)
    heart3d(out, (x, y, z), R, thick, proj, face=lit)
    return out


# ------------------------------------------------------------- under the surface
def under(t):
    """Under the surface: the soul sinking into the dark, the surface far above a
    ceiling of light; its trail of ink and bubbles. Its own little 3D world: y up,
    the surface at y = 0, the soul went in at the origin."""
    a = t - T_SPLASH
    h2, w2 = H // 2, W // 2                               # (the 2 px grain)
    depth = lambda a_: 30 + 230 * (1 - math.exp(-a_ / .45)) + 70 * a_   # (plunging in, then on down, steadily)
    soul = np.array([6 * math.sin(a * 2.1), -depth(a), 4 * math.sin(a * 1.4)])
    u = smooth((t - T_UNDER) / (T_END - T_UNDER))
    phi = math.radians(-90 + 110 * u)                     # (round it, and down below it)
    R = 125 + 75 * u
    hgt = 22 - 80 * u
    eye = soul + np.array([R * math.cos(phi), hgt, R * math.sin(phi)])
    look = soul + np.array([0, 12 - 4 * u, 0])             # (a little above it: the surface over it shows)
    fwd = look - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 1, 0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    f = 300.0                                              # (in half-size px)
    jj, ii = np.mgrid[0:h2, 0:w2]
    dx, dy = (ii - w2 / 2 + .5) / f, -(jj - h2 / 2 + .5) / f
    D = fwd[None, None] + dx[..., None] * right[None, None] + dy[..., None] * up[None, None]
    D /= np.linalg.norm(D, axis=-1, keepdims=True)
    img = np.full((h2, w2), 4.0)
    upw = D[..., 1] > 1e-3
    s = np.where(upw, (0 - eye[1]) / np.maximum(D[..., 1], 1e-3), 0)
    hx, hz = eye[0] + s * D[..., 0], eye[2] + s * D[..., 2]
    wx = hx + 9 * np.sin(hz * .03 + t * 1.3)
    wz = hz + 9 * np.sin(hx * .028 - t * 1.1)
    cau = np.clip(1 - np.abs(np.sin(wx * .06 + t * 1.7) * np.sin(wz * .055 - t * 1.3)) * 4, 0, 1) ** 2   # (a net of light)
    r_in = np.hypot(hx, hz)
    hole = np.exp(-(r_in / 40) ** 2) * math.exp(-a / .5) * 120          # where it went in, still bright
    wave = np.exp(-((r_in - 260 * a) / 18) ** 2) * math.exp(-a / .7) * 70
    fog = np.exp(-s / 520)
    sky = (40 + 120 * cau + hole + wave) * fog
    img = np.where(upw, np.maximum(img, sky), img)
    img += np.clip(D[..., 1], 0, 1) ** 2 * 26              # (the light from above, through the water)
    out = np.clip(img, 0, 255).astype(np.uint8).repeat(2, 0).repeat(2, 1)

    def proj(P):
        v = np.asarray(P, float) - eye
        zc = v @ fwd
        if zc < 4:
            return None
        return W / 2 + (v @ right) / zc * f * 2, H / 2 - (v @ up) / zc * f * 2, f * 2 / zc

    im = Image.fromarray(out)
    d = ImageDraw.Draw(im)
    for st in range(3):                                    # the ink it drags down after it: wisps, wavering
        pts = []
        for b in np.linspace(max(0.0, a - .55), a, 18):
            age = a - b
            sway = (4 + 26 * age) * np.array([math.sin(b * 5 + st * 2.1 + t * 2), 0,
                                              math.cos(b * 4 + st * 1.7 - t * 1.5)])
            P = np.array([6 * math.sin(b * 2.1), -depth(b) + 6, 4 * math.sin(b * 1.4)]) + sway
            q = proj(P)
            if q is not None:
                pts.append((q, age))
        for j in range(len(pts) - 1):
            (q0, age0), (q1, _) = pts[j], pts[j + 1]
            k_ = 1 - age0 / .55                            # (thick and dark by it, thin and fading behind)
            d.line([q0[:2], q1[:2]], fill=int(14 + 52 * k_), width=max(1, int(round(min(5.0, q0[2] * 1.4) * k_))))
    rng = np.random.default_rng(4700)
    for j in range(70):                                    # bubbles: a burst where it went in, then its trail
        tb = T_SPLASH + (rng.uniform(0, .12) if j < 34 else rng.uniform(.05, T_END - T_SPLASH))
        b = t - tb
        if b < 0:
            continue
        ab = tb - T_SPLASH
        o = np.array([6 * math.sin(ab * 2.1), -depth(ab), 4 * math.sin(ab * 1.4)])
        o = o + rng.uniform(-14, 14, 3) * (2.2 if j < 34 else 1)
        vy = rng.uniform(70, 150)
        P = o + np.array([4 * math.sin(b * 7 + j), vy * b, 4 * math.cos(b * 6 + j)])
        if P[1] > -4:
            continue
        q = proj(P)
        if q is None:
            continue
        rr = max(1.0, rng.uniform(1.2, 3.6) * q[2])
        d.ellipse([q[0] - rr, q[1] - rr, q[0] + rr, q[1] + rr], outline=200, width=1 if rr < 4 else 2)
    out = np.asarray(im).copy()
    th = -(phi + math.pi / 2) * .85 + .5 * math.sin(a * 1.7 + .5)   # the soul, a solid heart, rocking as it sinks
    cth, sth = math.cos(th), math.sin(th)                  # (it keeps about facing us as we go round it)
    Ry = np.array([[cth, 0, sth], [0, 1, 0], [-sth, 0, cth]])
    tilt = math.radians(14) * math.sin(a * 2.2)
    Rz = np.array([[math.cos(tilt), -math.sin(tilt), 0], [math.sin(tilt), math.cos(tilt), 0], [0, 0, 1]])
    R = Ry @ Rz @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])   # (its picture's down is down, it faces us)
    heart3d(out, soul, R, 5.0, proj)
    g = 1 - smooth((t - T_UNDER) / .16)                    # (breaking through: the flash going)
    if g > 0:
        out = np.maximum(out, (230 * g)).astype(np.uint8)
    return out


# ------------------------------------------------------------------ the frame
def frame(t, tl, soul_at):
    import scene as S
    global _SOUL_AT
    _SOUL_AT = soul_at
    if t >= T_UNDER:
        return under(t)
    pos, hp, vis = soul_at(int(round(t * 60)))
    flat = S.compose(t, tl, pos, hp, False)                 # (the soul is put in over it, below)
    lp = S.LAST_PLANE
    pad = fx.PAD
    plane = lp[1] if lp is not None and abs(lp[0] - t) < 1e-6 else None
    if plane is None:
        plane = np.zeros((H + 2 * pad, W + 2 * pad), np.uint8)
        plane[pad:pad + H, pad:pad + W] = flat
    plane = soak(plane, t)
    if t >= T_DROP - .3:
        plane = ripples(plane, t, pad)
    p = cam(t)
    out = fx.warp(flat, p, plane)
    if fx.is_identity(p):
        out = plane[pad:pad + H, pad:pad + W].copy()
    out = over(out, t, soul_at)
    g = smooth((t - (T_DIVE - .1)) / .1)                   # (going under: the surface rushes up, white)
    if g > 0:
        out = np.maximum(out, 230 * g).astype(np.uint8)
    return out
