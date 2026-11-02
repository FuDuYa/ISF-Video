"""M49-M55: the rest after the soul has gone under (and sunk a while) - then (M55, dive.py)
it starts up, through the attacks, to break out of the surface on the last break (M79).
Calm: three long shots (cut on the break at M51 and at M53 - going dark, coming up
again; so in from under the surface): in each
the soul - huge, a little of it off the screen, lying back on a slant, a flat pixel
heart - with the camera going slowly round it; about it the still dark of the ink,
light coming down from far above, motes and a few bubbles drifting up. In the second
(from the break) it sinks, from rest, ever faster - the camera going down with it,
falling behind a little; the motes rushing up past it, bubbles going up off it, the
light going as it goes deeper. The notes are
only felt: a faint ring going out from it, it brightening a little (the break more).
In the last it falls, upright, straight down through the screen, its heart beginning
to glow; from the last notes (M54) a faint red shows in it - blood, the one pigment it
has of its own, the only colour in all this: an omen, only - beating with its heart,
breathing out a red mist into the ink; its glow lighting the dark round it. (leap_frame:
it breaking out of the surface, red - dive.py's end, on the last break.)

Its own little 3D world: the soul at the origin (x right, y down, z away from the camera
at the start); the camera goes round it (yaw, pitch, roll), at a distance, its view
shifted so the soul sits off the middle."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

import choreo as C
import plunge
from common import W, H, clamp, smooth

T0, T_END = C.tm(49), C.tm(55)
DIP_OUT, DIP_IN = .15, .25        # s: at a cut, what was there goes dark, then the new shot comes up
F = 1100.0                        # px: the focal length (long: it keeps its shape, close up)
NEAR = .5
LIGHT = np.array([-.45, -.65, -.6]) / np.linalg.norm([-.45, -.65, -.6])   # (from up left, our side)
SINK_A = 26.0                     # units/s/s: in the second shot it sinks, ever faster
DRIFT = 7.0                       # units/s: in the first it sinks too - slowly, steadily
SINK_LAG = 150                    # px: (the camera falling behind it: it slips down the screen)
FALL_V = (16.0, 5.0)              # units/s: (the last shot: the camera going down too, slowing - the ink going up past it)
NOTES = [(C.tm(49), 1.0), (C.tm(50), 1.0), (C.tm(50.5), .6), (C.tm(51), 2.2),
         (C.tm(52), 1.0), (C.tm(52.5), .6), (C.tm(53), 1.0)]                # (when, how strong)
BREAK = C.tm(51)
GLOW0, BURST = C.tm(53), C.tm(54)    # (the fall: it glows, from here; the red omen shows in it, from here)
SLIDE = (C.tm(54.25), C.tm(54.875))  # (the slide: the star moves a beat after its tap)
UP = C.tm(79)                        # (it breaks out of the surface: dive.py)
BEATS = [b for b in C._CHART_T if BURST <= b < C.tm(81)]   # (its heart beats on every note from the omen on,
                                                          # through the dive (dive.py) to the chorus)
RISE = 140.0                         # units: how far it comes up M58-M62 (from rest, ever faster)
SURFACE = 620.0                      # units: how far under the surface it is at M58
RED = np.array([255.0, 38.0, 48.0])
FALL = (-7.3, 17.4, -5.6)            # units: (the last shot: y = a + b u + c u u - it falls from up off the
                                     # screen, through the middle on the burst, slowing, low in it by M55)

# the shots: from, to; the soul's slant on screen, how far it lies back, its turn
# (deg); the camera's yaw, pitch, roll (deg), its distance, and where the soul sits off
# the middle (px; off along its tip's way, so its two lobes stay in) - each (start, end);
# sink: it sinks, ever faster; fall: it falls straight down the screen, glowing; rise: it starts up
# to the surface, ever faster; leap: it breaks out of it (the chorus)
SHOTS = [
    dict(t0=T0, t1=BREAK, slant=-30, lie=40, turn=12, yaw=(-10, 2), pitch=(-4, -1.3), roll=(0, -2), drift=True,
         dist=(28.0, 26.5), shift=((68, 117), (63, 109))),
    dict(t0=BREAK, t1=C.tm(53), slant=32, lie=44, turn=-14, yaw=(14, 2), pitch=(7, 3), roll=(0, 3),
         dist=(29.0, 26.6), shift=((-180, -25), (-163, -17)), sink=True),
    dict(t0=C.tm(53), t1=C.tm(55), slant=0, lie=28, turn=0, yaw=(-6, 4), pitch=(-2, 1), roll=(0, 1),
         dist=(40.0, 39.0), shift=((0, 0), (0, 0)), fall=True),
]
LEAP = dict(slant=0, lie=26, turn=0, yaw=(-4, 3), pitch=(24, 8), roll=(0, 0), dist=(46.0, 290.0),
            shift=((0, 60), (0, 30)), leap=True)     # (it breaking out of the surface, up into the air, the
                                                     # camera drawing back: dive.py, on M79)
LEAP_H = (46.0, .42)                 # units, s: (how high it leaps: 46 (1 - e^(-a/.42)) - out at once, then hanging)
LEAP_TURN = (38.0, 16.0, -10.0)      # deg: (as it goes up it turns - round, over, aslant - to this, and stays)
FREEZE_V = 75.0                      # units/s: (the red ring going out from where it came up - the ink setting behind it)


def leap_h(a):
    return LEAP_H[0] * (1 - math.exp(-max(0.0, a) / LEAP_H[1]))


def active(t: float) -> bool:
    return T0 - DIP_OUT <= t < T_END


def sink(s, t):
    """How far it has sunk in this shot, and how fast it is going (the second shot:
    from rest, ever faster; rising: the other way, from rest, ever faster; the others:
    it hangs)."""
    if s.get("rise"):
        T = s["t1"] - s["t0"]
        u = clamp((t - s["t0"]) / T)
        return -RISE * u ** 2.4, -RISE * 2.4 * u ** 1.4 / T
    if s.get("fall"):                                        # (the camera going down with it, slowing)
        T = s["t1"] - s["t0"]
        tau = clamp(t - s["t0"], 0, T)
        a, b = FALL_V
        return a * tau + .5 * (b - a) * tau * tau / T, a + (b - a) * tau / T
    if s.get("drift"):                                       # (the first: going down, slowly, steadily)
        tau = clamp(t - s["t0"], 0, s["t1"] - s["t0"])
        return DRIFT * tau, DRIFT
    if not s.get("sink"):
        return 0.0, 0.0
    tau = clamp(t - s["t0"], 0, s["t1"] - s["t0"])
    return .5 * SINK_A * tau * tau, SINK_A * tau


def depth_light(t):
    """How much of the light from above gets down to it: less as it sinks in the
    second shot; so it stays, deeper."""
    s = SHOTS[1]
    return 1 - .55 * clamp((t - s["t0"]) / (s["t1"] - s["t0"])) ** 2


def heartbeat(t):
    """Its heart beating, from the omen on: a beat on every note, dying away."""
    h = 0.0
    for b in BEATS:
        if b > t:
            break
        h = max(h, math.exp(-(t - b) / .14))
    return h


def omen(t):
    """How red its heart shows: an omen, not yet all red - growing to the leap."""
    return .34 + .26 * clamp((t - BURST) / (UP - BURST)) ** 1.2


# ------------------------------------------------------------------ 3D helpers
def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class Cam:
    """The shot's camera at t: going slowly round the soul."""

    def __init__(self, s, t):
        u = clamp((t - s["t0"]) / (s["t1"] - s["t0"]))
        lerp = lambda ab, e=u: ab[0] + (ab[1] - ab[0]) * e
        e_in = 1 - (1 - u) ** 2.5 if s.get("leap") else u    # (the leap: drawing back fast, then slowly)
        self.tgt = np.zeros(3)                               # (what it looks at)
        self.R = (_rz(math.radians(lerp(s["roll"]))) @ _rx(math.radians(lerp(s["pitch"]))) @
                  _ry(math.radians(lerp(s["yaw"]))))
        self.d = lerp(s["dist"], e_in)
        (ax, ay), (bx, by) = s["shift"]
        self.sx, self.sy = ax + (bx - ax) * e_in, ay + (by - ay) * e_in
        if s.get("sink"):
            self.sy += SINK_LAG * u * u

    def to_cam(self, P):
        return self.R @ (np.asarray(P, float) - self.tgt) + [0.0, 0.0, self.d]

    def pr(self, Pc):
        """Camera point -> (screen x, y, px per unit), or None behind us."""
        if Pc[2] < NEAR:
            return None
        k = F / Pc[2]
        return W / 2 + self.sx + Pc[0] * k, H / 2 + self.sy + Pc[1] * k, k


# ------------------------------------------------------------------ the soul
def soul_pose(s, t, cam):
    """(its centre in the camera's frame, its turn there): floating, turning a little;
    in the last shot, falling straight down (slowing as it goes)."""
    Rh = (_rz(math.radians(s["slant"])) @ _rx(-math.radians(s["lie"])) @
          _ry(math.radians(s["turn"] + 4 * math.sin(t * .6))) @ np.diag([1, 1, -1.0]))
    y = .3 * math.sin(t * 1.1)
    if s.get("fall"):
        u = clamp((t - s["t0"]) / (s["t1"] - s["t0"]))
        y = FALL[0] + FALL[1] * u + FALL[2] * u * u
    c = cam.to_cam([0.0, y, 0.0])
    return c, cam.R @ Rh


def glow(t):
    """How much the notes light it up just now."""
    g = 0.0
    for tn, k in NOTES:
        if t >= tn:
            g += 14 * k * math.exp(-(t - tn) / .4)
    return g


def glint(s, t):
    """Where the glint across its face is (along its diagonal; off it: none): once,
    slowly, through the first two shots (the last: it glows instead)."""
    if t >= GLOW0:
        return -9.0
    return -.3 + 2.6 * clamp((t - s["t0"]) / (s["t1"] - s["t0"]))


def big_heart(out, P, R, cam, t, s):
    """The soul, a flat pixel heart, close, in perspective: lit (the light from up
    left), the glint going across it; from the fall on its heart glowing, beating."""
    beat = 1 + .025 * heartbeat(t)                           # (it swells a little on each beat)
    U, V, N = R[:, 0] * beat, R[:, 1] * beat, R[:, 2]
    lam = max(0.0, N @ -LIGHT)
    gl, lift = glint(s, t), glow(t)
    g = core_glow(t, s)
    q = [cam.pr(np.asarray(P, float) + (u - .5) * 16 * U + (v - .5) * 16 * V) for u, v in ((0, 0), (1, 0), (1, 1), (0, 1))]
    if any(qq is None for qq in q):
        return None
    dim = 30 * g                                             # (round its glowing heart it dims a little: it shows)

    def face(uu, vv):
        v = 178 - dim + 50 * lam + lift + 16 * (.5 - vv) + 8 * (.5 - uu)
        v = v + 30 * np.exp(-((uu * .7 + vv * .7 - gl) / .1) ** 2)
        if g > 0:
            sg = .1 + .14 * g
            v = v + (dim + 130 * g) * np.exp(-((uu - .5) ** 2 + (vv - .55) ** 2) / (2 * sg * sg))
        return np.clip(v, 0, 255).astype(np.uint8)

    quad = [qq[:2] for qq in q]
    plunge.fill_heart(out, quad, face)
    return quad


def core_glow(t, s):
    """How much its heart glows: in the fall slowly coming, to the omen; then full, and more on each beat."""
    if s.get("leap"):
        return 1.0
    if t >= BURST:
        return 1 + .3 * heartbeat(t)
    if s.get("fall"):
        return smooth(clamp((t - GLOW0) / (BURST - GLOW0))) ** 1.3
    return 0.0


# ------------------------------------------------------------------ about it
_GRID = None


def _grid():
    global _GRID
    if _GRID is None:
        Y, X = np.mgrid[0:H // 2, 0:W // 2].astype(np.float32) * 2
        r = np.hypot((X - W / 2) / W, (Y - H / 2) / H)
        top = (1 - Y / H) ** 1.4                             # (lighter up towards the light)
        _GRID = (X, Y, top, np.exp(-(r / .4) ** 2))
    return _GRID


def dark(t, roll, sunk=0.0):
    """The still dark of the ink: lighter up above, and light coming down through it
    in slow soft shafts, leaning, swaying (less of it, deeper down); going down, layers
    of the ink - faint, soft, level - going up past it."""
    X, Y, top, vig = _grid()
    L = depth_light(t)
    lean = math.tan(math.radians(16 + roll))
    xs = X + (Y - H) * lean
    sh = np.zeros_like(X)
    for k, (x0, w, a, f) in enumerate(((60, 26, 10, .21), (190, 40, 13, .17), (330, 22, 8, .26),
                                       (450, 50, 12, .13), (590, 30, 9, .19), (720, 36, 11, .15))):
        xc = x0 + 30 * math.sin(t * f + k * 1.3)
        sh += a * (.65 + .35 * math.sin(t * f * 1.7 + k)) * np.exp(-((xs - xc) / w) ** 2)
    img = 5 + 15 * L * top + 6 * vig + L * sh * (1 - Y / H) ** 1.2
    if sunk:
        yy = Y + X * math.tan(math.radians(roll)) + sunk * 9      # (far off: 9 px a unit)
        img = img + (5 * np.sin(yy / 46) ** 8 + 3 * np.sin(yy / 17 + 1.3) ** 6) * (.6 + .4 * top)
    return np.clip(img, 0, 255).astype(np.uint8).repeat(2, 0).repeat(2, 1)


_MOTES = None


MOTE_D = 32.0                     # (the specks are strewn through what the camera sees from about this far)


def motes():
    """Specks in the ink round it (and a few bubbles): how far from the camera, where
    across and up that view (as slopes: the lens is long, the view narrow), how big."""
    global _MOTES
    if _MOTES is None:
        rng = np.random.default_rng(4810)
        n = 230
        _MOTES = dict(zc=3 + 280 * rng.random(n) ** 1.6, ax=rng.uniform(-.5, .5, n), ay=rng.uniform(-.4, .4, n),
                      ph=rng.uniform(0, math.tau, n), sp=rng.uniform(.3, .8, n), r=rng.uniform(.08, .2, n),
                      bub=rng.random(n) < .1)
        k = 220                                              # (and more, only going down: rushing up past it)
        rush = dict(zc=6 + 120 * rng.random(k) ** 1.3, ax=rng.uniform(-.5, .5, k), ay=rng.uniform(-.4, .4, k),
                    ph=rng.uniform(0, math.tau, k), sp=rng.uniform(.3, .8, k), r=rng.uniform(.1, .25, k),
                    bub=rng.random(k) < .15)
        _MOTES = {key: np.concatenate([_MOTES[key], rush[key]]) for key in _MOTES}
        _MOTES["rush"] = np.arange(n + k) >= n
    return _MOTES


def draw_motes(layer, cam, t, front, zcut, sunk=0.0, speed=0.0):
    """The specks drifting up (front: the ones nearer than the soul): far ones small and
    dim, near ones big soft discs; as it sinks they go up past it, streaking."""
    m = motes()
    d = ImageDraw.Draw(layer)
    for i in range(len(m["zc"])):
        if m["rush"][i] and speed < 3:
            continue
        zc, z = m["zc"][i], m["zc"][i] - MOTE_D
        sp = m["sp"][i] * (2.2 if m["bub"][i] else 1)
        Y = .4 * zc                                          # (they go round, top to bottom of the view)
        y = (m["ay"][i] * zc - sp * (t - T0) - sunk + Y) % (2 * Y) - Y
        x = m["ax"][i] * zc + .3 * math.sin(t * .4 + m["ph"][i])
        Pc = cam.to_cam([x, y, z])
        if Pc[2] < 1.5 or (Pc[2] < zcut) != front:
            continue
        sx, sy, k = cam.pr(Pc)
        r = clamp(m["r"][i] * k * .35, .8, 16)
        v = 120 * math.exp(-Pc[2] / 200) * min(1.0, 3.5 / r) * (.5 if front else 1)   # (dimmer over it)
        tail = cam.pr(cam.to_cam([x, y + speed * .08, z]))       # (where it was a moment ago)
        if tail is not None and math.hypot(tail[0] - sx, tail[1] - sy) > 2 * r:
            wd = int(clamp(round(r * .8), 1, 4))             # (bright at its head, going up; fading behind)
            vv = min(255, v * (1.3 if not front else .8))
            for j, f in enumerate((1.0, .55, .25)):
                a0, a1 = j / 3, (j + 1) / 3
                d.line([(sx + (tail[0] - sx) * a0, sy + (tail[1] - sy) * a0),
                        (sx + (tail[0] - sx) * a1, sy + (tail[1] - sy) * a1)], fill=int(vv * f), width=wd)
        elif m["bub"][i]:
            d.ellipse([sx - r, sy - r, sx + r, sy + r], outline=int(min(255, v * 1.5)), width=1)
        else:
            d.ellipse([sx - r * 1.6, sy - r * 1.6, sx + r * 1.6, sy + r * 1.6], fill=int(v * .35))   # (soft)
            d.ellipse([sx - r, sy - r, sx + r, sy + r], fill=int(v))


_TRAIL = None


def trail():
    """Bubbles shed as it sinks: (how far it had sunk then, where on it - its own
    plane - how far in front or behind it, how big, how fast it rises)."""
    global _TRAIL
    if _TRAIL is None:
        rng = np.random.default_rng(4812)
        _TRAIL = [(.6 + .7 * k, rng.uniform(-7, 7, 2), rng.uniform(-1.5, 1.5), rng.uniform(.25, .6),
                   rng.uniform(2, 5)) for k in range(80)]
    return _TRAIL


def draw_trail(layer, cam, s, t, Rh, front, zcut):
    """The bubbles going up off it as it sinks (it going down past them)."""
    if s.get("fall"):
        return draw_fall_trail(layer, cam, s, t, Rh, front, zcut)
    if not (s.get("sink") or s.get("drift")):
        return
    sunk, _ = sink(s, t)
    d = ImageDraw.Draw(layer)
    for d0, (u, v), w, r, rise in trail():
        if d0 > sunk:
            continue
        t0 = s["t0"] + (d0 / DRIFT if s.get("drift") else math.sqrt(2 * d0 / SINK_A))
        P = Rh @ np.array([u, v, w]) + [0.0, -(sunk - d0) - rise * (t - t0), 0.0]
        Pc = cam.to_cam(P)
        if Pc[2] < 1.5 or (Pc[2] < zcut) != front:
            continue
        sx, sy, k = cam.pr(Pc)
        rr = max(1.5, r * k * .5)
        g = clamp((t - t0) / .15)                            # (coming out of it)
        d.ellipse([sx - rr, sy - rr, sx + rr, sy + rr], outline=int(150 * g), width=2 if rr > 6 else 1)


def draw_fall_trail(layer, cam, s, t, Rh, front, zcut):
    """The last shot: bubbles shed as it falls, left behind - going up (the camera going
    down too, slower)."""
    d = ImageDraw.Draw(layer)
    T = s["t1"] - s["t0"]
    sunk, _ = sink(s, t)
    for k, (_, (u, v), w, r, rise) in enumerate(trail()):
        t0 = s["t0"] + .06 * k
        if t0 > t:
            break
        u0 = (t0 - s["t0"]) / T
        y0 = FALL[0] + FALL[1] * u0 + FALL[2] * u0 * u0
        P = Rh @ np.array([u, v, w]) + [0.0, y0 - (sunk - sink(s, t0)[0]) - rise * (t - t0), 0.0]
        Pc = cam.to_cam(P)
        if Pc[2] < 1.5 or (Pc[2] < zcut) != front:
            continue
        sx, sy, kk = cam.pr(Pc)
        rr = max(1.5, r * kk * .5)
        g = clamp((t - t0) / .15) * (1 - clamp((t - t0 - 1.6) / .4))
        d.ellipse([sx - rr, sy - rr, sx + rr, sy + rr], outline=int(150 * g), width=2 if rr > 6 else 1)


def rings(out, P, cam, t):
    """A faint ring going out from it on each note (the break: brighter, two, and a
    soft glow round it)."""
    c = cam.pr(P)
    if c is None:
        return out
    im = Image.fromarray(out)
    d = ImageDraw.Draw(im)
    for tn, k in NOTES:
        a = t - tn
        if not 0 <= a < 1.8:
            continue
        for j in range(2 if k > 2 else 1):
            b = a - .18 * j
            if b < 0:
                continue
            r = c[2] * (9 + 30 * (1 - math.exp(-b / .7)))
            v = int(min(255, 60 * k * (1 - b / 1.8) ** 1.6))
            if v >= 3:
                d.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], outline=v, width=2)
    out = np.asarray(im).copy()
    a = t - BREAK
    if 0 <= a < 1.2:
        X, Y = _grid()[:2]
        rr = np.hypot(X - c[0], Y - c[1]) / (c[2] * 14)
        halo = (30 * math.exp(-a / .5) * np.exp(-rr ** 2)).repeat(2, 0).repeat(2, 1)
        out = np.clip(out + halo, 0, 255).astype(np.uint8)
    return out


# ------------------------------------------------------------------ the red
def omen_red(t, s, c, quad):
    """How red each pixel is (0..1), the omen: its heart faintly red, beating; a faint
    red mist breathed out on each beat, drifting off, fading (blood in the ink); on the
    slide (M54) a faint red wisp going round it."""
    if t < BURST or c is None or quad is None:
        return None
    k = c[2] * 16                                            # (px: how big it is)
    cx, cy = c[0], c[1]
    lay = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(lay)
    for i, b in enumerate(BEATS):                            # the mist
        a = t - b
        if a < 0:
            break
        if a >= 1.6:
            continue
        rng = np.random.default_rng(4830 + i)
        ang, v = rng.uniform(0, math.tau), rng.uniform(18, 40)
        x, y = cx + math.cos(ang) * v * a, cy + math.sin(ang) * v * a - 10 * a
        r = k * (.03 + .07 * a / 1.6)
        al = 1.1 * omen(b) * (1 - a / 1.6) ** 1.5 * clamp(a / .2)   # (breathed out: not a dot at its heart)
        for f, rr in ((.35, 1.0), (.65, .7), (1.0, .42)):    # (soft)
            d.ellipse([x - r * rr, y - r * rr, x + r * rr, y + r * rr], fill=int(255 * al * f))
    if s.get("fall") and SLIDE[0] <= t < SLIDE[1] + .2:      # the slide: a faint wisp round it
        head = smooth(clamp((t - SLIDE[0]) / (SLIDE[1] - SLIDE[0])))
        pts = []
        for i in range(26):
            s_ = head - .35 * i / 25
            if s_ < 0:
                break
            th = math.radians(67.5 + 225 * s_)               # (from key 2, clockwise round to key 7)
            rr = k * .36 * (1 - .18 * math.sin(math.pi * s_))
            fade = 1 - clamp((t - SLIDE[1]) / .2)
            pts.append(((cx + rr * math.sin(th), cy - rr * math.cos(th)), (1 - i / 25) * fade))
        for (p0, a0), (p1, _) in zip(pts, pts[1:]):
            d.line([p0, p1], fill=int(45 * a0), width=max(1, int(round(10 * a0))))
        for (p0, a0), (p1, _) in zip(pts, pts[1:]):
            d.line([p0, p1], fill=int(110 * a0), width=max(1, int(round(4 * a0))))
    out = np.asarray(lay, np.float32) / 255
    core = np.zeros((H, W), np.uint8)                        # its heart glowing red, beating - white hot
    hb = heartbeat(t)                                        # at the very middle, red round that
    g = omen(t) * (.6 + .4 * hb) * min(1.0, (t - BURST) / .1)
    sg = .15 + .05 * hb

    def red_core(uu, vv):
        r2 = (uu - .5) ** 2 + (vv - .55) ** 2
        a = min(1.0, 1.3 * g) * np.exp(-r2 / (2 * sg * sg)) * (1 - .6 * np.exp(-r2 / (2 * (.4 * sg) ** 2)))
        return (255 * a).astype(np.uint8)

    plunge.fill_heart(core, quad, red_core)
    face = np.zeros((H, W), np.uint8)
    plunge.fill_heart(face, quad, 255)
    X, Y = _grid()[:2]                                       # (and the dark round it lit red - not its face)
    rr = np.hypot(X - cx, Y - cy) / (k * (.85 + .15 * hb))
    glo = (.55 * g * np.exp(-rr ** 2)).astype(np.float32).repeat(2, 0).repeat(2, 1)
    glo[face > 0] = 0
    return np.maximum(np.maximum(out, core / 255.0), glo)


def tint(grey, red):
    """The grey frame with the red laid over it (RGB)."""
    rgb = np.repeat(grey[..., None].astype(np.float32), 3, 2)
    if red is not None:
        rgb = rgb * (1 - red[..., None]) + RED * red[..., None]
    return np.clip(rgb, 0, 255).astype(np.uint8)


def halo(out, c, t, s):
    """The dark round it lit by its glowing heart."""
    g = core_glow(t, s)
    if c is None or g <= 0:
        return out
    X, Y = _grid()[:2]
    rr = np.hypot(X - c[0], Y - c[1]) / (c[2] * 16)
    h = (g * (85 * np.exp(-(rr / .7) ** 2) + 30 * np.exp(-(rr / 1.8) ** 2))).repeat(2, 0).repeat(2, 1)
    return np.clip(out + h, 0, 255).astype(np.uint8)


def _rays(cam, X, Y):
    """The ray through each pixel (X, Y: its centre), in the world: (camera's place, x, y, z of the way)."""
    Rt = cam.R.T
    dx, dy = (X - W / 2 - cam.sx) / F, (Y - H / 2 - cam.sy) / F
    w = [Rt[i, 0] * dx + Rt[i, 1] * dy + Rt[i, 2] for i in range(3)]
    return cam.tgt - Rt @ np.array([0.0, 0.0, cam.d]), w


def ceiling(cam, t, above):
    """Looking up as it rises: the surface over it, `above` up - a net of wavering light,
    small and dim far up, coming down bigger and brighter (half size)."""
    X, Y = _grid()[:2]
    o, (wx, wy, wz) = _rays(cam, X, Y)
    up = wy < -1e-3
    sd = np.where(up, (-above - o[1]) / np.minimum(wy, -1e-3), 0)
    hx, hz = o[0] + sd * wx, o[2] + sd * wz
    hx2, hz2 = hx + 3 * np.sin(hz * .09 + t * 1.3), hz + 3 * np.sin(hx * .085 - t * 1.1)
    cau = np.clip(1 - np.abs(np.sin(hx2 * .2 + t * 1.7) * np.sin(hz2 * .18 - t * 1.3)) * 4, 0, 1) ** 2
    fog = np.exp(-sd * np.sqrt(wx * wx + wy * wy + wz * wz) / 300)
    return np.where(up, (18 + 120 * cau) * fog, 0)


def leap_frame(s, t):
    """The chorus: it breaks out of the surface - seen from over the pool: it leaps up out
    of the ink, red (its own colour back), a splash round it, its blood flung out on each
    note, rings going out on the pool, red. (The world: y down, the surface y = 0, it
    comes up at the origin; the parts of it still under the ink are hidden.)"""
    a = t - s["t0"]
    pre = a < 0                                              # (before it leaps: its light under the surface)
    cam = Cam(s, t)
    h = leap_h(a) if not pre else -14.0                      # units: how high it has leapt
    cam.tgt = np.array([0.0, -.45 * max(0.0, h), 0.0])      # (the camera going up after it, some of the way)
    X, Y = _grid()[:2]
    o, (wx, wy, wz) = _rays(cam, X, Y)
    down = wy > 1e-3
    sd = np.where(down, -o[1] / np.maximum(wy, 1e-3), 1e4)
    hx, hz = o[0] + sd * wx, o[2] + sd * wz
    fog = np.exp(-sd * np.sqrt(wx * wx + wy * wy + wz * wz) / 420)
    bend = 2.5 * np.sin(hx * .13 + .7) + 1.5 * np.sin(hx * .31 + 2.1)   # (the pool: the plunge's swells)
    r = np.hypot(hx, hz)
    R0 = 7 + FREEZE_V * max(0.0, a)                          # (where the red ring has got to: in it, the ink has set -
    tt = t if pre else np.where(r < R0, s["t0"] + (r - 7) / FREEZE_V, t)   # its swells stopped as it went by)
    lines = (np.clip((np.sin((hz + bend) * .55 + tt * 2.2) - .82) / .18, 0, 1) * 30 +
             np.clip((np.sin((hz * .9 - hx * .2 + bend) * .35 - tt * 1.5) - .9) / .1, 0, 1) * 18)
    pool = 9 + 34 * (1 - fog) + lines * (.35 + .65 * fog)
    ring = np.zeros_like(r)
    red = np.zeros_like(r)
    if not pre:
        for j in range(1, 3):                                # (rings going out from where it came up)
            ring += np.exp(-((r - (7 + 70 * a - 7 * j)) / 1.1) ** 2) * max(0.0, 1 - a / .6)
        front = np.exp(-((r - R0) / (1.4 + .02 * R0)) ** 2) * (1 - .55 * clamp(a / (s["t1"] - s["t0"])))
        ring += front                                        # (the red ring, going on out)
        set_ = (r < R0) * np.clip((R0 - r) / 6, 0, 1)        # (set: dark red, glossy - its last swells caught in it)
        pool = pool + set_ * (6 + lines * .25)
        red = np.clip(ring * fog, 0, 1) * .8 + set_ * fog * .22
    else:                                                    # its light under the surface, growing, beating;
        g = smooth(clamp((t - s["pre"]) / (s["t0"] - s["pre"]))) * (.55 + .45 * heartbeat(t))
        lit = np.exp(-(r / (5 + 9 * g)) ** 2) * g            # (a faint ring going out on each beat)
        for b in BEATS:
            if b > t:
                break
            if t - b < .8:
                ring += np.exp(-((r - (6 + 40 * (t - b))) / 1.5) ** 2) * (1 - (t - b) / .8) * .5 * g
        pool = pool + 150 * lit
        red = np.clip(lit, 0, 1) * .45
    pool = np.where(down, pool + 60 * ring * fog, 3)
    out = np.clip(pool, 0, 255).astype(np.uint8).repeat(2, 0).repeat(2, 1)
    red = np.where(down, red, 0).astype(np.float32).repeat(2, 0).repeat(2, 1)
    tw = smooth(clamp(a / .9)) if not pre else 0.0          # (turning as it goes up)
    yaw, over, tilt = (v * tw for v in LEAP_TURN)
    Rh = (_rz(math.radians(s["slant"] + tilt)) @ _rx(-math.radians(s["lie"] + over)) @
          _ry(math.radians(s["turn"] + yaw)) @ np.diag([1, 1, -1.0]))
    P, R = cam.to_cam([0.0, -h, 0.0]), cam.R @ Rh
    hl = np.zeros((H, W), np.uint8)
    quad = big_heart(hl, P, R, cam, t, s)
    if quad is not None:                                     # its glow on the pool round it, red
        c = cam.pr(P + R[:, 1] * .05 * 16)
        out = halo(out, c, t, s)
        rr = np.hypot(X - c[0], Y - c[1]) / (c[2] * 16)
        glo = (.42 * (.75 + .25 * heartbeat(t)) * np.exp(-(rr / .65) ** 2)).repeat(2, 0).repeat(2, 1)
        red = np.maximum(red, glo.astype(np.float32))
    if quad is not None:                                     # (under the surface: hidden)
        Yf, Xf = np.mgrid[0:H, 0:W].astype(np.float32) + .5
        d = np.stack([(Xf - W / 2 - cam.sx) / F, (Yf - H / 2 - cam.sy) / F, np.ones_like(Xf)], -1)
        N, nc = R[:, 2], cam.R @ np.array([0.0, 1.0, 0.0])
        s_h = (P @ N) / np.where(np.abs(d @ N) < 1e-9, 1e-9, d @ N)
        dn = d @ nc
        s_s = (nc[2] * cam.d - cam.tgt[1]) / np.where(np.abs(dn) < 1e-9, 1e-9, dn)
        vis = (hl > 0) & ~((s_s > 0) & (s_s < s_h))
        out[vis] = hl[vis]
        red[vis] = np.maximum(red[vis], .8)                  # (it is red again)
    splash, blood = Image.new("L", (W, H), 0), Image.new("L", (W, H), 0)
    d1, d2 = ImageDraw.Draw(splash), ImageDraw.Draw(blood)
    g = 90.0                                                 # units/s/s
    rng = np.random.default_rng(4840)
    for j in range(16):                                      # the crown: thin sheets thrown up round it
        ang = j / 16 * math.tau + rng.uniform(-.15, .15)
        hj = rng.uniform(5, 10) * math.sin(math.pi * clamp(a / .45))
        r0, r1 = 7 + 6 * a, 9 + 14 * a
        b0 = cam.pr(cam.to_cam([r0 * math.cos(ang), 0.0, r0 * math.sin(ang)]))
        b1 = cam.pr(cam.to_cam([r1 * math.cos(ang), -hj, r1 * math.sin(ang)]))
        if b0 and b1 and hj > .3:
            d1.line([b0[:2], b1[:2]], fill=215, width=2)
    for j in range(40):                                      # drops of ink thrown up (light), falling back
        ang, v, up = rng.uniform(0, math.tau), rng.uniform(6, 22), rng.uniform(20, 48)
        r0 = rng.uniform(4, 8)
        y = -(up * a - .5 * g * a * a)
        if y > 0:
            continue
        q = cam.pr(cam.to_cam([(r0 + v * a) * math.cos(ang), y, (r0 + v * a) * math.sin(ang)]))
        if q:
            rr = max(1.0, .25 * q[2])
            d1.ellipse([q[0] - rr, q[1] - rr, q[0] + rr, q[1] + rr], fill=225)
    for j in range(36):                                      # the ink running off it as it goes up, falling back
        te = s["t0"] + .03 + .045 * j
        if te > t:
            break
        rj = np.random.default_rng(4845 + j)
        x0, z0 = rj.uniform(-6, 6), rj.uniform(-2, 2)
        y = -(leap_h(te - s["t0"]) - 7) + .5 * g * (t - te) ** 2
        if y > 0:
            continue
        q = cam.pr(cam.to_cam([x0, y, z0]))
        if q:
            rr = max(1.0, rj.uniform(.2, .35) * q[2])
            d1.ellipse([q[0] - rr, q[1] - rr * 1.6, q[0] + rr, q[1] + rr * 1.6], fill=200)
    for i, b in enumerate([b for b in BEATS if b >= s["t0"]]):   # its blood, flung out on each note
        ab = t - b
        if ab < 0:
            break
        rb = np.random.default_rng(4850 + i)
        n = 46 if i == 0 else 16
        hb = leap_h(b - s["t0"])
        for j in range(n):
            ang, v, up = rb.uniform(0, math.tau), rb.uniform(8, 30), rb.uniform(8, 40)
            life = rb.uniform(.35, .6)
            if ab >= life:
                continue
            y = -hb - (up * ab - .5 * g * ab * ab)
            q = cam.pr(cam.to_cam([v * ab * math.cos(ang), y, v * ab * math.sin(ang) * .6]))
            if q:
                rr = max(1.5, rb.uniform(.25, .5) * q[2])
                d2.ellipse([q[0] - rr, q[1] - rr, q[0] + rr, q[1] + rr], fill=int(255 * (1 - ab / life) ** .7))
    bl, sp = np.asarray(blood, np.float32) / 255, np.asarray(splash)
    if quad is not None:                                     # (the splash and the blood round it, not over it)
        bl, sp = np.where(vis, 0, bl), np.where(vis, 0, sp)
    red = np.maximum(red, bl)                                # (drawn apart: so as not to undo the red under)
    out = tint(out, red)
    return np.maximum(out, sp[..., None])


# ------------------------------------------------------------------ the frame
def shot_frame(s, t):
    cam = Cam(s, t)
    out = dark(t, s["roll"][0], sink(s, t)[0])
    if s.get("rise"):                                        # (the surface over it, coming down)
        sunk, _ = sink(s, t)
        sky = ceiling(cam, t, SURFACE + sunk).repeat(2, 0).repeat(2, 1)
        out = np.clip(out + sky, 0, 255).astype(np.uint8)
    P, R = soul_pose(s, t, cam)
    Rh = cam.R.T @ R
    sunk, speed = sink(s, t)
    back = Image.new("L", (W, H), 0)
    draw_motes(back, cam, t, False, P[2] - 3, sunk, speed)
    draw_trail(back, cam, s, t, Rh, False, P[2])
    out = np.clip(out.astype(np.int16) + np.asarray(back), 0, 255).astype(np.uint8)
    out = rings(out, P, cam, t)
    c = cam.pr(P + R[:, 1] * .05 * 16)                     # (its heart: a touch below its middle)
    if t >= GLOW0:
        out = halo(out, c, t, s)
    quad = big_heart(out, P, R, cam, t, s)
    front = Image.new("L", (W, H), 0)
    draw_motes(front, cam, t, True, P[2] - 3, sunk, speed)
    draw_trail(front, cam, s, t, Rh, True, P[2])
    out = np.clip(out.astype(np.int16) + np.asarray(front), 0, 255).astype(np.uint8)
    if t >= BURST:
        return tint(out, omen_red(t, s, c, quad))
    return out


def frame(t: float) -> np.ndarray:
    if t < T0:                                              # (under the surface, going dark)
        g = 1 - .85 * smooth((t - (T0 - DIP_OUT)) / DIP_OUT)
        return (plunge.under(t) * g).astype(np.uint8)
    i = max(k for k, s in enumerate(SHOTS) if s["t0"] <= t)
    s = SHOTS[i]
    out = shot_frame(s, t)
    g = .15 + .85 * smooth((t - s["t0"]) / DIP_IN)          # (coming up out of the dark)
    if i + 1 < len(SHOTS):                                  # (going dark for the next)
        g *= 1 - .85 * smooth((t - (s["t1"] - DIP_OUT)) / DIP_OUT)
    if g < 1:
        out = (out * g).astype(np.uint8)
    return out
