"""M55-M81: under the ink, going up, in 3D, to break out of the surface.

M55-M59.75 seen from under it, off to its side, the camera going slowly round it: it goes up,
           faster and faster, gathering itself - light drawn into it from all round, a ring
           of light going out of it on each note, a wake of light below it, the light of the
           surface above. On M59.75 it goes: a ring flung out behind it, the camera swinging
           round behind it, a tunnel of ink rushing at it.
M60-M61    the sixteenths: curtains of orbs, three on each beat, one hole in each - it
           threads them, the hole going where it goes.
M62        an X of blades - the first prompt (down): it ducks under where they cross.
M63-M69    the sweeps: a C of ink on each note, its gap going round with the sweep - it
           chases the gap round the wall; on each pair of slides (64, 66, 68) two paths of
           light drawn across it - it goes along one, a blade along the other.
M70        bars of bone, one after another (left, right, left, right, down, up).
M71-M74    the slides: a path of light for each - it goes along them; tentacles of ink lash
           in at it out of the dark round it, more and more, each staying, writhing round it.
M74.5      the breaks: it goes forward, towards us, three times - on the first one cut all
           the way round it: all the tentacles cut away at once (and no more of them).
M75-M78.75 faster and faster: a ring of light coming down at it on each note; the mash
           (M77): faster on each press -
M79        on the last break it leaps out, red, up into the air (bga.leap_frame).

Its own little 3D world (in the units of bga: the heart is 16 across): x right, y down,
z up (the way it goes); the camera mostly under it, looking up; the soul ZS ahead of where
the camera rides, moving about the tunnel's cross-section; all else comes at it."""

from __future__ import annotations

import bisect
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import bga
import choreo as C
from chart import parse_chart
from common import FONTS, ROOT, W, H, clamp, smooth, key_angle

T0 = C.tm(55)                        # (after the fall: it starts up)
T_LEAP = C.tm(79)                    # (the last break: it leaps out of the surface)
T_END = C.tm(81)

F = 520.0                            # px
NEAR = 4.0
ZS = 160.0                           # the soul, this far ahead of the camera
V0, V1, V2 = 1000.0, 1900.0, 2900.0  # units/s: everything coming at it (once it goes; faster after the six
FAST = (C.tm(74.625), C.tm(75))      # breaks; faster still once it tears free)
T_GO = C.tm(59.75)                   # (M55-M59: under it, it goes up, slowly, gathering itself - from here, fast)
SWING = (C.tm(59.625), C.tm(59.95))  # (the camera: from under it, off to its side, round to behind it)
LUNGES = [C.tm(74.5), C.tm(74.5625), C.tm(74.625)]   # (the six breaks: it goes forward - towards us - in three)
FORE = 90.0                          # units: how far forward
SLASH_CAM = (C.tm(74.375), C.tm(74.75))   # (the camera: off to its side, under it, going round it)
SURGE = (C.tm(78.75), C.tm(79))      # (torn free: up to the surface)
Z_FAR = 1100.0                       # units: a thing is seen coming from this far ahead
FOG = 700.0
RT = 78.0                            # the tunnel's radius (at the soul's distance: ~250 px on screen)
R_GAP = 54.0                         # (where in the C's gap it goes)
R_PATH = 54.0
LEAD = .55                           # s: a prompt shows this long before its note (its ring closing)
CAM_UP, VP_UP = 34.0, 52.0           # (the camera rides this far over it; the light ahead this far up the screen)
SOUL = dict()                        # (bga's shot settings: none - it faces us, flat)


def active(t: float) -> bool:
    return T0 <= t < T_END


def bar(t):
    return (t - C.tm(0)) / (C.tm(1) - C.tm(0))


def ang_pos(a, r):
    return np.array([r * math.sin(a), -r * math.cos(a)])


def kpos(k, r):
    return ang_pos(key_angle(k), r)


# the dodging (M55-M62), in bars: the holds (when, how long, key) - blocks of ink; the taps - orbs
# at it (it steps aside); the runs of three (the hold's bar) - orbs after it as it runs; the quick
# pairs - orbs past it, one each side; the sixteenths - curtains, a group of three on each beat (its key)
SLABS = []                           # (M55-M59.75 it goes up gathering itself (no blocks, no orbs - not yet): the
SHOT_M = []                          # machinery is kept, empty)
STRAFE_M = {}
PINCER_M = []
CURTAIN_KEYS = [3, 6, 2, 7, 1, 8, 1, 8]          # (M60 + a beat each)
SLAB_OFF = 12.0                      # (a block fills the tunnel this far past its middle)
STATION = 34.0                       # (it keeps this far off the middle, away from a block)
MISS = (18.0, 28.0)                  # (an orb goes by it this far off: at least, at most)
MOVE = .13                           # s: it takes this long to get round to a block's edge, and as long on
ORB_R = 6.0
DOT_R, DOT_GAP, DOT_HOLE = 4.0, 18.0, 17.0   # (the curtains: its orbs, how far apart; its hole)

# paths: (from, to (bars); its own path (start key, shape, end key); a blade's along the other, or none)
PATHS = [(64.5, 64.875, (4, "v", 7), (3, ">", 8)),
         (66.5, 66.875, (6, "p", 2), (5, "q", 1)),
         (68.5, 68.875, (1, "<", 6), (8, ">", 3)),
         (71.0, 71.375, (5, "-", 2), None), (71.375, 71.875, (1, "s", 5), None),
         (72.0, 72.375, (8, "-", 3), None), (72.375, 72.875, (4, "z", 8), None),
         (73.0, 73.375, (5, "-", 1), None), (73.375, 73.875, (3, "s", 7), None),
         (74.0, 74.375, (8, "-", 4), None)]
PATHS = [(C.tm(a), C.tm(b), p, q) for a, b, p, q in PATHS]
SIDE = [(C.tm(70.0), (1, 0)), (C.tm(70.125), (-1, 0)), (C.tm(70.25), (1, 0)), (C.tm(70.375), (-1, 0)),
        (C.tm(70.5), (0, 1)), (C.tm(70.625), (0, -1))]       # (the bars: which way it gets out of each, from
                                                             # where it is - it comes off the sweeps up left)
DODGE = 40.0
X_T = C.tm(62.5)                     # the X of blades
LASH_M = [71, 71.25, 71.5, 71.75, 72, 72.25, 72.375, 72.5, 72.75,           # (the slides: tentacles, more and more,
          73, 73.125, 73.25, 73.5, 73.625, 73.75, 74, 74.25]              # each staying round it once it has struck)
CLEAR_T = C.tm(74.5)                 # (the first break: it cuts them all away, at once - and no more of them)
PARRY = (C.tm(74.375), C.tm(77))     # (the breaks, the triplets: tentacles closing on it - it cuts them)
MASH = (C.tm(77), C.tm(78.75))


# ------------------------------------------------------------------ how far it has come
RISE_V = (60.0, 420.0)               # units/s: (going up, gathering itself: from slowly to faster)
BOOST = 45.0                         # units/s: (the mash: faster on each press)


def speed(t):
    u = clamp((t - T0) / (T_GO - T0))
    go = smooth(clamp((t - T_GO) / .25))                     # (going up, faster and faster; then it goes)
    v = (RISE_V[0] + (RISE_V[1] - RISE_V[0]) * u ** 1.5) * (1 - go) + V0 * go
    v += (V1 - V0) * smooth(clamp((t - FAST[0]) / (FAST[1] - FAST[0])))
    v += BOOST * sum(clamp((t - b) / .06) for b in mash_notes())
    return v + (V2 - V1) * smooth(clamp((t - MASH[1]) / (T_LEAP - MASH[1])))


_MN = None


def mash_notes():
    global _MN
    if _MN is None:
        _MN = at(77, 78.75)
    return _MN


def soul_z(t):
    """How far ahead of where the camera rides it is: forward (towards us) on the breaks; torn
    free, up and away, ever faster, to the surface."""
    z = ZS
    for tb in LUNGES:
        z -= FORE / 3 * (1 - (1 - clamp((t - (tb - .05)) / .09)) ** 3)
    return z + (FORE + 130) * clamp((t - SURGE[0]) / (SURGE[1] - SURGE[0])) ** 2.2


def att(t):
    """The attacks: none before it goes; coming out of the dark above it as it does."""
    return smooth(clamp((t - (T_GO - .5)) / .6))


_XT = None


def dist(t):
    """How far the world has come at it by t."""
    global _XT
    if _XT is None:
        ts = np.arange(T0 - 3, T_END + 3, 1 / 600)
        v = np.array([speed(x) for x in ts])
        _XT = (ts, np.concatenate([[0], np.cumsum((v[1:] + v[:-1]) / 2 / 600)]))
    return float(np.interp(t, *_XT))


# ------------------------------------------------------------------ the chart
_NOTES = None


def notes():
    global _NOTES
    if _NOTES is None:
        ns, _ = parse_chart(ROOT / "maidata.txt", C.OFFSET)
        _NOTES = [n for n in ns if T0 - .01 <= n.t < T_END]
    return _NOTES


def at(m0, m1):
    """Note times in bars [m0, m1) (each time once)."""
    return sorted({round(n.t, 5) for n in notes() if m0 - 1e-4 <= bar(n.t) < m1 - 1e-4})


_GAPS = None


def gaps():
    """(when, which key): a C of ink on each note of the sweeps (the tap of a pair) -
    not in the paths."""
    global _GAPS
    if _GAPS is None:
        by = {}
        for n in notes():
            m = bar(n.t)
            if not (63 - 1e-4 <= m < 70 - 1e-4) or n.kind == "slide":
                continue
            if any(a - 1e-3 <= n.t < b for a, b, _, _ in PATHS):
                continue
            k = round(n.t, 5)
            if k not in by or (by[k][1] == "hold" and n.kind != "hold"):
                by[k] = (n.key, n.kind)
        _GAPS = sorted((t, k) for t, (k, _) in by.items())
    return _GAPS


def runs():
    """The gaps in runs (no path between)."""
    out, cur = [], []
    for g in gaps():
        if cur and g[0] - cur[-1][0] > .2:
            out.append(cur)
            cur = []
        cur.append(g)
    return out + [cur]


def path_pts(spec, r=R_PATH, n=48):
    """The polyline of a slide's shape, on the cross-section."""
    s, shape, e = spec
    A, B = kpos(s, r), kpos(e, r)
    a_s, a_e = key_angle(s), key_angle(e)
    us = np.linspace(0, 1, n)
    if shape == "-":
        return [A + (B - A) * u for u in us]
    if shape in "<>":
        da = (a_e - a_s) % math.tau if shape == ">" else -((a_s - a_e) % math.tau)
        return [ang_pos(a_s + da * u, r) for u in us]
    if shape == "v":
        return [A * (1 - 2 * u) if u < .5 else B * (2 * u - 1) for u in us]
    if shape in "pq":
        da = (a_e - a_s) % math.tau if shape == "q" else -((a_s - a_e) % math.tau)
        if abs(da) < math.pi:
            da += math.copysign(math.tau, da)
        return [ang_pos(a_s + da * u, r * (1 - .55 * math.sin(math.pi * u))) for u in us]
    d = B - A                                                # s / z: through the middle, an S
    perp = np.array([-d[1], d[0]]) / max(1e-6, np.hypot(*d))
    sg = 1 if shape == "s" else -1
    return [A + d * u + perp * .32 * r * math.sin(math.tau * u) * sg for u in us]


def along(pts, u):
    """The point u (0..1) of the way along a polyline."""
    P = np.asarray(pts)
    seg = np.hypot(*(P[1:] - P[:-1]).T)
    cum = np.concatenate([[0], np.cumsum(seg)])
    x = u * cum[-1]
    i = int(np.clip(np.searchsorted(cum, x) - 1, 0, len(seg) - 1))
    f = (x - cum[i]) / max(1e-9, seg[i])
    return P[i] + (P[i + 1] - P[i]) * f


# ------------------------------------------------------------------ the dodging: where it goes
def slab_dir(k):
    return ang_pos(key_angle(k), 1.0)


def slabs():
    return [(C.tm(a), C.tm(a + d), k) for a, d, k in SLABS]


def _switch(k0, k1):
    """Where it can be as one block ends and the next begins: clear of both."""
    s = slab_dir(k0) + slab_dir(k1)
    n = math.hypot(*s)
    return np.zeros(2) if n < .3 else -s / n * 20


_KEYS = None


def dodge_keys():
    """(when, where): M55-M62 - it goes from each to the next (smoothly; staying put
    between two the same)."""
    global _KEYS
    if _KEYS is not None:
        return _KEYS
    sl = slabs()
    keys = [(T0 - .3, np.zeros(2)), (T_GO + .05, np.zeros(2))]
    if sl:
        keys.append((sl[0][0] - .05, -slab_dir(sl[0][2]) * STATION))
    for i, (ts, te, k) in enumerate(sl):
        P, d = -slab_dir(k) * STATION, slab_dir(k)
        tg = np.array([-d[1], d[0]])
        if i:                                                # round from the last block's side to this one's
            pe, kp = sl[i - 1][1], sl[i - 1][2]
            sw = (pe + ts) / 2
            keys += [(sw - MOVE, keys[-1][1]), (sw, _switch(kp, k)), (sw + MOVE, P)]
        leave = ((te + sl[i + 1][0]) / 2 - MOVE) if i + 1 < len(sl) else C.tm(60) - .1
        run = [v for m, v in STRAFE_M.items() if abs(C.tm(m) - ts) < 1e-6]
        if run:                                              # (it comes in at the near end and runs along: they follow it)
            if i and math.hypot(*(P + tg * 30 - keys[-2][1])) < math.hypot(*(P - tg * 30 - keys[-2][1])):
                tg = -tg
            keys[-1] = (keys[-1][0], P - tg * 30)
            keys.append((leave, P + tg * 30))
            continue
        cur, sg = keys[-1][1], 1
        for m in SHOT_M:                                     # (a step aside from each orb)
            tj = C.tm(m)
            if ts <= tj < te and keys[-1][0] + .02 <= tj - .14 and tj - .02 <= leave:
                nxt = P + tg * 18 if sg > 0 else P                # (aside and back: a step each)
                keys += [(tj - .14, cur), (tj - .02, nxt)]
                cur, sg = nxt, -sg
    for g, k in enumerate(CURTAIN_KEYS):                     # the curtains: to each beat's key (the hole goes with it)
        keys.append((C.tm(60.0625 + .25 * g), kpos(k, 22)))
    keys.append((C.tm(62) - .03, np.zeros(2)))
    keys.sort(key=lambda x: x[0])
    _KEYS = keys
    return keys


def dodge_xy(t):
    keys = dodge_keys()
    ts = [k[0] for k in keys]
    if t <= ts[0]:
        return keys[0][1]
    if t >= ts[-1]:
        return keys[-1][1]
    i = bisect.bisect_right(ts, t) - 1
    (ta, pa), (tb, pb) = keys[i], keys[i + 1]
    return pa + (pb - pa) * smooth(clamp((t - ta) / max(1e-6, tb - ta)))


# ------------------------------------------------------------------ where the soul is
_WIN = None


def windows():
    """(from, to, where it is then): what it does, one after another; between them it
    goes, quickly, from the end of one to the start of the next."""
    global _WIN
    if _WIN is None:
        w = [(T0 - .3, C.tm(62), dodge_xy),
             (C.tm(62), X_T - .04, lambda t: np.array([0.0, 0.0])),
             (X_T + .06, C.tm(62.625), lambda t: np.array([0.0, 38.0]))]
        for run in runs():
            def chase(t, run=run):
                for (ta, ka), (tb, kb) in zip(run, run[1:]):
                    if t < tb:
                        a0, a1 = key_angle(ka), key_angle(kb)
                        da = (a1 - a0 + math.pi) % math.tau - math.pi
                        return ang_pos(a0 + da * clamp((t - ta) / (tb - ta)), R_GAP)
                return kpos(run[-1][1], R_GAP)
            w.append((run[0][0], run[-1][0] + .02, chase))
        for t0, t1, spec, _ in PATHS:
            pts = path_pts(spec)
            w.append((t0, t1, lambda t, pts=pts, t0=t0, t1=t1: along(pts, smooth(clamp((t - t0) / (t1 - t0))))))
        for i, (tn, d) in enumerate(SIDE):                   # the bars: out of the way of each, quickly
            t1 = SIDE[i + 1][0] - .04 if i + 1 < len(SIDE) else C.tm(70.85)
            w.append((tn + .06, t1, lambda t, p=side_pos(i + 1): p))
        w.append((PARRY[0] + .08, PARRY[1], lambda t: np.array([0.0, 0.0])))
        w.append((MASH[0], MASH[1], lambda t: np.array([0.0, 0.0])))
        w.append((MASH[1] + .15, T_END, lambda t: np.array([0.0, 0.0])))
        _WIN = sorted(w, key=lambda x: x[0])
    return _WIN


def side_pos(i):
    """Where it is after the i-th bar (0: before them - where the sweeps left it)."""
    p = kpos(gaps()[-1][1], R_GAP)
    for tn, d in SIDE[:i]:
        p = p + np.array(d, float) * DODGE
    return p


def struggle(t):
    """Held by the ink: jerking about on each note, a little."""
    p = np.array([0.0, 0.0])
    for i, b in enumerate(at(77, 78.75)):
        if b > t:
            break
        rng = np.random.default_rng(4900 + i)
        a = rng.uniform(0, math.tau)
        p = p + np.array([math.cos(a), math.sin(a)]) * 7 * math.exp(-(t - b) / .08)
    return p


def soul_xy(t):
    w = windows()
    for i, (a, b, f) in enumerate(w):
        if a <= t < b:
            return f(t)
        if t < a:                                            # (between: going from the last to this)
            pa, ta = (w[i - 1][2](w[i - 1][1]), w[i - 1][1]) if i else (f(a), a - .01)
            return pa + (f(a) - pa) * smooth((t - ta) / max(1e-6, a - ta))
    return w[-1][2](t)


# ------------------------------------------------------------------ the camera
def _axes(fwd, down):
    """World-to-camera axes: looking along fwd, down (as near as can be) along down."""
    f = fwd / np.linalg.norm(fwd)
    dn = down - (down @ f) * f
    dn = dn / np.linalg.norm(dn)
    return np.array([np.cross(dn, f), dn, f])


def _orbit(S, phi, rad, below, up, focal):
    """A view of S from off to its side (phi round it, rad away) and below it, looking up
    at it - the deep (-z) down the screen: (where, which way, down, focal, shift x, y)."""
    pos = S + np.array([rad * math.sin(phi), -rad * math.cos(phi), -below])
    return pos, S + np.array([0.0, 0.0, up]) - pos, np.array([0.0, 0.0, -1.0]), focal, 0.0, 0.0


class Cam:
    """Behind the soul (under it, looking up the way it goes), following it some of the
    way, leaning into its turns. M55: off to its side and under it, going slowly round it
    as it gathers itself - swinging round behind it as it goes. From the breaks (M74.5):
    off to its side again, under it, going round it, as it cuts its way through; torn free,
    behind it again."""

    def __init__(self, t):
        p = soul_xy(t)
        z = soul_z(t)
        v = (soul_xy(t + .03) - soul_xy(t - .03)) / .06
        roll = clamp(-v[0] * .00035, -.18, .18)
        poses = []
        wg = 1 - smooth(clamp((t - SWING[0]) / (SWING[1] - SWING[0])))
        if wg > 0:
            u = clamp((t - T0) / (SWING[1] - T0))
            poses.append((wg, _orbit(np.array([p[0], p[1], z]), math.radians(200 + 140 * u), 48 - 6 * u, 70 - 8 * u,
                                     45, 470.0)))
        ws = smooth(clamp((t - SLASH_CAM[0]) / (SLASH_CAM[1] - SLASH_CAM[0]))) * \
            (1 - smooth(clamp((t - SURGE[0]) / (SURGE[1] - SURGE[0]))))
        if ws > 0:                                           # (still down the way it came, off to one side,
            q = (soul_xy(t - .08) + p + soul_xy(t + .08)) / 3   # going round; looking at it, close now)
            u = clamp((t - SLASH_CAM[0]) / (SURGE[0] - SLASH_CAM[0]))
            side = ang_pos(math.radians(110 + 50 * u), 30)
            pos = np.array([q[0] * .5 + side[0], q[1] * .5 + side[1], 0.0])
            poses.append((ws, (pos, np.array([q[0], q[1], z + 45]) - pos, np.array([0.0, 1.0, 0.0]), F, 0.0, 0.0)))
        wc = 1 - sum(w for w, _ in poses)
        poses.append((wc, (np.array([p[0] * .5, p[1] * .5 - CAM_UP, 0.0]), np.array([0.0, 0.0, 1.0]),
                           np.array([0.0, 1.0, 0.0]), F, 0.0, -VP_UP)))
        self.pos = sum(w * q[0] for w, q in poses)
        fwd = sum(w * q[1] / np.linalg.norm(q[1]) for w, q in poses)
        down = sum(w * q[2] for w, q in poses)
        self.F = sum(w * q[3] for w, q in poses)
        self.sx = sum(w * q[4] for w, q in poses)
        self.sy = sum(w * q[5] for w, q in poses)
        roll *= wc
        self.wg, self.ws = wg, ws                            # (how much it is the view from under it (M55); the one
                                                             # as it cuts its way through)
        cr, sr = math.cos(roll), math.sin(roll)
        self.R = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1.0]]) @ _axes(fwd, down)

    def to_cam(self, P):
        return self.R @ (np.asarray(P, float) - self.pos)

    def pr(self, Pc):
        if Pc[2] < NEAR:
            return None
        k = self.F / Pc[2]
        return W / 2 + self.sx + Pc[0] * k, H / 2 + self.sy + Pc[1] * k, k

    def p(self, P):
        return self.pr(self.to_cam(P))

    def vp(self):
        """Where the way ahead (+z) is on the screen."""
        d = self.R[:, 2]
        if d[2] < .05:
            return W / 2, -4 * H
        return W / 2 + self.sx + self.F * d[0] / d[2], H / 2 + self.sy + self.F * d[1] / d[2]

    def rays(self, X, Y):
        """The way through each pixel, in the world."""
        dx, dy = (X - W / 2 - self.sx) / self.F, (Y - H / 2 - self.sy) / self.F
        Rt = self.R.T
        return [Rt[i, 0] * dx + Rt[i, 1] * dy + Rt[i, 2] for i in range(3)]


def depth(t, t_hit):
    """How far ahead a thing is that gets to the soul at t_hit."""
    return ZS + dist(t_hit) - dist(t)


def shade(z, v):
    """Its grey at z: dim far off (out of the dark ahead); gone soon once past it."""
    return v * math.exp(-max(0.0, z - ZS) / FOG) * clamp((z - (ZS - 50)) / 50)


class Pen:
    def __init__(self, d, cam, glow=None, img=None):
        self.d, self.cam, self.glow = d, cam, glow            # (glow: a layer of light added on, for fills;
        self.img = img                                       # img: what d draws on - to lay things on it)

    def line(self, A, B, v, w=2.0, add=False):
        """A line in 3D (w: its width in units, at least a pixel; add: as light, added on)."""
        far = max(A[2], B[2]) > ZS
        A, B = self.cam.to_cam(A), self.cam.to_cam(B)
        if A[2] < NEAR and B[2] < NEAR:
            return
        if A[2] < NEAR:
            A = A + (B - A) * (NEAR - A[2]) / (B[2] - A[2])
        elif B[2] < NEAR:
            B = B + (A - B) * (NEAR - B[2]) / (A[2] - B[2])
        A[2], B[2] = max(A[2], NEAR), max(B[2], NEAR)       # (cut at the near plane: exactly on it, not a hair in front)
        a, b = self.cam.pr(A), self.cam.pr(B)
        if v >= 20 or (v >= 3 and far):                      # (once past it, faint: not at all)
            d = self.glow if add and self.glow is not None else self.d
            d.line([a[:2], b[:2]], fill=int(min(255, v)), width=max(1, int(round(w * (a[2] + b[2]) / 2))))

    def poly(self, pts, fill, outline=None, w=2.0, add=False):
        q = [self.cam.p(P) for P in pts]
        if any(x is None for x in q):
            return
        if fill is not None and fill < 1:                    # (faded out: nothing - not a black shape)
            fill = None
        if add and self.glow is not None and fill is not None:
            self.glow.polygon([x[:2] for x in q], fill=int(fill))
        elif fill is not None:
            self.d.polygon([x[:2] for x in q], fill=int(fill))
        if outline is not None and outline >= 3:
            k = sum(x[2] for x in q) / len(q)
            self.d.line([x[:2] for x in q] + [q[0][:2]], fill=int(outline), width=max(1, int(round(w * k))))

    def bone(self, A, B, v, w=4.5):
        self.line(A, B, v, w)
        for P in (A, B):
            q = self.cam.p(P)
            if q and v >= 3:
                r = max(1.5, w * .9 * q[2])                   # (its knobs)
                self.d.ellipse([q[0] - r, q[1] - r, q[0] + r, q[1] + r], fill=int(v))

    def orb(self, P, r, v, trail=0.0):
        """A ball of ink (a ring of light round it, lit inside, a glint), a streak behind it."""
        q = self.cam.p(P)
        if q is None or v < 3:
            return
        if trail > 0:
            for j in range(3):
                self.line(P + np.array([0, 0, trail * j / 3]), P + np.array([0, 0, trail * (j + 1) / 3]),
                          v * .45 * (1 - j / 3), r * .8 * (1 - j / 3.5))
        rr = max(1.0, r * q[2])
        if self.glow is not None:
            self.glow.ellipse([q[0] - rr, q[1] - rr, q[0] + rr, q[1] + rr], fill=int(v * .3))
        self.d.ellipse([q[0] - rr, q[1] - rr, q[0] + rr, q[1] + rr], outline=int(v), width=max(1, int(rr * .3)))
        if rr > 3:
            h = rr * .28
            self.d.ellipse([q[0] - rr * .35 - h, q[1] - rr * .35 - h, q[0] - rr * .35 + h, q[1] - rr * .35 + h],
                           fill=int(v))


# ------------------------------------------------------------------ the tunnel
_GRID = None


def _grid():
    global _GRID
    if _GRID is None:
        Y, X = np.mgrid[0:H // 2, 0:W // 2].astype(np.float32) * 2 + 1
        _GRID = (X, Y)
    return _GRID


def tunnel_back(t, cam):
    """The dark of the ink, the light of the surface ahead (nearer: brighter) - half size."""
    X, Y = _grid()
    vx, vy = cam.vp()
    g = np.hypot(X - vx, Y - vy) * (F / cam.F)
    near = .3 + .7 * clamp((t - T0) / (T_LEAP - T0)) ** 1.3 + .6 * cam.wg   # (M55: looking up at the light)
    return 6 + 70 * near * np.exp(-(g / (40 + 50 * near)) ** 2) + 14 * near * np.exp(-(g / 260) ** 2)


def surface(t, cam):
    """Once it is torn free: the surface ahead, coming at it - a net of wavering light,
    small far off, going by bigger (not a white-out) - half size."""
    t0 = MASH[1]
    if t < t0:
        return 0.0
    zs = soul_z(T_LEAP) + dist(T_LEAP) - dist(t)
    X, Y = _grid()
    wx, wy, wz = cam.rays(X, Y)
    s = (zs - cam.pos[2]) / np.maximum(wz, 1e-3)
    hx, hy = cam.pos[0] + s * wx, cam.pos[1] + s * wy
    hx2, hy2 = hx + 3 * np.sin(hy * .09 + t * 1.3), hy + 3 * np.sin(hx * .085 - t * 1.1)
    cau = np.clip(1 - np.abs(np.sin(hx2 * .2 + t * 1.7) * np.sin(hy2 * .18 - t * 1.3)) * 4, 0, 1) ** 2
    return (10 + 75 * cau) * smooth(clamp((t - t0) / .25))


_MOTES = None


def tunnel(pen, t):
    """Its walls going by (faint rings), specks streaking past - the speed."""
    cz = pen.cam.pos[2]
    for k in range(10):
        z = (k * 200 - dist(t)) % 2000 + cz + 5
        pts = [np.array([RT * 1.12 * math.cos(u), RT * 1.12 * math.sin(u), z]) for u in np.linspace(0, math.tau, 33)]
        v = 60 * math.exp(-(z - cz) / FOG) * clamp((z - cz - 40) / 120) * smooth(clamp((t - T_GO + .1) / .45))
        for A, B in zip(pts, pts[1:]):
            pen.line(A, B, v, .8, add=True)
    global _MOTES
    if _MOTES is None:
        rng = np.random.default_rng(4910)
        _MOTES = [(math.sqrt(rng.uniform(.05, 1)) * RT * 1.05, rng.uniform(0, math.tau), rng.uniform(0, 1800))
                  for _ in range(120)]
    streak = 34 * speed(t) / V0
    for r, a, z0 in _MOTES:
        z = (z0 - dist(t)) % 1800 + cz + 5
        P = np.array([r * math.cos(a), r * math.sin(a), z])
        pen.line(P, P + np.array([0, 0, streak]), 150 * math.exp(-(z - cz) / FOG) * clamp((z - cz - 15) / 60), .5,
                 add=True)


# ------------------------------------------------------------------ the dodging: the attacks
def slab_blocks(pen, t):
    """The holds: a block of ink filling half the tunnel (past its middle by SLAB_OFF),
    as long as the hold - its face coming at it, its sides going back into the dark."""
    th = math.acos(SLAB_OFF / RT)
    for ts, te, k in slabs():
        z0, z1 = depth(t, ts), depth(t, te)
        if z1 < ZS - 50 or z0 > ZS + Z_FAR:
            continue
        a = key_angle(k)
        rim = [ang_pos(a + th * u, RT) for u in np.linspace(-1, 1, 17)]
        zf, zb = max(z0, ZS - 45), min(z1, ZS + Z_FAR)
        g = att(t)
        if z0 >= ZS - 45:                                    # its face
            pen.poly([np.append(p, z0) for p in rim], shade(z0, 40) * g, shade(z0, 240) * g, 1.3, add=True)
        if z1 <= ZS + Z_FAR:                                 # its end
            pen.poly([np.append(p, z1) for p in rim], None, shade(z1, 160) * g, 1.0)
        zs_ = np.linspace(zf, zb, 9)
        for i in (0, 4, 8, 12, 16):                          # (its edges, going back)
            for za, zb_ in zip(zs_, zs_[1:]):
                v = shade((za + zb_) / 2, 200 if i in (0, 16) else 110) * g
                pen.line(np.append(rim[i], za), np.append(rim[i], zb_), v, 1.0 if i in (0, 16) else .7)
        for za, zb_ in zip(zs_, zs_[1:]):                    # (its inner face, towards the soul: lit a little)
            pen.poly([np.append(rim[0], za), np.append(rim[16], za), np.append(rim[16], zb_), np.append(rim[0], zb_)],
                     shade((za + zb_) / 2, 16) * g, add=True)


_ORBS = None


def orbs():
    """(when it gets to the soul, where, how big, sighted on it): the taps' orbs (where it
    was, just before it stepped aside - by it at least MISS), the runs', the pairs'."""
    global _ORBS
    if _ORBS is None:
        out = []

        def aim(tj, back=.15):
            s, p = dodge_xy(tj), dodge_xy(tj - back)
            d = p - s
            n = math.hypot(*d)
            if not MISS[0] <= n <= MISS[1]:
                v = dodge_xy(tj + .03) - dodge_xy(tj - .03)
                if n > 1e-3:
                    u = d / n
                elif math.hypot(*v) > 1e-3:
                    u = np.array([-v[1], v[0]]) / math.hypot(*v)
                else:
                    u = np.array([1.0, 0.0])
                p = s + u * clamp(n, *MISS)
            return p

        for m in SHOT_M:
            out.append((C.tm(m), aim(C.tm(m)), ORB_R, True))
        for ms in STRAFE_M.values():
            for m in ms:
                out.append((C.tm(m), aim(C.tm(m), .17), ORB_R, True))
        for m in PINCER_M:                                   # (lub on one side of it, dub on the other)
            for j, mm in enumerate((m, m + .0625)):
                tj = C.tm(mm)
                v = dodge_xy(tj + .03) - dodge_xy(tj - .03)
                if math.hypot(*v) < 1e-3:
                    v = np.array([0.0, 1.0])
                nrm = np.array([-v[1], v[0]]) / math.hypot(*v)
                out.append((tj, dodge_xy(tj) + nrm * (17 if j == 0 else -17), ORB_R - 1, False))
        _ORBS = sorted(out, key=lambda x: x[0])
    return _ORBS


def dodge_orbs(pen, t):
    """The orbs coming at it: far off small, a streak behind each."""
    for tj, xy, r, _ in orbs():
        z = depth(t, tj)
        if ZS - 50 < z < ZS + Z_FAR:
            pen.orb(np.append(xy, z), r, shade(z, 235) * att(t), trail=70)


def sights(d, t, cam):
    """A sight on it before each orb at it (where it is; flashing at the last) - then it steps aside."""
    for tj, xy, r, sighted in orbs():
        a = t - (tj - .45)
        if not sighted or a < 0 or t > tj + .05:
            continue
        q = cam.p(np.append(xy, ZS))
        if q is None:
            continue
        g = clamp(a / .1) * (1 - clamp((t - tj) / .05))
        v = 150 if (tj - t > .12 or int(t * 30) % 2) else 230
        rr = 11 * q[2]
        d.ellipse([q[0] - rr, q[1] - rr, q[0] + rr, q[1] + rr], outline=int(v * g), width=2)
        for j in range(4):
            u = ang_pos(j * math.pi / 2, 1.0)
            d.line([(q[0] + u[0] * rr * .6, q[1] + u[1] * rr * .6), (q[0] + u[0] * rr * 1.4, q[1] + u[1] * rr * 1.4)],
                   fill=int(v * g), width=2)


_CURTAIN = None


def curtain_layers():
    """(when, the orbs of it): three on each beat of M60-M61, a lattice filling the tunnel,
    a hole in it where it will be."""
    global _CURTAIN
    if _CURTAIN is None:
        out = []
        base = [np.array([i * DOT_GAP + (j % 2) * DOT_GAP / 2, j * DOT_GAP * .866])
                for i in range(-6, 7) for j in range(-6, 7)]
        for g in range(len(CURTAIN_KEYS)):
            rot = .41 * g
            c, s_ = math.cos(rot), math.sin(rot)
            for j in range(3):
                tl = C.tm(60 + .25 * g + .0625 * j)
                hole = soul_xy(tl)
                pts = [np.array([c * p[0] - s_ * p[1], s_ * p[0] + c * p[1]]) for p in base]
                pts = [p for p in pts if math.hypot(*p) < RT - 6 and math.hypot(*(p - hole)) > DOT_HOLE]
                out.append((tl, pts))
        _CURTAIN = out
    return _CURTAIN


def curtains(pen, t):
    for tl, pts in curtain_layers():
        z = depth(t, tl)
        if not (ZS - 50 < z < ZS + Z_FAR):
            continue
        v = shade(z, 225) * att(t)
        for p in pts:
            pen.orb(np.append(p, z), DOT_R, v)


# ------------------------------------------------------------------ the attacks
def x_blades(pen, t):
    """M62: two blades coming, turning into an X - crossing where it was."""
    z = depth(t, X_T)
    if not (NEAR < z < ZS + Z_FAR):
        return
    pulse = sum(math.exp(-(t - b) / .1) for b in at(62, 62.5) if t >= b)
    for sg in (1, -1):
        a = math.radians(45 * sg + 50 * sg * max(0.0, X_T - t))
        d, nrm = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
        L, w = 120.0, 7.0
        pts = [np.append(d * -L, z), np.append(nrm * w, z), np.append(d * L, z), np.append(nrm * -w, z)]
        pen.poly(pts, shade(z, 70 + 40 * pulse), shade(z, 235), 1.0)


def c_rings(pen, t):
    """The sweeps: for each note a disc of ink coming, a slice out of it (where the
    note's key is) - the one way through: it chases the gap round the wall."""
    for tg, k in gaps():
        z = depth(t, tg)
        if not (NEAR < z < ZS + Z_FAR):
            continue
        a0 = key_angle(k) + math.radians(30)
        span = math.radians(300)
        us = np.linspace(0, 1, 26)
        rim = [np.append(ang_pos(a0 + span * u, RT), z) for u in us]
        pen.poly([np.array([0.0, 0.0, z])] + rim, shade(z, 34), shade(z, 235), 1.1, add=True)


def paths(pen, t):
    """The paths of light (drawn just before), its way along one; the blade along the other."""
    for t0, t1, spec, blade in PATHS:
        if not (t0 - .35 <= t < t1 + .12):
            continue
        reveal = clamp((t - (t0 - .35)) / .25)
        fade = 1 - clamp((t - t1) / .12)
        pts = path_pts(spec)
        n = max(2, int(len(pts) * reveal))
        for A, B in zip(pts[:n - 1], pts[1:n]):
            pen.line(np.append(A, ZS), np.append(B, ZS), 150 * fade, 1.6)
        if reveal >= 1:                                      # (an arrowhead at its end)
            e = np.asarray(pts[-1]); dvec = e - np.asarray(pts[-5])
            dvec = dvec / max(1e-6, np.hypot(*dvec))
            nrm = np.array([-dvec[1], dvec[0]])
            pen.poly([np.append(e + dvec * 8, ZS), np.append(e - dvec * 4 + nrm * 6, ZS),
                      np.append(e - dvec * 4 - nrm * 6, ZS)], 170 * fade)
        if blade is not None and t0 <= t < t1 + .1:          # a blade along the other: a comet of ink
            bp = path_pts(blade)
            u = smooth(clamp((t - t0) / (t1 - t0)))
            pts_b = [along(bp, u - .025 * j) for j in range(14) if u - .025 * j >= 0]
            for j, (A, B) in enumerate(zip(pts_b, pts_b[1:])):
                pen.line(np.append(A, ZS), np.append(B, ZS), 245 * (1 - j / 13) * fade, 3.2 * (1 - j / 14))


def bars(pen, t):
    """M70: bars of bone where it is - it gets out of each (left, right, ...)."""
    for i, (tn, d) in enumerate(SIDE):
        z = depth(t, tn)
        x0, y0 = side_pos(i)
        if not (NEAR < z < ZS + Z_FAR):
            continue
        if d[1] == 0:                                        # (it goes sideways: an upright bar where it was)
            pen.bone((x0, -RT, z), (x0, RT, z), shade(z, 230))
        else:
            pen.bone((-RT, y0, z), (RT, y0, z), shade(z, 230))


# ------------------------------------------------------------------ tentacles
def bez(P0, P1, P2, P3, u):
    a = 1 - u
    return a ** 3 * P0 + 3 * a * a * u * P1 + 3 * a * u * u * P2 + u ** 3 * P3


def tube(pen, pts, wid, body=200, rim=245, suckers=True, g=1.0, ghost=None):
    """A tentacle in 3D: its spine (root first) and how thick it is along it (units) - drawn
    as discs along it, as big as it is near (an edge round them all, then them); what of it
    is behind us, or too near, cut off. (ghost: only a faint fill of it, this grey.) Its
    points on the screen, from where it is cut off."""
    q, ws = [], []
    for P, w in zip(pts, wid):
        Pc = pen.cam.to_cam(P)
        x = pen.cam.pr(Pc) if Pc[2] > 18 else None
        q.append(x)
        ws.append(w)
    i0 = max([i + 1 for i, x in enumerate(q) if x is None] or [0])
    q, ws = q[i0:], ws[i0:]
    if len(q) < 3 or g <= .02:
        return None
    n_all = len(pts)
    disc = []                                                # (as close together as they need to be; its root
    for i in range(len(q) - 1):                              # end coming out of the dark)
        (x0, y0, k0), (x1, y1, k1) = q[i], q[i + 1]
        r0, r1 = max(.7, ws[i] * k0), max(.7, ws[i + 1] * k1)
        f0, f1 = (clamp((i0 + i) / (.3 * n_all)) for i in (i, i + 1))
        n = int(min(40, max(1, math.hypot(x1 - x0, y1 - y0) / max(.8, .45 * min(r0, r1)))))
        for j in range(n):
            f = j / n
            disc.append((x0 + (x1 - x0) * f, y0 + (y1 - y0) * f, r0 + (r1 - r0) * f, f0 + (f1 - f0) * f))
    x, y, k = q[-1]
    disc.append((x, y, max(.7, ws[-1] * k), 1.0))
    bx0 = max(0, int(min(x - r for x, y, r, _ in disc)) - 3)  # (drawn apart, then laid on - as much of it as shows)
    by0 = max(0, int(min(y - r for x, y, r, _ in disc)) - 3)
    bx1 = min(W, int(max(x + r for x, y, r, _ in disc)) + 4)
    by1 = min(H, int(max(y + r for x, y, r, _ in disc)) + 4)
    if bx1 <= bx0 or by1 <= by0:
        return q
    col, al = Image.new("L", (bx1 - bx0, by1 - by0), 0), Image.new("L", (bx1 - bx0, by1 - by0), 0)
    dc, da = ImageDraw.Draw(col), ImageDraw.Draw(al)
    if ghost is not None:
        for x, y, r, f in disc:
            box = [x - r - bx0, y - r - by0, x + r - bx0, y + r - by0]
            dc.ellipse(box, fill=int(ghost))
            da.ellipse(box, fill=255)
    else:
        for x, y, r, f in disc:
            box = [x - r - 1.5 - bx0, y - r - 1.5 - by0, x + r + 1.5 - bx0, y + r + 1.5 - by0]
            dc.ellipse(box, fill=int(rim))
            da.ellipse(box, fill=int(255 * g))
        for x, y, r, f in disc:
            box = [x - r - bx0, y - r - by0, x + r - bx0, y + r - by0]
            dc.ellipse(box, fill=int(body))
        if suckers:                                          # (small rings down its inner edge)
            for i in range(2, len(q) - 3, 2):
                dx, dy = q[i + 1][0] - q[i - 1][0], q[i + 1][1] - q[i - 1][1]
                m = math.hypot(dx, dy) or 1.0
                off = ws[i] * .45 * q[i][2]
                x, y = q[i][0] + dy / m * off - bx0, q[i][1] - dx / m * off - by0
                r = ws[i] * .22 * q[i][2]
                if r > 1.2:
                    dc.ellipse([x - r, y - r, x + r, y + r], outline=120, width=1)
    if pen.img is None:
        return q
    a = np.asarray(al, np.float32) / 255
    j1 = next((j for j, d_ in enumerate(disc) if d_[3] >= 1), 0)
    if j1 > 0:                                               # (its root end coming out of the dark: smoothly, along it)
        (x0, y0, _, f0), (x1, y1) = disc[0], disc[j1][:2]
        L2 = (x1 - x0) ** 2 + (y1 - y0) ** 2
        if L2 > 1:
            Yg, Xg = np.mgrid[by0:by1, bx0:bx1].astype(np.float32)
            a = a * np.clip(f0 + (1 - f0) * ((Xg - x0) * (x1 - x0) + (Yg - y0) * (y1 - y0)) / L2, 0, 1)
    base = np.asarray(pen.img.crop((bx0, by0, bx1, by1)), np.float32)
    pen.img.paste(Image.fromarray((base * (1 - a) + np.asarray(col, np.float32) * a).astype(np.uint8)), (bx0, by0))
    return q


def _perps(d):
    n1 = np.cross(d, [0.0, 0.0, 1.0])
    if np.linalg.norm(n1) < 1e-6:
        n1 = np.array([1.0, 0.0, 0.0])
    n1 = n1 / np.linalg.norm(n1)
    n2 = np.cross(d, n1)
    return n1, n2 / (np.linalg.norm(n2) or 1.0)


def soul3(t):
    return np.append(soul_xy(t), soul_z(t))


_LASH = None


def lashers():
    """The slides (M71-M74): tentacles - more and more of them - each lashing in at it on its
    beat out of the dark round it, just missing it, and staying, writhing round it, its root
    out there going along with it: (when it strikes, which way its root is (3D, from it; off
    to its sides and ahead - never ours), how far, where its tip stays (from it), its wobble)."""
    global _LASH
    if _LASH is None:
        rng = np.random.default_rng(4960)
        out = []
        for k, m in enumerate(LASH_M):
            th = k * 2.39996 + rng.uniform(-.3, .3)          # (each from a new side)
            el = math.radians(rng.uniform(0, 55))
            d = np.array([math.cos(th) * math.cos(el), math.sin(th) * math.cos(el), math.sin(el)])
            n1, n2 = _perps(d)
            side = n1 * math.cos(th * 3) + n2 * math.sin(th * 3)
            rest = d * rng.uniform(26, 36) + side * rng.uniform(-6, 6)
            out.append((C.tm(m), d, rng.uniform(100, 125), rest, rng.uniform(0, math.tau)))
        _LASH = out
    return _LASH


def _lash_spine(t, ts, d, rr, rest, wob, e=None):
    """Its spine (root to tip): lashing in (e of the way) to just by it (20 of it), easing back
    to where it stays, writhing."""
    S = soul3(t)
    root = S + d * rr
    n1, n2 = _perps(d)
    side = n1 * math.cos(wob) + n2 * math.sin(wob)
    a = t - ts
    near = .72 if a <= 0 else 1 - .28 * math.exp(-a / .09)  # (in close on the strike; back a little)
    tip = S + rest * near + side * 3 * math.sin(t * 3 + wob) * clamp(a / .3)
    if e is None:
        e = 1.0
    P1 = root - d * rr * .3 + side * 18
    P2 = tip + (root - tip) * .3 - side * 10
    pts, wid = [], []
    for u in np.linspace(0, 1, 26):
        w = (2.5 + 6 * max(0.0, 1 - e)) * math.sin(math.pi * u) * math.sin(math.tau * .8 * u - t * 4 + wob)
        pts.append(bez(root, P1, P2, tip, u * e) + n1 * w)
        wid.append(.8 + 4.6 * (1 - u) ** 1.2)
    return pts, wid


CUT_AT = 14                                                  # (of its 26: where it is cut)


def lashing(pen, t):
    """M71-M74.5: the tentacles - a faint ghost of each first (flashing), then it lashes in,
    and stays; more and more round it. On the break (M74.5) all cut away at once: the root
    ends drawing back, the cut ends flung off and falling behind, drops of ink."""
    live = [L for L in lashers() if L[0] - .28 <= t < CLEAR_T + .5]
    live.sort(key=lambda L: -pen.cam.to_cam(soul3(t) + L[1] * L[2])[2])   # (far ones first)
    for ts, d, rr, rest, wob in live:
        if t < CLEAR_T:
            if t < ts - .09:                                 # its ghost: where it will strike
                pts, wid = _lash_spine(ts, ts, d, rr, rest, wob)
                S0 = soul3(ts)
                tube(pen, [P - S0 + soul3(t) for P in pts], wid,
                     ghost=120 if (ts - t < .17 and int(t * 30) % 2) else 55)
                continue
            e = 1.06 * smooth(clamp((t - (ts - .09)) / .09)) ** 1.5 if t < ts else 1.06 - .06 * smooth((t - ts) / .06)
            pts, wid = _lash_spine(t, ts, d, rr, rest, wob, e)
            q = tube(pen, pts, wid)
            if q and ts <= t < ts + .1:                      # (the crack of it: a ring at its tip)
                b = (t - ts) / .1
                r = q[-1][2] * (4 + 14 * b)
                pen.d.ellipse([q[-1][0] - r, q[-1][1] - r, q[-1][0] + r, q[-1][1] + r], outline=int(230 * (1 - b)),
                              width=2)
            continue
        if ts >= CLEAR_T:
            continue
        a = t - CLEAR_T                                      # cut away
        pts, wid = _lash_spine(CLEAR_T, ts, d, rr, rest, wob)
        S, S0 = soul3(t), soul3(CLEAR_T)
        if a < .3:                                           # the root end drawing back
            k = max(3, int(CUT_AT * (1 - smooth(a / .3))))
            tube(pen, [P + (S - S0) + d * 60 * smooth(a / .3) for P in pts[:k]], wid[:k], g=1 - a / .3)
        if a < .45:                                          # the cut end flung off, out and back
            cut = pts[CUT_AT]
            out = cut - S0
            out = out / (np.linalg.norm(out) or 1)
            sh = out * 170 * a - np.array([0.0, 0.0, 240 * a])
            tube(pen, [P + sh for P in pts[CUT_AT - 1:]], wid[CUT_AT - 1:], suckers=False, g=1 - a / .45)
            rng = np.random.default_rng(int(ts * 1000) % 100000)
            for j in range(5):                               # (drops of ink)
                v = rng.normal(size=3)
                q = pen.cam.p(cut + v * 80 * a + out * 60 * a - np.array([0.0, 0.0, 200 * a]))
                if q:
                    r = max(1.0, rng.uniform(.6, 1.2) * q[2])
                    pen.d.ellipse([q[0] - r, q[1] - r, q[0] + r, q[1] + r], fill=int(220 * (1 - a / .45)))


def clear_slash(d_w, d_r, t, cam):
    """The break: one cut all the way round it - a ring of light, white, red at its edges (its
    blood), swept round in a flash, gone."""
    a = t - CLEAR_T
    c = cam.p(soul3(t))
    if c is None or not -.01 <= a < .22:
        return
    rho = c[2] * 16 * 1.25
    head = clamp((a + .01) / .06)
    g = 1 - clamp((a - .06) / .16)
    n = 48
    outer, inner, outer_r, inner_r = [], [], [], []
    for i in range(n + 1):
        u = i / n
        ang = -math.pi / 2 + math.tau * head * u
        w = 3 + 9 * math.sin(math.pi * u) ** .5
        ca, sa = math.cos(ang), math.sin(ang)
        outer.append((c[0] + (rho + w) * ca, c[1] + (rho + w) * sa))
        inner.append((c[0] + (rho - w * .4) * ca, c[1] + (rho - w * .4) * sa))
        outer_r.append((c[0] + (rho + w + 4) * ca, c[1] + (rho + w + 4) * sa))
        inner_r.append((c[0] + (rho - w * .4 - 4) * ca, c[1] + (rho - w * .4 - 4) * sa))
    d_r.polygon(outer_r + inner_r[::-1], fill=int(255 * .9 * g))
    d_r.polygon(outer + inner[::-1], fill=0)
    d_w.polygon(outer + inner[::-1], fill=int(255 * g))


def lunges(pen, t):
    """The breaks: forward - a ring of light going out round it, level, each time; the mash
    (M77): on each press, a small one - faster."""
    for tb, big in [(tb, True) for tb in LUNGES] + [(tb, False) for tb in mash_notes()]:
        b = (t - (tb - .05 * big)) / (.35 if big else .2)
        if not 0 <= b < 1:
            continue
        S = soul3(t)
        r = (10 + 55 * (1 - (1 - b) ** 2)) * (1 if big else .55)
        pts = [S + np.array([r * math.cos(u), r * math.sin(u), -20 * b]) for u in np.linspace(0, math.tau, 33)]
        for A, B in zip(pts, pts[1:]):
            pen.line(A, B, (230 if big else 150) * (1 - b), 1.4 if big else .9, add=not big)


_HOOPS = None


def hoops_plan():
    global _HOOPS
    if _HOOPS is None:
        rng = np.random.default_rng(4990)
        _HOOPS = [(tn, rng.uniform(-8, 8, 2), rng.uniform(30, 46), rng.uniform(-.25, .25), rng.uniform(0, math.tau))
                  for tn in at(75, 78.75)]
    return _HOOPS


def hoops(pen, t):
    """M75-M78.75, going faster and faster: on each note a ring of light coming down at it
    out of the light ahead - it going through them."""
    zs = soul_z(t)
    for tn, off, rad, tilt, ph in hoops_plan():
        z = soul_z(tn) + dist(tn) - dist(t)
        if not (zs - 40 < z < zs + Z_FAR):
            continue
        v = 235 * math.exp(-max(0.0, z - zs) / FOG) * clamp((z - (zs - 40)) / 40)
        c, s_ = math.cos(ph), math.sin(ph)
        pts = []
        for u in np.linspace(0, math.tau, 41):
            x, y = rad * math.cos(u), rad * math.sin(u)
            pts.append(np.array([off[0] + x, off[1] + y, z + (x * c + y * s_) * tilt]))
        for A, B in zip(pts, pts[1:]):
            pen.line(A, B, v, 1.6, add=True)
            pen.line(A, B, v * .35, 4.0, add=True)


# ------------------------------------------------------------------ M55: it gathers itself, and goes
_GMOTES = None


def gather_motes():
    global _GMOTES
    if _GMOTES is None:
        rng = np.random.default_rng(4980)
        out = []
        for _ in range(300):
            d = rng.normal(size=3)
            d = d / np.linalg.norm(d)
            dur = rng.uniform(.3, .6)
            t0 = T0 + (T_GO - .02 - dur - T0) * rng.random() ** .6     # (more and more of them, to its going)
            out.append((t0, dur, d, rng.uniform(45, 100), rng.uniform(1.5, 3.5) * rng.choice([-1, 1])))
        _GMOTES = out
    return _GMOTES


def gather(pen, t):
    """M55-M59.75, under it, going up: specks of light drawn into it from all round,
    spiralling, more and faster; a ring of light going out of it, level, on each note; a wake
    of light below it, longer as it goes faster; then it goes - a ring flung out and down
    behind it."""
    if not (T0 <= t < T_GO + .6):
        return
    S = soul3(t)

    def at_u(d, r0, spin, u):
        a = spin * u
        c, s_ = math.cos(a), math.sin(a)
        return S + np.array([d[0] * c - d[1] * s_, d[0] * s_ + d[1] * c, d[2]]) * r0 * (1 - u) ** 2

    for t0, dur, d, r0, spin in gather_motes():
        u = (t - t0) / dur
        if 0 <= u < 1:                                       # (a bright head, its tail fading behind)
            g = 240 * min(1.0, u * 4)
            for j, f in enumerate((1.0, .5, .2)):
                pen.line(at_u(d, r0, spin, max(0.0, u - .04 * (j + 1))), at_u(d, r0, spin, max(0.0, u - .04 * j)),
                         g * f, .35)
    for tb in at(55, 59.75):
        b = (t - tb) / .4
        if 0 <= b < 1:                                       # (left behind, below it, as it goes up)
            r = 10 + 42 * (1 - (1 - b) ** 2)
            dz = -min(45.0, dist(t) - dist(tb))
            pts = [S + np.array([r * math.cos(u), r * math.sin(u), dz]) for u in np.linspace(0, math.tau, 33)]
            for A, B in zip(pts, pts[1:]):
                pen.line(A, B, 200 * (1 - b) * (1 + dz / 60), .8, add=True)
    if t < T_GO:                                             # its wake
        rng = np.random.default_rng(4986)
        L = speed(t) * .12
        for j in range(10):
            a = rng.uniform(0, math.tau)
            rad = rng.uniform(1, 7)
            ph = (t * 3.1 + rng.uniform(0, 1)) % 1
            P = S + np.array([rad * math.cos(a), rad * math.sin(a), -6 - L * ph])
            pen.line(P - np.array([0.0, 0.0, L * .4]), P, 110 * (1 - ph) * min(1.0, L / 20), .25, add=True)
    b = (t - T_GO) / .5
    if 0 <= b < 1:                                           # (it goes)
        r = 12 + 130 * (1 - (1 - b) ** 2)
        pts = [S + np.array([r * math.cos(u), r * math.sin(u), -70 * b]) for u in np.linspace(0, math.tau, 41)]
        for A, B in zip(pts, pts[1:]):
            pen.line(A, B, 245 * (1 - b), 2.2)
        rng = np.random.default_rng(4985)
        for j in range(14):                                  # (its wake: streaks of light below it)
            a = rng.uniform(0, math.tau)
            rad = rng.uniform(2, 9)
            P = S + np.array([rad * math.cos(a), rad * math.sin(a), -10 - 160 * b * rng.uniform(.6, 1)])
            pen.line(P, P + np.array([0.0, 0.0, 40 * (1 - b)]), 220 * (1 - b), .8)


def heart_scale(t):
    """It draws itself in (gathering), out as it goes."""
    a = smooth(clamp((t - C.tm(59.625)) / (T_GO - C.tm(59.625))))
    b = clamp((t - T_GO) / .25)
    return 1 - .18 * a * (1 - b) + .12 * math.sin(math.pi * b) * (b > 0)


# ------------------------------------------------------------------ the prompts
_FONT = {}


def font(size):
    if size not in _FONT:
        _FONT[size] = ImageFont.truetype(str(FONTS / "fusion-pixel-12px.ttf"), size)
    return _FONT[size]


ARROWS = {(-1, 0): "left", (1, 0): "right", (0, -1): "up", (0, 1): "down"}


def prompts():
    """(when, what, where - off the soul, on screen - how long its ring takes)."""
    out = [(X_T, "down", (0, 1), LEAD)]
    out += [(tn, ARROWS[d], d, LEAD) for tn, d in SIDE]
    out += [(t0, "slide", None, LEAD) for t0, _, _, _ in PATHS]
    return out


def key_icon(d, x, y, glyph, v=235, k=1.0):
    """A key on the keyboard: a square, its glyph in it."""
    r = 13 * k
    d.rectangle([x - r, y - r, x + r, y + r], fill=int(min(18, v * .08)), outline=v, width=2)
    if glyph in ("left", "right", "up", "down"):
        u = {"left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1)}[glyph]
        p = np.array(u, float)
        n = np.array([-p[1], p[0]])
        c = np.array([x, y])
        tri = [c + p * 8 * k, c - p * 2 * k + n * 7 * k, c - p * 2 * k - n * 7 * k]
        d.polygon([tuple(q) for q in tri], fill=v)
        d.rectangle([min(c[0] - p[0] * 7 * k - abs(n[0]) * 2.5 * k, c[0] - p[0] * 1 * k + abs(n[0]) * 2.5 * k),
                     min(c[1] - p[1] * 7 * k - abs(n[1]) * 2.5 * k, c[1] - p[1] * 1 * k + abs(n[1]) * 2.5 * k),
                     max(c[0] - p[0] * 7 * k - abs(n[0]) * 2.5 * k, c[0] - p[0] * 1 * k + abs(n[0]) * 2.5 * k),
                     max(c[1] - p[1] * 7 * k - abs(n[1]) * 2.5 * k, c[1] - p[1] * 1 * k + abs(n[1]) * 2.5 * k)],
                    fill=v)
    elif glyph == "slide":                                   # (a star: the slide's)
        pts = []
        for i in range(10):
            rr = (9 if i % 2 == 0 else 4) * k
            a = -math.pi / 2 + i * math.pi / 5
            pts.append((x + rr * math.cos(a), y + rr * math.sin(a)))
        d.polygon(pts, fill=v)
    else:
        f = font(24)
        w = d.textlength(glyph, font=f)
        d.text((x - w / 2, y - 13), glyph, font=f, fill=v)


def draw_prompts(d, t, cam):
    """The prompts: a key showing by the soul, a ring closing on it - on the note, pressed:
    it flashes and breaks."""
    s = cam.p(soul3(t))
    off_px = max(62.0, s[2] * 8 + 34)                        # (clear of it, however close we are)
    nxt = min([p[0] for p in prompts() if p[0] >= t] or [1e9])
    for tn, glyph, off, lead in prompts():
        a = t - (tn - lead)
        if a < 0 or t > tn + .18:
            continue
        dimq = 1.0 if tn <= nxt + 1e-6 else .4               # (not its turn yet: dim)
        if glyph == "slide":                                 # (at the start of its path)
            spec = [p for p in PATHS if abs(p[0] - tn) < 1e-6][0][2]
            q = cam.p(np.append(path_pts(spec)[0], ZS))
            x, y = q[0], q[1] - 30
        else:
            x, y = (s[0] + off[0] * off_px, s[1] + off[1] * off_px)
        if t < tn:
            g = clamp(a / .1)
            rr = 15 + 40 * (1 - a / lead)
            d.ellipse([x - rr, y - rr, x + rr, y + rr], outline=int(210 * g * dimq), width=2)
            key_icon(d, x, y, glyph, int(235 * g * dimq))
        else:                                                # pressed: a flash, it breaks
            b = (t - tn) / .18
            key_icon(d, x, y, glyph, int(255 * (1 - b)), 1 + .5 * b)
            for j in range(8):
                a2 = j / 8 * math.tau
                px, py = x + math.cos(a2) * (16 + 40 * b), y + math.sin(a2) * (16 + 40 * b)
                sz = 3 * (1 - b)
                d.rectangle([px - sz, py - sz, px + sz, py + sz], fill=int(255 * (1 - b)))
    if MASH[0] - .3 <= t < MASH[1] + .08:                   # mash: the key, pulsing on every note; a bar filling
        hits = at(77, 78.75)
        n = sum(1 for b in hits if b <= t)
        pulse = max([math.exp(-(t - b) / .06) for b in hits if b <= t] or [0])
        g = clamp((t - (MASH[0] - .3)) / .15) * (1 - clamp((t - MASH[1]) / .08))
        x, y = s[0], s[1] - off_px - 8
        key_icon(d, x, y, "Z", int(235 * g), 1 + .18 * pulse)
        f = font(12)
        d.text((x + 20, y - 7), "连打!", font=f, fill=int(220 * g))
        bw = 64
        d.rectangle([x - bw / 2, y + 20, x + bw / 2, y + 26], outline=int(200 * g), width=1)
        d.rectangle([x - bw / 2 + 1, y + 21, x - bw / 2 + 1 + (bw - 2) * n / max(1, len(hits)), y + 25],
                    fill=int(235 * g))


# ------------------------------------------------------------------ the frame
def glow_more(out, c, k, t):
    """M55: it gathers itself - its glow growing to the dub, flaring as it goes."""
    g = smooth(clamp((t - T0) / (T_GO - T0))) * .8 + .9 * math.exp(-max(0.0, t - T_GO) / .18) * (t >= T_GO)
    g *= 1 - smooth(clamp((t - T_GO - .2) / .4))
    if c is None or g <= .01:
        return out
    X, Y = _grid()
    rr = np.hypot(X - c[0], Y - c[1]) / (k * 16 * 1.3)
    h = (90 * g * np.exp(-rr ** 2)).repeat(2, 0).repeat(2, 1)
    return np.clip(out + h[..., None] if out.ndim == 3 else out + h, 0, 255).astype(np.uint8)


def frame(t: float) -> np.ndarray:
    if t >= T_LEAP:                                          # the last break: it leaps out
        return bga.leap_frame(dict(bga.LEAP, t0=T_LEAP, t1=T_END), t)
    cam = Cam(t)
    back = tunnel_back(t, cam) + surface(t, cam)
    out = np.clip(back, 0, 255).astype(np.uint8).repeat(2, 0).repeat(2, 1)
    im, gl = Image.fromarray(out), Image.new("L", (W, H), 0)
    pen = Pen(ImageDraw.Draw(im), cam, ImageDraw.Draw(gl))
    tunnel(pen, t)
    gather(pen, t)
    slab_blocks(pen, t)
    curtains(pen, t)
    dodge_orbs(pen, t)
    x_blades(pen, t)
    c_rings(pen, t)
    bars(pen, t)
    hoops(pen, t)
    lunges(pen, t)
    out = np.clip(np.asarray(im).astype(np.int16) + np.asarray(gl), 0, 255).astype(np.uint8)
    xy = soul_xy(t)                                          # the soul: facing down the way it came, leaning into its turns
    vxy = (soul_xy(t + .03) - soul_xy(t - .03)) / .06
    lean = clamp(vxy[0] * .0012, -.5, .5)
    Rw = bga._rz(lean * .4) @ bga._ry(-lean) @ np.diag([1, 1, -1.0])
    wf = cam.wg + cam.ws
    if wf > 0:                                               # (M55: turned to us, tipped back a little - as the camera
        S = soul3(t)                                         # goes round behind it, facing down the way it came; so
        to = cam.pos - S                                     # as it cuts its way through)
        to = to / np.linalg.norm(to)
        sw = .55 * math.sin(2.4 * (t - T0)) * (cam.wg > 0)   # (M55: turning slowly, this way and that)
        to = np.array([to[0] * math.cos(sw) - to[1] * math.sin(sw), to[0] * math.sin(sw) + to[1] * math.cos(sw), to[2]])
        tip_back = .75 if cam.wg > 0 else .35
        nrm = to * math.cos(tip_back) + np.array([0.0, 0.0, -1.0]) * math.sin(tip_back)
        nrm = Rw[:, 2] * (1 - wf) + nrm / np.linalg.norm(nrm) * wf
        nrm = nrm / np.linalg.norm(nrm)
        down = np.array([0.0, 0.0, -1.0]) if cam.wg > 0 else np.array([0.0, 1.0, 0.0])
        v = Rw[:, 1] * (1 - wf) + down * wf
        v = v - (v @ nrm) * nrm
        v = v / np.linalg.norm(v)
        Rw = np.column_stack([np.cross(nrm, v), v, nrm])
    R = cam.R @ Rw @ np.diag([heart_scale(t), heart_scale(t), 1.0])
    P = cam.to_cam(soul3(t))
    c = cam.pr(P + R[:, 1] * .05 * 16)
    out = bga.halo(out, c, t, SOUL)
    if c is not None:
        out = glow_more(out, c, c[2], t)
    im = Image.fromarray(out)
    sights(ImageDraw.Draw(im), t, cam)                       # (under it: it is on the sight)
    out = np.asarray(im).copy()
    quad = bga.big_heart(out, P, R, cam, t, SOUL)
    im = Image.fromarray(out)
    pen = Pen(ImageDraw.Draw(im), cam, img=im)
    paths(pen, t)                                            # (over it: what it goes along, the ink at it)
    lashing(pen, t)
    red = Image.new("L", (W, H), 0)
    clear_slash(ImageDraw.Draw(im), ImageDraw.Draw(red), t, cam)
    omen = bga.omen_red(t, SOUL, c, quad)
    cut = np.asarray(red, np.float32) / 255
    out = bga.tint(np.asarray(im), cut if omen is None else np.maximum(omen, cut))
    ui = Image.new("L", (W, H), 0)
    du = ImageDraw.Draw(ui)
    if c is not None and 0 <= t - T0 < .5:                   # (in on the downbeat: a beat of its heart going out)
        b = (t - T0) / .5
        r = c[2] * 16 * (.7 + 3 * (1 - (1 - b) ** 2))
        du.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], outline=int(220 * (1 - b)), width=3)
    draw_prompts(du, t, cam)
    return np.maximum(out, np.asarray(ui)[..., None])
