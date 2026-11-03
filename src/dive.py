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
T_GO = C.tm(62)                      # (M55-M61, the build: under it, it goes up, slowly, faster and faster - on the
SWING = (C.tm(61.75), C.tm(62.06))   # drop (M62) flat out; the camera from under it, off to its side, round behind it)
BUILD = [(55, 57, (18, 40), (-50, -47), (66, 54), (26, 22), 0, 470, 20, 20),      # (the build's shots - cut on the
         (57, 59, (-30, -15), (-16, -20), (74, 60), (12, 10), 90, 560, -28, 8),    # bars, each pushing slowly in,
         (59, 60, (40, 52), (-38, -42), (46, 38), (10, 8), -80, 470, 25, 35),      # each nearer than the last; the
         (60, 62.2, (80, 120), (-70, -76), (44, 30), (6, 4), 0, 430, 0, 62)]       # light always up the screen, it
BUILD = [(C.tm(a), C.tm(b), *r) for a, b, *r in BUILD]   # always going up it: from, to (bars); round it, up from level
                                                         # (deg); how far; looking this far over it; where it is across
                                                         # the screen; the lens; how it is turned from us, how far it
                                                         # lies back (deg: fixed through the shot - the camera going
                                                         # round it shows it; its top always up the screen)
LUNGES = [C.tm(74.5), C.tm(74.5625), C.tm(74.625)]   # (the six breaks: it goes forward - towards us - in three)
FORE = 90.0                          # units: how far forward
RACE = C.tm(74.75)                   # (after the cut: the camera stays behind it, under it - a racing game's: as it
RACE_LAG = .022                      # speeds up it pulls ahead of us a little, and we catch up (units per unit/s gained
RACE_FOV = .25                       # in the last .4 s); the lens widens with the speed (F (V0 / v) ** this); it weaves
RACE_WEAVE = (9.0, 5.0)              # a little, the camera leaning after it)
SURGE_OFF = (45.0, 130.0)            # units: (torn free: off to our right, and away ahead - at the surface - faster
                                     # and faster: aslant, as it comes out)
ORBIT = (C.tm(75), C.tm(79))         # (the last of the tunnel, speeding up: the camera going over it - round its x -
ORBIT_PHI = 40.0                     # deg: so the way ahead (the tunnel's middle) goes on up the screen, out at the
ORBIT_IN = .5                        # top; nearer it too (this much the way, at the end); and it tips its top over
TIP, TIP_BY = 90.0, .5               # deg, of the way: towards the way ahead, round its own x - till its top is square to the
AIM = (C.tm(78.5), C.tm(79))         # tunnel's rings, as it comes out (bga's leap: top first). Torn free: aslant,
AIM_SLANT = 32.0                     # deg: its top to our right (bga.LEAP_SLANT))
SURGE_FOLLOW = .85                   # (the camera after it, as it goes up at the surface, this much of the way)
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
CURTAIN_KEYS = []                    # (M60-M61's curtains of orbs: gone - the build goes on to the drop, M62)
SLAB_OFF = 12.0                      # (a block fills the tunnel this far past its middle)
STATION = 34.0                       # (it keeps this far off the middle, away from a block)
MISS = (18.0, 28.0)                  # (an orb goes by it this far off: at least, at most)
MOVE = .13                           # s: it takes this long to get round to a block's edge, and as long on
ORB_R = 6.0
DOT_R, DOT_GAP, DOT_HOLE = 4.0, 18.0, 17.0   # (the curtains: its orbs, how far apart; its hole)

# paths: (from, to (bars); its own path (start key, shape, end key); a blade's along the other, or none)
PATHS = []                           # (the sweeps' pairs of slides (M64.5, M66.5, M68.5): no paths of light, no
                                     # star to follow - the sweep goes on through them, round the way one of the
SWEEP_ARCS = [(64.5, (3, 4, 5, 6, 7, 8)),   # pair goes (3>8, 5q1, 8>3): a disc a sixteenth, on from where it is)
              (66.5, (5, 6, 7, 8, 1)),
              (68.5, (8, 1, 2, 3))]
TURN_KEYS = {1: ("right", (576.0, 240.0)), -1: ("left", (64.0, 240.0))}   # (the sweeps: a key each side, where
                                     # they stay - the one it turns to lit on each turn: round to the right
                                     # (clockwise), to the left)
SIDE = [(C.tm(70.0), (1, 0)), (C.tm(70.125), (-1, 0)), (C.tm(70.25), (1, 0)), (C.tm(70.375), (-1, 0)),
        (C.tm(70.5), (0, 1)), (C.tm(70.625), (0, -1))]       # (the bars: which way it gets out of each, from
                                                             # where it is - it comes off the sweeps up left)
DODGE = 40.0
X_T = C.tm(62.5)                     # the X of blades
PROJ = [((71.0, 2), (71.375, 2), 71.625), ((72.0, 3), (72.375, 3), 72.625),   # (the stars, a projection a bar:
        ((73.0, 4), (73.375, 4), 73.625), ((74.0, 5), 74.25)]                    # its cuts projected on the two
                                     # stars' taps - half on each, not cut yet - and all let go at once as the zigzag
                                     # slide sets off (M74: one star, its slide); (tap, how many) ..., let go)
DASH = .08                           # s: (let go: through all of them in this long)
BACK = .2                            # s: (and then back to the middle, upright again, in this long)
R_GHOST = 30.0                       # (its ghosts: this far off where it is, more the more of them; in and out by turns)
QTE_KEYS = ("left", "right", "up", "down", "Z", "X")
FRAME_DT = 1 / 24                    # s: (the projection's frames: 24 a second, one ghost each)
CLEAR_T = C.tm(74.5)                 # (the first break: a swarm of them out of the wall all round it at once (from the
SWARM_N = 12                         # tap just before) - all cut away with one cut; no more of them)
LASH_WARN = .12                      # s: the wall cracking where one will come out, before it does
RW = RT * 1.1                        # (the wall they come out of: where the tunnel's rings are)
ROOT_FLOW = .35                      # (their roots come on slower than the world: they keep up with it, a little)
PARRY = (C.tm(74.375), C.tm(77))     # (the breaks, the triplets: tentacles closing on it - it cuts them)
MASH = (C.tm(77), C.tm(78.75))


# ------------------------------------------------------------------ how far it has come
RISE_V0, RISE_P = 40.0, 3.0          # units/s, power: (M55-M60 going up: from slowly, faster and faster - smoothly,
RISE_TO = C.tm(62)                   # no leap in it - till by the drop (M62) it is going at V0)
BOOST = 45.0                         # units/s: (the mash: faster on each press)


def speed(t):
    u = clamp((t - T0) / (RISE_TO - T0))
    v = RISE_V0 + (V0 - RISE_V0) * u ** RISE_P               # (going up, slowly, faster and faster)
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
    """How far ahead of where the camera rides it is: forward (towards us) on the breaks; torn free,
    away ahead (and a little across: race_xy), faster and faster, to the surface; speeding up, ahead
    of us a little (race_lag)."""
    z = ZS
    for tb in LUNGES:
        z -= FORE / 3 * (1 - (1 - clamp((t - (tb - .05)) / .09)) ** 3)
    return z + SURGE_OFF[1] * surge_u(t) + race_lag(t)


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
        for m, ks in SWEEP_ARCS:
            for i, k in enumerate(ks):
                by[round(C.tm(m + i / 16), 5)] = (k, "arc")
        _GAPS = sorted((t, k) for t, (k, _) in by.items())
    return _GAPS


_TURNS = None


def turns():
    """(when, which way: 1 clockwise - right, -1 left): where it sets off round the wall the
    other way (and the first)."""
    global _TURNS
    if _TURNS is None:
        out, last = [], 0
        g = gaps()
        for (ta, ka), (tb, kb) in zip(g, g[1:]):
            da = (key_angle(kb) - key_angle(ka) + math.pi) % math.tau - math.pi
            if abs(da) < 1e-6:
                continue
            w = 1 if da > 0 else -1
            if w != last:
                out.append((ta, w))
            last = w
        _TURNS = out
    return _TURNS


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
        w.append((C.tm(71), C.tm(74.375), tangle_xy))
        for i, (tn, d) in enumerate(SIDE):                   # the bars: out of the way of each, quickly
            t1 = SIDE[i + 1][0] - .04 if i + 1 < len(SIDE) else C.tm(70.85)
            w.append((tn + .06, t1, lambda t, p=side_pos(i + 1): p))
        w.append((PARRY[0] + .08, PARRY[1], race_xy))
        w.append((MASH[0], T_END, race_xy))
        _WIN = sorted(w, key=lambda x: x[0])
    return _WIN


def tangle_xy(t):
    """M71-M74.375, the stars: still, where its last cuts left it, while it projects the next; let
    go, through all its ghosts in a flash (DASH) - the last of them is where it waits next; after the
    last, back to the middle (the swarm)."""
    P = projections()
    for pj in P:
        if t < pj["tr"]:
            return pj["home"]
        if t < pj["tr"] + DASH:
            return along(pj["path"], (t - pj["tr"]) / DASH)
        if t < pj["tr"] + DASH + BACK:                       # (cut: straight back to the middle)
            u = (t - pj["tr"] - DASH) / BACK
            return pj["ghosts"][-1] * (1 - u) ** 3
    return np.zeros(2)


def orbit(t):
    """The last of the tunnel: how far the camera has gone over it (the way ahead up the screen), how
    far it has tipped its top over (rad)."""
    u = clamp((t - ORBIT[0]) / (ORBIT[1] - ORBIT[0]))
    return math.radians(ORBIT_PHI) * u ** 1.6, math.radians(TIP) * smooth(clamp(u / TIP_BY))


def surge_u(t):
    return clamp((t - SURGE[0]) / (SURGE[1] - SURGE[0])) ** 2.2


def race_lag(t):
    """After the cut: how far it has pulled ahead of us, speeding up."""
    if t < RACE:
        return 0.0
    return RACE_LAG * max(0.0, speed(t) - speed(t - .4))


def race_xy(t):
    """After the cut: weaving a little as it goes; torn free, off a little to our right."""
    w = smooth(clamp((t - RACE) / .5))
    tau = t - RACE
    return (np.array([RACE_WEAVE[0] * math.sin(1.3 * tau), RACE_WEAVE[1] * math.sin(1.9 * tau + 1.0)]) * w +
            np.array([SURGE_OFF[0], 0.0]) * surge_u(t))


def cam_xy(t):
    """Where the camera follows it: among the tentacles, not its flashes from cut to cut - only
    from where it waits to where it waits next, easing across each."""
    w = smooth(clamp((t - C.tm(70.75)) / .15)) * (1 - smooth(clamp((t - C.tm(74.45)) / .15)))
    if w <= 0:
        return soul_xy(t)
    return soul_xy(t) * (1 - w)                              # (it is in the middle, and comes back to it)


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
def build_heart(t):
    """The build: how the soul faces in this shot (its face, its top), fixed through it: turned
    from the camera (as it is halfway through the shot), lying back."""
    for t0, t1, az, el, d, up, sx, f, turn, lie in BUILD:
        if t < t1 or t1 == BUILD[-1][1]:
            a_ = math.radians((az[0] + az[1]) / 2 + turn)
            h = np.array([math.sin(a_), -math.cos(a_), 0.0])
            b = math.radians(lie)
            return h * math.cos(b) - np.array([0.0, 0.0, math.sin(b)]), h * math.sin(b) + np.array([0.0, 0.0, math.cos(b)])


def _axes(fwd, down):
    """World-to-camera axes: looking along fwd, down (as near as can be) along down."""
    f = fwd / np.linalg.norm(fwd)
    dn = down - (down @ f) * f
    dn = dn / np.linalg.norm(dn)
    return np.array([np.cross(dn, f), dn, f])


def build_pose(t, S):
    """The build (M55-M62): which shot, and in it the camera pushing slowly in."""
    for t0, t1, az, el, d, up, sx, f, _, _ in BUILD:
        if t < t1 or t1 == BUILD[-1][1]:
            u = clamp((t - t0) / (t1 - t0))
            lerp = lambda ab: ab[0] + (ab[1] - ab[0]) * u
            a_, e_, D = math.radians(lerp(az)), math.radians(lerp(el)), lerp(d)
            pos = S + D * np.array([math.cos(e_) * math.sin(a_), -math.cos(e_) * math.cos(a_), math.sin(e_)])
            return pos, S + np.array([0.0, 0.0, lerp(up)]) - pos, np.array([0.0, 0.0, -1.0]), float(f), float(sx), 0.0


class Cam:
    """Behind the soul (under it, looking up the way it goes), following it some of the
    way, leaning into its turns. M55: off to its side and under it, going slowly round it
    as it goes up (cut on the bars, pushing in) - swinging round behind it on the drop. After
    the cut: behind it still - a racing game's camera (race_lag, race_xy, the lens widening);
    torn free, it shoots off ahead at the surface."""

    def __init__(self, t):
        p = cam_xy(t)
        z = soul_z(t)
        v = (cam_xy(t + .03) - cam_xy(t - .03)) / .06
        roll = clamp(-v[0] * .00035, -.18, .18)
        poses = []
        wg = 1 - smooth(clamp((t - SWING[0]) / (SWING[1] - SWING[0])))
        if wg > 0:
            poses.append((wg, build_pose(t, np.array([p[0], p[1], z]))))
        wc = 1 - sum(w for w, _ in poses)
        fov = (V0 / max(V0, speed(t))) ** RACE_FOV if t > RACE else 1.0   # (faster: wider)
        pos, fwd, down = np.array([p[0] * .5, p[1] * .5 - CAM_UP, 0.0]), np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])
        if t > ORBIT[0]:                                     # (the last of it: going over it, round its x - the way
            phi = orbit(t)[0]                                # ahead going up the screen; nearer)
            su = surge_u(t)                                  # (torn free: after it some of the way - it up the
            pos[0] -= SURGE_OFF[0] * su * .5                 # screen at the surface, on it in sight - not across)
            pos[2] += SURGE_OFF[1] * su * SURGE_FOLLOW
            Q = np.array([p[0], p[1], z - SURGE_OFF[1] * su * (1 - SURGE_FOLLOW) - race_lag(t)])
            rx = lambda v: np.array([v[0], v[1] * math.cos(phi) + v[2] * math.sin(phi),
                                     -v[1] * math.sin(phi) + v[2] * math.cos(phi)])
            near = 1 - (1 - ORBIT_IN) * smooth(clamp((t - ORBIT[0]) / (ORBIT[1] - ORBIT[0])))
            pos, fwd, down = Q + rx(pos - Q) * near, rx(fwd), rx(down)
        poses.append((wc, (pos, fwd, down, F * fov, 0.0, -VP_UP)))
        self.pos = sum(w * q[0] for w, q in poses)
        fwd = sum(w * q[1] / np.linalg.norm(q[1]) for w, q in poses)
        down = sum(w * q[2] for w, q in poses)
        self.F = sum(w * q[3] for w, q in poses)
        self.sx = sum(w * q[4] for w, q in poses)
        self.sy = sum(w * q[5] for w, q in poses)
        roll *= wc
        self.wg, self.ws = wg, 0.0                           # (how much it is the build's shots (M55-M62); ws: a view
                                                             # from beside it - none now)
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
    """The dark of the ink, the light of the surface ahead - straight up (nearer: brighter);
    looking up at it (the build): soft shafts of that light coming down, swaying slowly - half
    size."""
    X, Y = _grid()
    wx, wy, wz = cam.rays(X, Y)
    ang = np.arccos(np.clip(wz / np.sqrt(wx * wx + wy * wy + wz * wz), -1, 1))   # (from straight up)
    g = 520 * np.tan(np.minimum(ang, 1.5))                   # (as px off the light, at our focal length)
    near = .3 + .7 * clamp((t - T0) / (T_LEAP - T0)) ** 1.3 + .6 * cam.wg   # (M55: looking up at the light)
    out = 6 + 70 * near * np.exp(-(g / (40 + 50 * near)) ** 2) + 14 * near * np.exp(-(g / 260) ** 2)
    out = out + cam.ws * near * 26 * np.clip(np.cos(ang), 0, 1) ** 1.5   # (beside it: lighter up towards the surface)
    k = cam.wg + .6 * cam.ws
    if k > .01:
        az = np.arctan2(wy, wx)
        sh = np.zeros_like(X)
        for j in range(9):
            c0 = j / 9 * math.tau + .5 * math.sin(j * 2.3)
            w, a, f = .16 + .12 * ((j * .618) % 1), .6 + .4 * ((j * .37) % 1), .3 + .3 * ((j * .71) % 1)
            c = c0 + .05 * math.sin(t * f + j * 1.7)
            sh += a * (.7 + .3 * math.sin(t * f * 1.9 + j)) * np.exp(-(((az - c + math.pi) % math.tau - math.pi) / w) ** 2)
        out = out + 14 * k * sh * np.exp(-ang / .9) * np.clip(ang / .25, 0, 1)
    return out


def surface(t, cam):
    """Once it is torn free: the surface ahead, coming at it - a net of wavering light,
    small far off, going by bigger (not a white-out) - half size."""
    t0 = MASH[1]
    if t < t0:
        return 0.0
    zs = soul_z(T_LEAP) + dist(T_LEAP) - dist(t)
    X, Y = _grid()
    wx, wy, wz = cam.rays(X, Y)
    nw = np.sqrt(wx * wx + wy * wy + wz * wz)
    up = wz / nw > .06                                      # (only looking up at it: towards its far edge, dim)
    s = np.where(up, (zs - cam.pos[2]) / np.maximum(wz, 1e-3), 0)
    hx, hy = cam.pos[0] + s * wx, cam.pos[1] + s * wy
    hx2, hy2 = hx + 3 * np.sin(hy * .09 + t * 1.3), hy + 3 * np.sin(hx * .085 - t * 1.1)
    cau = np.clip(1 - np.abs(np.sin(hx2 * .2 + t * 1.7) * np.sin(hy2 * .18 - t * 1.3)) * 4, 0, 1) ** 2
    edge = np.clip((wz / nw - .06) / .25, 0, 1)
    win = np.clip((wz / nw - math.cos(math.radians(49))) / .07, 0, 1)   # (straight over it, the light through the
    v = (10 + 75 * cau) * (.35 + .65 * win) + 40 * win                  # surface; further off, only its own sheen)
    return np.where(up, v * edge, 0) * smooth(clamp((t - t0) / .25))


_MOTES = None


def tunnel(pen, t):
    """Its walls going by (faint rings), specks streaking past - the speed."""
    cz = pen.cam.pos[2]
    for k in range(10):
        z = (k * 200 - dist(t)) % 2000 + cz + 5
        pts = [np.array([RT * 1.12 * math.cos(u), RT * 1.12 * math.sin(u), z]) for u in np.linspace(0, math.tau, 33)]
        v = 60 * math.exp(-(z - cz) / FOG) * clamp((z - cz - 40) / 120) * smooth(clamp((t - T_GO + .1) / .45)) * \
            (1 - pen.cam.ws)                                 # (beside it, under it: not the walls - we are near them)
        for A, B in zip(pts, pts[1:]):
            pen.line(A, B, v, .8, add=True)
    global _MOTES
    if _MOTES is None:
        rng = np.random.default_rng(4910)
        _MOTES = [(math.sqrt(rng.uniform(.05, 1)) * RT * 1.05, rng.uniform(0, math.tau), rng.uniform(0, 1800))
                  for _ in range(120)]
    streak = 34 * speed(t) / V0
    for r, a, z0 in _MOTES:
        z = (z0 - dist(t)) % 1800 + cz + 5 - 420 * pen.cam.ws   # (beside it: round us, above and below)
        P = np.array([r * math.cos(a), r * math.sin(a), z])
        pen.line(P, P + np.array([0, 0, streak]), 150 * math.exp(-(z - cz) / FOG) * clamp((z - cz - 15) / 60), .5,
                 add=True)


_RMOTES = None
RM_SPAN = 640.0                      # units: (the specks round it, from RM_SPAN * .3 below it to the rest above)


def rise_motes(pen, t):
    """M55-M60, going up: specks in the ink round it, going down past it - slowly at first,
    then faster and faster (streaking as it gets up speed): far ones small and dim, those
    near us big and soft. Gone as it is going flat out, into the tunnel."""
    g = max(1 - smooth(clamp((t - (T_GO + .05)) / .3)), pen.cam.ws)   # (and beside it, under it, after the cut)
    if g <= 0 or t < T0 - .1:
        return
    global _RMOTES
    if _RMOTES is None:
        rng = np.random.default_rng(4975)
        n = 340
        r = 150 * np.sqrt(rng.uniform(.002, 1, n))
        a = rng.uniform(0, math.tau, n)
        _RMOTES = (r * np.cos(a), r * np.sin(a), rng.uniform(0, RM_SPAN, n), rng.uniform(.18, .5, n),
                   rng.uniform(.6, 1.0, n))
    xs, ys, z0s, rs, bs = _RMOTES
    S = soul3(t)
    v = speed(t)
    D = dist(t)
    for x, y, z0, r, b in zip(xs, ys, z0s, rs, bs):
        z = (z0 - D) % RM_SPAN + S[2] - .3 * RM_SPAN
        P = np.array([S[0] * .3 + x, S[1] * .3 + y, z])
        Pc = pen.cam.to_cam(P)
        if Pc[2] < 6:
            continue
        q = pen.cam.pr(Pc)
        val = 150 * b * math.exp(-Pc[2] / 320) * g
        rr = r * q[2]
        if rr > 2.5:                                        # (near us: big and soft, dimmer)
            val *= 2.5 / rr
            rr = min(rr, 12.0)
        tail = pen.cam.p(P + np.array([0.0, 0.0, v * min(.07, 40 / v)]))  # (where it was a moment ago: above)
        if tail is not None and math.hypot(tail[0] - q[0], tail[1] - q[1]) > 2 * rr + 1:
            wd = max(1, int(round(min(rr, 3.0))))
            for j, f in enumerate((1.0, .55, .25)):          # (bright at its head, fading behind)
                a0, a1 = j / 3, (j + 1) / 3
                pen.glow.line([(q[0] + (tail[0] - q[0]) * a0, q[1] + (tail[1] - q[1]) * a0),
                               (q[0] + (tail[0] - q[0]) * a1, q[1] + (tail[1] - q[1]) * a1)],
                              fill=int(val * f), width=wd)
        elif rr > 1.2:
            pen.glow.ellipse([q[0] - rr * 1.6, q[1] - rr * 1.6, q[0] + rr * 1.6, q[1] + rr * 1.6], fill=int(val * .35))
            pen.glow.ellipse([q[0] - rr, q[1] - rr, q[0] + rr, q[1] + rr], fill=int(val))
        else:
            pen.glow.rectangle([q[0] - .5, q[1] - .5, q[0] + .5, q[1] + .5], fill=int(val))


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


def tube(pen, pts, wid, body=200, rim=245, suckers=True, g=1.0, ghost=None, near=18.0):
    """A tentacle in 3D: its spine (root first) and how thick it is along it (units) - drawn
    as discs along it, as big as it is near (an edge round them all, then them); what of it
    is behind us, or too near, cut off. (ghost: only a faint fill of it, this grey.) Its
    points on the screen, from where it is cut off."""
    q, ws = [], []
    for P, w in zip(pts, wid):
        Pc = pen.cam.to_cam(P)
        x = pen.cam.pr(Pc) if Pc[2] > near else None
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


_PROJ = None


def projections():
    """M71-M74.25, the stars - Naoya's projection, one a bar: by each star's tap tentacles have burst
    out of the tunnel's wall at it and are held there, short of it (as in a frame); on the tap it
    projects its cuts for them - frame after frame (24 a second) a ghost of itself where it will cut
    each, in a frame, a key over each (random) - but does not cut: the bar's two star taps project
    half each; as the zigzag slide sets off it lets them all go at once: through all of its ghosts
    in a flash, each tentacle cut clean as it goes by. Each: tp (the first tap), tr (let go), home
    (where it is), ghosts (zigzagging across it, the last back towards the middle: it waits there for
    the next), shows (when each ghost shows), path, keys, ten (its tentacles)."""
    global _PROJ
    if _PROJ is None:
        rng = np.random.default_rng(4960)
        out = []
        home = np.zeros(2)
        prev = C.tm(70.75)
        for j, spec in enumerate(PROJ):
            batches, tr = [(C.tm(m), k) for m, k in spec[:-1]], C.tm(spec[-1])
            n = sum(k for _, k in batches)
            back = math.atan2(-home[1], -home[0]) if np.linalg.norm(home) > 4 else rng.uniform(0, math.tau)
            step = 2.6 if j % 2 == 0 else -2.6               # (one across from the last: a zigzag)
            ths = [back - (n - 1 - k) * step + rng.uniform(-.15, .15) for k in range(n)]
            rad = [R_GHOST + 3 * n + (5 if k % 2 else -5) for k in range(n)]
            ghosts = [home + r * np.array([math.cos(x), math.sin(x)]) for x, r in zip(ths, rad)]
            path = [home] + ghosts
            seg = np.array([np.linalg.norm(q1 - q0) for q0, q1 in zip(path, path[1:])])
            cum = np.cumsum(seg) / seg.sum()
            ten, shows, keys, k = [], [], [], 0
            for tb, nb in batches:
                lead = min(.22, tb - prev - .01)             # (out of the wall, just before its tap)
                stag = min(.025, (lead - .08) / max(1, nb - 1))
                kb = [QTE_KEYS[i] for i in rng.permutation(len(QTE_KEYS))[:nb]]
                for jb in range(nb):
                    x = ths[k]
                    d = np.array([math.cos(x), math.sin(x)])
                    ten.append(dict(te=tb - lead + stag * jb, tf=tb - .02, tp=tb, tr=tr, tc=tr + DASH * cum[k], th=x,
                                    D=rng.uniform(45, 85), wob=rng.uniform(0, math.tau), curl=rng.uniform(-1, 1),
                                    tip=np.append(ghosts[k] + 9 * d, ZS + 6), ref=np.append(ghosts[k], ZS),
                                    swarm=False, seed=4961 + 10 * j + k))
                    shows.append(tb + jb * FRAME_DT)
                    keys.append(kb[jb])
                    k += 1
                prev = tb
            out.append(dict(tp=batches[0][0], tr=tr, home=home, ghosts=ghosts, path=path, ten=ten, shows=shows,
                            keys=keys))
            prev = tr                                        # (cut: back to the middle, upright, each time)
        _PROJ = out
    return _PROJ


_LASH = None


def lashers():
    """All the tentacles: the stars' (projections), then the swarm on the break - out of the wall
    all round it at once from the tap before, in at it, all cut with one cut (CLEAR_T). Each a
    dict: te (out of the wall), tf (all the way in), tc (cut), th (where round the wall), D (its
    root this far ahead of the soul, held or cut), its wobble, curl; the stars': tp, tr (held from
    the one to the other), tip, ref (where it is cut from)."""
    global _LASH
    if _LASH is None:
        out = [L for pj in projections() for L in pj["ten"]]
        rng = np.random.default_rng(4990)
        sw = C.tm(74.375)
        for j in range(SWARM_N):
            out.append(dict(te=sw - .1 + .08 * rng.random(), tf=CLEAR_T - .06, tp=None, tr=None, tc=CLEAR_T,
                            th=j / SWARM_N * math.tau + rng.uniform(-.2, .2), D=rng.uniform(30, 95),
                            stop=rng.uniform(18, 26), wob=rng.uniform(0, math.tau), curl=rng.uniform(-1, 1),
                            swarm=True, seed=4990 + j))
        _LASH = out
    return _LASH


def t_eff(L, t):
    """Its own time: held (as in a frame) from the projection to letting go."""
    if L["tp"] is None or t < L["tp"]:
        return t
    return L["tp"] if t < L["tr"] else t - (L["tr"] - L["tp"])


def lash_root(L, t):
    """Where its root is - in the wall, which goes by with the world (slower; held with it)."""
    ref = L["tp"] if L["tp"] is not None else L["tc"]
    return np.array([RW * math.cos(L["th"]), RW * math.sin(L["th"]),
                     ZS + L["D"] + ROOT_FLOW * (dist(ref) - dist(t_eff(L, t)))])


def lash_tip(L, t):
    if not L["swarm"]:
        return L["tip"]
    u = lash_root(L, L["tc"]) - soul3(L["tc"])
    return soul3(t) + u / (np.linalg.norm(u) or 1) * L["stop"]


def lash_reach(L, t):
    """How far out of the wall it is: it shoots out, straight at it - a hair past, and back."""
    a = clamp((t - L["te"]) / (L["tf"] - L["te"]))
    return 1 + 2.4 * (a - 1) ** 3 + 1.4 * (a - 1) ** 2


def lash_spine(L, t, e=None):
    """Its spine (root to as far as it reaches) and how thick it is along it: out of the wall,
    taut - all but straight at its tip, which is a point; held, a tremble in it."""
    e = lash_reach(L, t) if e is None else e
    tt = t_eff(L, t)
    P0, T = lash_root(L, t), lash_tip(L, t)
    inn = -np.array([math.cos(L["th"]), math.sin(L["th"]), 0.0])
    n1, n2 = _perps(T - P0)
    side = n1 * math.cos(L["wob"]) + n2 * math.sin(L["wob"])
    P1 = P0 + inn * 24 - np.array([0.0, 0.0, 6.0])
    P2 = T + (P0 - T) * .3 + side * 6 * L["curl"]
    held = L["tp"] is not None and L["tp"] <= t < L["tc"]
    trem = .45 * math.sin(t * 95 + L["wob"]) if held else 0.0
    pts, wid = [], []
    for v in np.linspace(0, 1, 26):
        w = (.3 + 1.4 * max(0.0, 1 - e)) * math.sin(math.pi * v) * math.sin(math.tau * .7 * v - tt * 7 + L["wob"])
        pts.append(bez(P0, P1, P2, T, v * e) + n1 * (w + trem * v))
        wid.append(.25 + 5.4 * (1 - v) ** .8)
    return pts, wid


CUTS = (11, 19)                                              # (of its 26: where it is cut - in three)


def _wall_frame(L):
    """Two ways along the wall at its root (round it, along it)."""
    return np.array([-math.sin(L["th"]), math.cos(L["th"]), 0.0]), np.array([0.0, 0.0, 1.0])


def _wall_ring(pen, L, t, r, v, w=1.0, n=20):
    C0 = lash_root(L, t)
    e1, e2 = _wall_frame(L)
    pts = [C0 + (e1 * math.cos(x) + e2 * math.sin(x)) * r for x in np.linspace(0, math.tau, n + 1)]
    for A, B in zip(pts, pts[1:]):
        pen.line(A, B, v, w)


def lash_pieces(L, t):
    """Once it is cut: its bits (spines, widths, how faded) - the end flung off, spinning, the
    middle after it, falling behind; the stump drawn back into the wall."""
    a = t - L["tc"]
    pts, wid = lash_spine(L, L["tc"], 1.0)
    S0 = L.get("ref")
    S0 = soul3(L["tc"]) if S0 is None else S0
    rng = np.random.default_rng(L["seed"])
    out = []
    k = max(2, int(CUTS[0] * (1 - smooth(clamp(a / .2)))) + 1)    # the stump, going back in
    drift = lash_root(L, t) - lash_root(L, L["tc"])
    out.append(([P + drift for P in pts[:k]], wid[:k], 1 - clamp((a - .12) / .12)))
    bounds = CUTS + (len(pts),)
    for j in range(len(CUTS)):
        i0, i1 = bounds[j] - 1, bounds[j + 1]
        seg = np.array(pts[i0:i1])
        c0 = seg.mean(0)
        o = c0 - S0
        o = o / (np.linalg.norm(o) or 1)
        vel = o * rng.uniform(90, 150) * (1 + .6 * j) + rng.normal(size=3) * 25
        ax = rng.normal(size=3)
        ax = ax / np.linalg.norm(ax)
        ang = rng.uniform(8, 14) * (1 + .5 * j) * a * rng.choice([-1, 1])
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        Rm = np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K
        c1 = c0 + vel * a * (1 - .35 * a) - np.array([0.0, 0.0, .5 * (dist(t) - dist(L["tc"]))])
        out.append(([c1 + Rm @ (P - c0) for P in seg], wid[i0:i1], 1 - clamp((a - .1) / .25)))
    return out


def lashing(pen, t, front):
    """M71-M74.5: the tentacles. Where one will come out, the tunnel's wall cracks first (a ring,
    flashing); then it shoots out of it - bits of the wall flung off, a hole left - straight at the
    soul, taut, its point stopping short (the stars': held there till it lets its cuts go); cut,
    it comes apart in three - the end flung off spinning, the middle after it, the stump drawn back
    into the wall. (front: only what is nearer us than the soul - the rest, before it is drawn.)"""
    zs = pen.cam.to_cam(soul3(t))[2]
    mine = lambda P: (pen.cam.to_cam(P)[2] < zs) == front    # (this pass's: in front of it, or behind it)
    for L in lashers():
        if not L["te"] - LASH_WARN <= t < L["tc"] + .4:
            continue
        root = lash_root(L, t)
        if t < L["te"]:                                      # the wall cracking where it will come out
            if not mine(root):
                continue
            b = (t - (L["te"] - LASH_WARN)) / LASH_WARN
            v = 210 if (b > .5 and int(t * 30) % 2) else 130
            _wall_ring(pen, L, t, 3 + 5 * b, v * min(1.0, b * 4), 1.0)
            e1, e2 = _wall_frame(L)
            for j in range(5):
                x = j / 5 * math.tau + L["wob"]
                d = e1 * math.cos(x) + e2 * math.sin(x)
                pen.line(root + d * 2, root + d * (4 + 9 * b), v * .8 * min(1.0, b * 4), .5)
            continue
        if mine(root) and pen.cam.to_cam(root)[2] > NEAR:   # the hole it came out of (the wall: not from beside it)
            _wall_ring(pen, L, t, 6, 110 * (1 - pen.cam.ws), .8)
        a = t - L["te"]
        if a < .2 and mine(root):                            # bits of the wall flung off as it bursts out
            rng = np.random.default_rng(L["seed"] + 500)
            inn = -np.array([math.cos(L["th"]), math.sin(L["th"]), 0.0])
            for j in range(8):
                vel = inn * rng.uniform(50, 130) + rng.normal(size=3) * 45
                q = pen.cam.p(root + vel * a)
                if q:
                    r = max(1.0, rng.uniform(.6, 1.4) * q[2])
                    pen.d.polygon([(q[0] - r, q[1] + r * .6), (q[0] + r * .3, q[1] - r), (q[0] + r, q[1] + r * .4)],
                                  fill=int(210 * (1 - a / .2)))
            _wall_ring(pen, L, t, 6 + 26 * (1 - (1 - a / .2) ** 2), 200 * (1 - a / .2), 1.2)
        if t < L["tc"]:
            pts, wid = lash_spine(L, t)
            if mine(pts[-1]):
                tube(pen, pts, wid, body=228, rim=255, suckers=False)
            continue
        for k, (pts, wid, g) in enumerate(lash_pieces(L, t)):   # cut
            c = np.mean(pts, 0)
            g *= clamp((pen.cam.to_cam(c)[2] - 45) / 40)     # (gone before it is right up against us)
            if g <= .02 or not mine(c):
                continue
            tube(pen, pts, wid, body=228, rim=255, suckers=False, g=g, near=40.0)
        a = t - L["tc"]
        if a < .25 and front:                                # (ink thrown off the cuts)
            rng = np.random.default_rng(L["seed"] + 900)
            pts, _ = lash_spine(L, L["tc"], 1.0)
            for j in range(6):
                P0 = pts[CUTS[j % 2]]
                v = rng.normal(size=3) * 70
                q = pen.cam.p(P0 + v * a - np.array([0.0, 0.0, .5 * (dist(t) - dist(L["tc"]))]))
                if q:
                    r = max(1.0, rng.uniform(.3, .7) * q[2])
                    pen.d.ellipse([q[0] - r, q[1] - r, q[0] + r, q[1] + r], fill=int(220 * (1 - a / .25)))


def slash_lines(d_w, d_r, t, cam):
    """Each cut: a hairline of light across the tentacle, drawn through it in a flash - long,
    keen, a thread of red along it (its blood) - and gone."""
    for L in lashers():
        a = t - L["tc"]
        if not 0 <= a < .16:
            continue
        pts, _ = lash_spine(L, L["tc"], 1.0)
        rng = np.random.default_rng(L["seed"] + 77)
        for j, (i, half, lag) in enumerate(((CUTS[1], 62, 0.0), (CUTS[0], 34, .015))):
            b = a - lag
            if b < 0:
                continue
            q0, q1 = cam.p(pts[i - 1]), cam.p(pts[i + 1])
            if q0 is None or q1 is None:
                continue
            ang = math.atan2(q1[1] - q0[1], q1[0] - q0[0]) + math.pi / 2 + rng.uniform(-.4, .4)
            cx, cy = (q0[0] + q1[0]) / 2, (q0[1] + q1[1]) / 2
            ux, uy = math.cos(ang), math.sin(ang)
            ext = min(1.0, b / .025)
            hl = half * (.5 if L["swarm"] else 1.0)
            A = (cx - ux * hl, cy - uy * hl)
            B = (cx - ux * hl + 2 * ux * hl * ext, cy - uy * hl + 2 * uy * hl * ext)
            g = 1 - clamp((b - .03) / .12)
            d_w.line([A, B], fill=int(70 * g), width=4)
            d_w.line([A, B], fill=int(255 * g), width=1)
            d_r.line([(A[0] - uy * 2, A[1] + ux * 2), (B[0] - uy * 2, B[1] + ux * 2)], fill=int(230 * g), width=1)


def ghost_hearts(out, t, cam, R):
    """The projection's ghosts of it (one a frame, where it will cut each), and as it lets them go
    the trail it leaves through them - its heart, faint, laid on."""
    lay = np.zeros((H, W), np.uint8)
    acc = np.zeros((H, W), np.float32)
    for pj in projections():
        if pj["tp"] <= t < pj["tr"]:
            for G, tk in zip(pj["ghosts"], pj["shows"]):
                if t < tk:
                    continue
                lay[:] = 0
                bga.big_heart(lay, cam.to_cam(np.append(G, soul_z(t))), R, cam, t, SOUL)
                acc += lay * (.7 if t - tk < .034 else .42)
        a = t - pj["tr"]
        if 0 <= a < DASH + .14:                              # (the trail: where it was, a moment ago - fading)
            fade = 1 - clamp((a - DASH) / .14)
            for j in range(1, 9):
                tj = t - j * .011
                if tj < pj["tr"] - .005:
                    break
                lay[:] = 0
                bga.big_heart(lay, cam.to_cam(soul3(tj)), R, cam, t, SOUL)
                acc += lay * .5 * (1 - j / 9) * fade
    if not acc.any():
        return out
    return np.clip(out + acc, 0, 255).astype(np.uint8)


def dash_streak(d_w, t, cam):
    """Letting go: a streak of light along its way through its ghosts, as it goes, fading."""
    for pj in projections():
        a = t - pj["tr"]
        if not 0 <= a < DASH + .16:
            continue
        u = clamp(a / DASH)
        g = 1 - clamp((a - DASH) / .16)
        pts = [along(pj["path"], u * i / 24) for i in range(25)]
        q = [cam.p(np.append(x, soul_z(t))) for x in pts]
        q = [x[:2] for x in q if x is not None]
        if len(q) > 1:
            d_w.line(q, fill=int(60 * g), width=7)
            d_w.line(q, fill=int(250 * g), width=2)


def proj_ui(d, dk, dm, t, cam):
    """The projection: round each ghost a frame (flashing as it comes, one a frame), the way it will
    go from one to the next (dashed), a key over each; let go: the keys all pressed at once - a
    flash, they break - the frames gone in a flash. (dk, dm: the keys and where they cover what is under them.)"""
    for pj in projections():
        if not pj["tp"] <= t < pj["tr"] + .2:
            continue
        prev = cam.p(np.append(pj["home"], soul_z(t)))
        for k, (G, key, tk) in enumerate(zip(pj["ghosts"], pj["keys"], pj["shows"])):
            if t < tk:
                break
            q = cam.p(np.append(G, soul_z(t)))
            if q is None:
                continue
            sz = q[2] * 16
            fw, fh = sz * .82, sz * .7
            ky = q[1] - fh - 17
            if t < pj["tr"]:
                fresh = t - tk < .034
                d.rectangle([q[0] - fw, q[1] - fh, q[0] + fw, q[1] + fh], outline=255 if fresh else 165,
                            width=2 if fresh else 1)
                for sx in (-1, 1):                           # (the frame's sprocket marks)
                    for yy in (q[1] - fh + 4, q[1] + fh - 8):
                        x0 = q[0] + sx * (fw - 7)
                        d.rectangle([x0 - 2, yy, x0 + 2, yy + 4], outline=120)
                if prev is not None:                         # (the way from the last: dashed)
                    dx, dy = q[0] - prev[0], q[1] - prev[1]
                    L_ = math.hypot(dx, dy)
                    for f in np.arange(0, L_, 11):
                        f1 = min(L_, f + 6)
                        d.line([(prev[0] + dx * f / L_, prev[1] + dy * f / L_),
                                (prev[0] + dx * f1 / L_, prev[1] + dy * f1 / L_)], fill=120, width=1)
                r = 13 * .9
                dm.rectangle([q[0] - r - 1, ky - r - 1, q[0] + r + 1, ky + r + 1], fill=255)
                key_icon(dk, q[0], ky, key, 240 if fresh else 220, .9)
            else:                                            # let go: all pressed at once
                b = (t - pj["tr"]) / .2
                if b < .17:
                    d.rectangle([q[0] - fw, q[1] - fh, q[0] + fw, q[1] + fh], outline=255, width=2)
                if b < .5:
                    r = 13 * .9 * (1 + .5 * b)
                    dm.rectangle([q[0] - r - 1, ky - r - 1, q[0] + r + 1, ky + r + 1], fill=255)
                key_icon(dk, q[0], ky, key, int(255 * (1 - b)), .9 * (1 + .5 * b))
                for j in range(6):
                    a2 = j / 6 * math.tau + k
                    px, py = q[0] + math.cos(a2) * (10 + 26 * b), ky + math.sin(a2) * (10 + 26 * b)
                    r = 2.5 * (1 - b)
                    d.rectangle([px - r, py - r, px + r, py + r], fill=int(255 * (1 - b)))
            prev = q


def clear_slash(d_w, d_r, t, cam):
    """The break: one cut all the way round it - a keen ring of light, a thread of red along it
    (its blood), swept round in a flash, gone."""
    a = t - CLEAR_T
    c = cam.p(soul3(t))
    if c is None or not -.01 <= a < .2:
        return
    rho = c[2] * 16 * 1.3
    head = clamp((a + .01) / .05)
    g = 1 - clamp((a - .05) / .15)
    n = 64
    ring, ring_r = [], []
    for i in range(n + 1):
        ang = -math.pi / 2 + math.tau * head * i / n
        ring.append((c[0] + rho * math.cos(ang), c[1] + rho * math.sin(ang)))
        ring_r.append((c[0] + (rho + 3) * math.cos(ang), c[1] + (rho + 3) * math.sin(ang)))
    d_r.line(ring_r, fill=int(230 * g), width=1)
    d_w.line(ring, fill=int(70 * g), width=5)
    d_w.line(ring, fill=int(255 * g), width=2)


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
        v *= smooth(clamp((zs + Z_FAR - z) / 500.0))       # (out of the far dark: fade in, don't pop)
        c, s_ = math.cos(ph), math.sin(ph)
        pts = []
        for u in np.linspace(0, math.tau, 41):
            x, y = rad * math.cos(u), rad * math.sin(u)
            pts.append(np.array([off[0] + x, off[1] + y, z + (x * c + y * s_) * tilt]))
        for A, B in zip(pts, pts[1:]):
            dz = pen.cam.to_cam((A + B) / 2)[2]              # (the near side of it, going by us: gone before it is
            vv = v * clamp((dz - 25) / 35)                   # right up against us)
            pen.line(A, B, vv, 1.6, add=True)
            pen.line(A, B, vv * .35, 4.0, add=True)


# ------------------------------------------------------------------ the prompts
_FONT = {}


def font(size):
    if size not in _FONT:
        _FONT[size] = ImageFont.truetype(str(FONTS / "fusion-pixel-12px.ttf"), size)
    return _FONT[size]


ARROWS = {(-1, 0): "left", (1, 0): "right", (0, -1): "up", (0, 1): "down"}


def prompts():
    """(when, what, where - off the soul, on screen - how long its ring takes)."""
    out = [(X_T, "down", (0, 1), LEAD, None)]
    out += [(tn, ARROWS[d], d, LEAD, None) for tn, d in SIDE]
    out += [(tn, TURN_KEYS[w][0], None, .4, TURN_KEYS[w][1]) for tn, w in turns()]   # (the sweeps' turns: at the side)
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
    g_ = gaps()
    gk = clamp((t - (g_[0][0] - .7)) / .2) * (1 - clamp((t - (g_[-1][0] + .15)) / .2))
    if gk > 0:                                               # (the sweeps: the two keys, at the sides, waiting)
        for glyph, (x, y) in TURN_KEYS.values():
            key_icon(d, x, y, glyph, int(105 * gk))
    nxt = min([p[0] for p in prompts() if p[0] >= t] or [1e9])
    for tn, glyph, off, lead, pos in prompts():
        a = t - (tn - lead)
        if a < 0 or t > tn + .18:
            continue
        dimq = 1.0 if tn <= nxt + 1e-6 or pos else .4         # (not its turn yet: dim)
        if pos is not None:                                  # (at its place, at the side)
            x, y = pos
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
        if s[1] < H * .45:                                   # (it high on the screen, looking up at it: under it)
            y = s[1] + off_px + 8
        key_icon(d, x, y, "Z", int(235 * g), 1 + .18 * pulse)
        f = font(12)
        d.text((x + 20, y - 7), "连打!", font=f, fill=int(220 * g))
        bw = 64
        d.rectangle([x - bw / 2, y + 20, x + bw / 2, y + 26], outline=int(200 * g), width=1)
        d.rectangle([x - bw / 2 + 1, y + 21, x - bw / 2 + 1 + (bw - 2) * n / max(1, len(hits)), y + 25],
                    fill=int(235 * g))


# ------------------------------------------------------------------ the frame
def frame(t: float) -> np.ndarray:
    if t >= T_LEAP:                                          # the last break: it leaps out
        return bga.leap_frame(dict(bga.LEAP, t0=T_LEAP, t1=T_END), t)
    cam = Cam(t)
    back = tunnel_back(t, cam) + surface(t, cam)
    out = np.clip(back, 0, 255).astype(np.uint8).repeat(2, 0).repeat(2, 1)
    im, gl = Image.fromarray(out), Image.new("L", (W, H), 0)
    pen = Pen(ImageDraw.Draw(im), cam, ImageDraw.Draw(gl), img=im)
    tunnel(pen, t)
    rise_motes(pen, t)
    slab_blocks(pen, t)
    curtains(pen, t)
    dodge_orbs(pen, t)
    x_blades(pen, t)
    c_rings(pen, t)
    bars(pen, t)
    hoops(pen, t)
    lunges(pen, t)
    lashing(pen, t, False)                                   # (the tentacles behind it)
    out = np.clip(np.asarray(im).astype(np.int16) + np.asarray(gl), 0, 255).astype(np.uint8)
    xy = soul_xy(t)                                          # the soul: facing down the way it came, leaning into its turns
    vxy = (soul_xy(t + .03) - soul_xy(t - .03)) / .06
    lean = clamp(vxy[0] * .0012, -.5, .5)
    Rw = bga._rz(lean * .4) @ bga._ry(-lean) @ np.diag([1, 1, -1.0])
    S = soul3(t)
    wf = cam.wg
    if wf > 0:                                               # (the build: fixed in the world - each shot sees it from its
        nrm, top = build_heart(t)                            # own side, its top up the screen; as the camera swings
        nrm = Rw[:, 2] * (1 - wf) + nrm * wf                 # round behind it on the drop, facing down the way it came)
        nrm = nrm / np.linalg.norm(nrm)
        v = Rw[:, 1] * (1 - wf) - top * wf
        v = v - (v @ nrm) * nrm
        v = v / np.linalg.norm(v)
        Rw = np.column_stack([np.cross(nrm, v), v, nrm])
    if t > ORBIT[0]:                                         # (the last of it: tipping its top over towards the way
        _, b = orbit(t)                                      # ahead, round its x - square to the rings at the end;
        sl = math.radians(AIM_SLANT) * smooth(clamp((t - AIM[0]) / (AIM[1] - AIM[0])))   # torn free, aslant)
        top = np.array([0.0, -math.cos(b), math.sin(b)]) * math.cos(sl) + np.array([math.sin(sl), 0.0, 0.0])
        nrm = np.array([0.0, -math.sin(b), -math.cos(b)])
        k = smooth(clamp((t - ORBIT[0]) / .3))               # (from how it was: no jump)
        v = Rw[:, 1] * (1 - k) - top * k
        v = v / np.linalg.norm(v)
        nrm = Rw[:, 2] * (1 - k) + nrm * k
        nrm = nrm - (nrm @ v) * v
        nrm = nrm / np.linalg.norm(nrm)
        Rw = np.column_stack([np.cross(nrm, v), v, nrm])
    R = cam.R @ Rw
    P = cam.to_cam(S)
    c = cam.pr(P + R[:, 1] * .05 * 16)
    out = bga.halo(out, c, t, SOUL)
    out = ghost_hearts(out, t, cam, R)                       # (the stars: its projected ghosts; the trail as it goes)
    im = Image.fromarray(out)
    sights(ImageDraw.Draw(im), t, cam)                       # (under it: it is on the sight)
    out = np.asarray(im).copy()
    quad = bga.big_heart(out, P, R, cam, t, SOUL)
    im = Image.fromarray(out)
    pen = Pen(ImageDraw.Draw(im), cam, img=im)
    paths(pen, t)                                            # (over it: what it goes along, the ink at it)
    lashing(pen, t, True)
    red, lit = Image.new("L", (W, H), 0), Image.new("L", (W, H), 0)   # (the cuts: light laid on - fading out, not
    clear_slash(ImageDraw.Draw(lit), ImageDraw.Draw(red), t, cam)       # going dark)
    slash_lines(ImageDraw.Draw(lit), ImageDraw.Draw(red), t, cam)
    dash_streak(ImageDraw.Draw(lit), t, cam)
    omen = bga.omen_red(t, SOUL, c, quad)
    cut = np.asarray(red, np.float32) / 255
    out = bga.tint(np.asarray(im), cut if omen is None else np.maximum(omen, cut))
    out = np.maximum(out, np.asarray(lit)[..., None])
    ui = Image.new("L", (W, H), 0)
    du = ImageDraw.Draw(ui)
    draw_prompts(du, t, cam)
    keys, kmask = Image.new("L", (W, H), 0), Image.new("L", (W, H), 0)
    proj_ui(du, ImageDraw.Draw(keys), ImageDraw.Draw(kmask), t, cam)
    out = np.maximum(out, np.asarray(ui)[..., None])
    m = np.asarray(kmask) > 0                                # (the keys: solid, over whatever is there)
    if m.any():
        out = np.where(m[..., None], np.asarray(keys)[..., None], out)
    return out
