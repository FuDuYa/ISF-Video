"""Turn the maimai chart into the phase-four attack timeline.

Mapping (maimai -> battle):
  * the 8 maimai buttons sit on a ring inside the arena; button k keeps its
    angle (1 = just right of 12 o'clock, clockwise)
  * slide  -> a brush paints the slide's path across the arena (the slide's
              one-beat star wait is the warning time)
  * hold   -> an ink well opens on that side and pours a beam for the hold length
  * tap    -> section-dependent: edge spike / falling drip / ink dart / bone
  * break  -> a splash ring of droplets plus screen shake
Each section of the song chooses its arena size and tap style.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import actions
import attacks as A
from chart import parse_chart, Note
from common import ROOT, clamp, key_angle, key_vec, smooth

BPM = 202.0
MEAS = 240.0 / BPM
BEAT = 60.0 / BPM
OFFSET = -0.010             # chart -> audio alignment, measured against onsets
BOX_BOTTOM = 388            # outer bottom edge of the arena stays fixed (UT style)
BORDER = 5
SONG_END = 147.0


def tm(m: float) -> float:
    return m * MEAS + OFFSET


TITLE_END = tm(16.25) + BEAT + MEAS * 3 / 8     # the title card ends with its two slides
# M15 triplet run: the pool rises on every note and engulfs the arena on the 7th;
# the frame soaks through on notes 7-9, the inside turns to paper on notes 10-12,
# and on the M16 downbeat the canvas is set (title card from M16.125)
TRIP_T = [tm(15 + k / 12) for k in range(12)]
ENGULF = TRIP_T[6]
SOAK_STEPS = TRIP_T[6:9]
PAPER_STEPS = TRIP_T[9:12]
CANVAS_SET = tm(16)
CANVAS = (lambda: (CANVAS_SET, tm(28.5)))      # from here on the arena is a canvas
# the rest after the cube (M25-M28.5): the canvas wraps the buttons too. The burst
# throws the soul a little left of the middle; from there it heads for FIGHT the
# whole time (arrow keys held, 8 directions - no turn is given) while the ink of
# the burst comes alive, rings it and closes in on it beat by beat. Touching a
# button's edge is enough to trigger it: the soul grazes FIGHT's right edge only
# on the M28 downbeat (the hardest hit of the long hold), just as the ring is
# about to touch it: one press, and one slash cuts the whole ring
REST_RECT = (-150, 226, 790, 630)              # the sheet runs on past the screen (scene.overscan)
# where the soul touches FIGHT: walking straight down-left from KNOCK_TO, the
# first spot where the heart overlaps fight1.png (at 32,432) - by one pixel, on
# the middle of its right edge (the step before, it is still clear of it)
FIGHT_SOUL = (149, 451)
PRESS_T = tm(28)
DEEP_T = tm(22.75) - .08     # the camera is deepest inside the canvas: the rectangle is there from now on (cinema)
KNOCK_DUR = .4
KNOCK_TO = (270, 330)                           # thrown just left of the middle
WALK_T = tm(24.75) + .42
_WALK_STEPS = int(PRESS_T * 60) - int(round(WALK_T * 60))
# from that close it walks much slower than UT's 2 px a frame (about 0.59), so
# that it still touches the button only on PRESS_T
WALK_SPEED = max(abs(KNOCK_TO[0] - FIGHT_SOUL[0]), abs(KNOCK_TO[1] - FIGHT_SOUL[1])) / _WALK_STEPS
WALK_ARRIVE = (int(round(WALK_T * 60)) + _WALK_STEPS) / 60
SOUL_SCRIPT = [dict(kind="fly", t0=tm(24.75), dur=KNOCK_DUR, to=KNOCK_TO, hold=WALK_T),
               dict(kind="walk", t0=WALK_T, to=FIGHT_SOUL, hold=tm(28.5), speed=WALK_SPEED)]
CLOSE_IN = [tm(25.75 + k / 4) for k in range(9)]   # the ring lunges in on every beat up to the press
FIGHT_SELECT = (WALK_ARRIVE, tm(28.5))
PRESSES = [PRESS_T]                             # FIGHT pressed once, the moment its edge is touched


# --------------------------------------------------------------------- sections
# m: first measure; box: outer (w, h); tap/hold: styles; head: Ink Sans head
SECTIONS = [
    dict(m=0, box=(140, 140), tap="none", hold="none", head="head_normal.png"),
    # 7.3-10.1 s: no attacks; the summoned brush paints two bars above the arena
    dict(m=6, box=(140, 140), tap="none", hold="none", head="head_close_eyes.png"),
    # first sixteenth run: eyes open, the bars turn into bones
    dict(m=8.5, box=(140, 140), tap="none", hold="none", head="head_black_eyes.png"),
    # bones shatter; resting passage: no attacks, background mood only (backdrop.py)
    dict(m=9, box=(140, 140), tap="none", hold="none", slide="bg", head="head_normal.png"),
    # "那就让我亲手把你拉进地狱里吧。" - black eye sockets
    dict(m=11, box=(140, 140), tap="none", hold="none", slide="bg", head="head_black_eyes.png"),
    # the arena stretches wide on the break star; its soaked top edge rains ink
    # (each half measure adds a new attack, see add_specials)
    dict(m=13, box=(360, 150), tap="rain", hold="beam", slide="wave", head="head_black_eyes.png",
         body="body.png", lead=MEAS / 4, dur=.28, ease="back"),
    # triplet run: flicked ink sprays plus heavier rain
    dict(m=15, box=(360, 150), tap="flick", hold="beam", slide="wave", head="head_black_eyes.png",
         body="body.png"),
    # eighths: rhythm-game style title card (cinema.py), no attacks
    dict(m=16, box=(360, 150), tap="none", hold="none", slide="none", head="head_black_eyes.png",
         body="body.png"),
    # M17-M24 on the paper canvas: Ink Sans paints at the soul without pause
    # (attacks are built by add_battle; every phrase has its own brush motif)
    dict(m=17, box=(200, 150), tap="custom", hold="none", slide="custom", head="head_black_eyes.png",
         body="body.png", lead=0.0, dur=.12),
    dict(m=21, box=(200, 150), tap="custom", hold="none", slide="custom", head="head_black.png",
         body="body.png"),
    dict(m=23, box=(200, 150), tap="custom", hold="none", slide="custom", head="head_black_eyes.png",
         body="body.png"),
    # the ensō closes the fight; resting passage
    dict(m=25, box=(200, 150), tap="none", hold="none", slide="none", head="head_close_eyes.png",
         body="body.png"),
    # (placeholder until the next part is directed: no attacks yet)
    dict(m=28, box=(200, 150), tap="none", hold="none", slide="none", head="head_close_eyes.png",
         body="body.png"),
    dict(m=29, box=(260, 130), tap="drip", hold="hbeam", head="head_normal.png"),
    dict(m=33, box=(150, 150), tap="spike", hold="beam", head="head_black.png"),
    dict(m=39, box=(300, 130), tap="bone", hold="hbeam", head="head_black_eyes.png"),
    dict(m=42, box=(260, 140), tap="dart", hold="hbeam", head="head_black_eyes.png"),
    dict(m=46, box=(160, 150), tap="drip", hold="beam", head="head_black_eyes.png"),
    dict(m=48, box=(150, 150), tap="slowdrip", hold="beam", head="head_close_eyes.png"),
    dict(m=54, box=(190, 150), tap="spike", hold="beam", head="head_normal.png"),
    dict(m=55, box=(190, 150), tap="spike", hold="beam", head="head_black_eyes.png"),
    dict(m=63, box=(160, 150), tap="spike", hold="beam", head="head_black.png"),
    dict(m=71, box=(230, 150), tap="drip", hold="beam", head="head_black.png"),
    dict(m=75, box=(190, 150), tap="spike", hold="beam", head="head_black_eyes.png"),
    dict(m=80, box=(190, 150), tap="spike", hold="beam", head="head_black.png"),
    dict(m=90, box=(150, 150), tap="spike", hold="beam", head="head_black.png"),
    dict(m=97, box=(200, 150), tap="dart", hold="beam", head="head_black_eyes.png"),
    dict(m=103, box=(200, 150), tap="slowdrip", hold="beam", head="head_close_eyes.png"),
    dict(m=104, box=(170, 150), tap="spike", hold="beam", head="head_black.png"),
    dict(m=109, box=(170, 150), tap="dart", hold="beam", head="head_black.png"),
    dict(m=111, box=(170, 150), tap="spike", hold="beam", head="head_black_eyes.png"),
    dict(m=118, box=(140, 140), tap="drip", hold="beam", head="head_close_eyes.png"),
]
for s in SECTIONS:
    s["t"] = tm(s["m"])
SECTION_TIMES = [s["t"] for s in SECTIONS]
BOX_TRANSITION = .32


def section_at(t: float) -> dict:
    i = bisect.bisect_right(SECTION_TIMES, t + 1e-6) - 1
    return SECTIONS[max(0, i)]


def box_rect(size) -> tuple[int, int, int, int]:
    w, h = size
    x = 320 - w // 2
    y = BOX_BOTTOM - h
    return x, y, w, h


def _trans(s):
    lead = s.get("lead", BOX_TRANSITION)
    return s["t"] - lead, s.get("dur", BOX_TRANSITION)


TRANS_STARTS = [_trans(s)[0] for s in SECTIONS]


def box_at(t: float) -> tuple[int, int, int, int]:
    """Arena rectangle; each section eases into its size (default: the
    BOX_TRANSITION before it starts; sections may set their own lead/dur)."""
    i = max(0, bisect.bisect_right(TRANS_STARTS, t) - 1)
    cur = SECTIONS[i]
    prev = SECTIONS[i - 1] if i > 0 else cur
    start, dur = _trans(cur)
    v = clamp((t - start) / max(dur, 1e-6))
    if cur.get("ease") == "back":            # overshoot, then settle
        u = 1 + 2.4 * (v - 1) ** 3 + 1.4 * (v - 1) ** 2
    else:
        u = smooth(v)
    w = prev["box"][0] + (cur["box"][0] - prev["box"][0]) * u
    h = prev["box"][1] + (cur["box"][1] - prev["box"][1]) * u
    w = int(round(w / 2)) * 2
    h = int(round(h / 2)) * 2
    return box_rect((w, h))


# --------------------------------------------------------------------- geometry
class Geo:
    def __init__(self, size):
        self.x, self.y, self.w, self.h = box_rect(size)
        self.ix0, self.iy0 = self.x + BORDER, self.y + BORDER
        self.ix1, self.iy1 = self.x + self.w - BORDER, self.y + self.h - BORDER
        self.iw, self.ih = self.ix1 - self.ix0, self.iy1 - self.iy0
        self.cx, self.cy = (self.ix0 + self.ix1) / 2, (self.iy0 + self.iy1) / 2
        self.rx, self.ry = self.iw / 2 - 16, self.ih / 2 - 16

    def ring(self, x, y):
        return (self.cx + x * self.rx, self.cy + y * self.ry)

    def key_point(self, k, r=1.0):
        ux, uy = key_vec(k)
        return self.ring(ux * r, uy * r)

    def edge(self, k):
        ux, uy = key_vec(k)
        tx = (self.iw / 2) / abs(ux) if abs(ux) > 1e-9 else 1e9
        ty = (self.ih / 2) / abs(uy) if abs(uy) > 1e-9 else 1e9
        t = min(tx, ty)
        p = (self.cx + ux * t, self.cy + uy * t)
        n = (-math.copysign(1, ux), 0.0) if tx < ty else (0.0, -math.copysign(1, uy))
        return p, n

    def outside(self, k, extra=36):
        ux, uy = key_vec(k)
        p, _ = self.edge(k)
        return (p[0] + ux * extra, p[1] + uy * extra)

    RAIN_COL = {5: 0, 6: 1, 7: 2, 8: 3, 1: 4, 2: 5, 3: 6, 4: 7}

    def rain_x(self, k):
        """Eight columns across the arena, left side 5-8, right side 1-4 (mirror pairs)."""
        col = self.RAIN_COL[k]
        return self.ix0 + (col + .5) * self.iw / 8

    def drip_x(self, k):
        s = math.sin(key_angle(k))
        f = .9 * s if k in (1, 2, 7, 8) else .52 * s
        return self.cx + f * (self.iw / 2 - 14)


# --------------------------------------------------------------------- slide paths
def _unit(k):
    return np.array(key_vec(k))


def _arc(a0, a1, r=1.0, c=(0.0, 0.0)):
    n = max(8, int(abs(a1 - a0) / .06))
    a = np.linspace(a0, a1, n)
    return np.stack([c[0] + np.sin(a) * r, c[1] - np.cos(a) * r], 1)


def slide_paths(start: int, segments) -> list[np.ndarray]:
    """Normalised slide geometry on the unit ring (y down)."""
    paths = []
    cur = [_unit(start)]
    k = start
    for shape, end in segments:
        a0, a1 = key_angle(k), key_angle(end)
        P0, P1 = _unit(k), _unit(end)
        if shape == "-":
            seg = np.stack([P0, P1])
        elif shape in "<>":
            cw = (shape == ">") == (k in (1, 2, 7, 8))
            if cw:
                d = (a1 - a0) % math.tau or math.tau
            else:
                d = -((a0 - a1) % math.tau or math.tau)
            seg = _arc(a0, a0 + d)
        elif shape == "^":
            d = (a1 - a0 + math.pi) % math.tau - math.pi
            seg = _arc(a0, a0 + d)
        elif shape == "v":
            seg = np.stack([P0, [0, 0], P1])
        elif shape in ("p", "q", "pp", "qq"):
            big = len(shape) == 2
            r = .55 if big else .38
            cw = shape[0] == "q"
            off = math.radians(35 if big else 70)
            s0 = a0 + (off if cw else -off)
            s1 = a1 - (off if cw else -off)
            if cw:
                d = (s1 - s0) % math.tau
                if big and d < math.pi:
                    d += math.tau
            else:
                d = -((s0 - s1) % math.tau)
                if big and -d < math.pi:
                    d -= math.tau
            ctr = (0.0, 0.0) if not big else tuple(_unit(k) * -.0)
            arc = _arc(s0, s0 + d, r, ctr)
            seg = np.vstack([P0, arc, P1])
        elif shape in ("s", "z"):
            v = P1 - P0
            perp = np.array([-v[1], v[0]]) / (np.linalg.norm(v) + 1e-9)
            sg = 1 if shape == "s" else -1
            m1 = P0 + v * .30 + perp * .55 * sg
            m2 = P0 + v * .70 - perp * .55 * sg
            seg = np.stack([P0, m1, m2, P1])
        elif shape == "w":
            for e in ((end - 2) % 8 + 1, end, end % 8 + 1):
                paths.append(np.stack([P0, _unit(e)]))
            k = end
            continue
        else:
            seg = np.stack([P0, P1])
        cur.extend(seg[1:])
        k = end
    if len(cur) > 1:
        paths.insert(0, np.array(cur))
    return paths


# --------------------------------------------------------------------- timeline
@dataclass
class Timeline:
    attacks: list = field(default_factory=list)
    swings: list = field(default_factory=list)      # times Ink Sans swings the brush
    flashes: list = field(default_factory=list)     # (time, luma)
    impacts: list = field(default_factory=list)     # (time, amplitude, seed)
    extra_impacts: list = field(default_factory=list)  # (time, amplitude) not tied to an attack

    def finalize(self):
        self.attacks.sort(key=lambda a: a.t0)
        self._t0 = [a.t0 for a in self.attacks]
        self.maxlen = max((a.t1 - a.t0 for a in self.attacks), default=0)
        for i, a in enumerate(self.attacks):
            if getattr(a, "shake", 0) and a.impact is not None:
                self.impacts.append((a.impact, a.shake, i))
        for k, (ti, amp) in enumerate(self.extra_impacts):
            self.impacts.append((ti, amp, 7919 + k))
        self.impacts.sort()
        self.swings.sort()

    def active(self, t):
        lo = bisect.bisect_left(self._t0, t - self.maxlen - .01)
        hi = bisect.bisect_right(self._t0, t)
        return [a for a in self.attacks[lo:hi] if a.t0 <= t <= a.t1]

    def shake(self, t):
        dx = dy = 0.0
        for ti, amp, seed in self.impacts:
            if ti > t:
                break
            tau = t - ti
            if tau < .3:
                e = amp * math.exp(-tau * 13)
                ang = seed * 2.399
                dx += math.cos(ang) * e * math.cos(tau * 70)
                dy += math.sin(ang) * e * math.cos(tau * 70)
        return int(round(dx)), int(round(dy))

    def flash(self, t):
        v = 0
        for ti, luma in self.flashes:
            if 0 <= t - ti < .16:
                v = max(v, int(luma * (1 - smooth((t - ti) / .16))))
        return v


AIM = None     # callable t -> (x, y) soul position from a previous plan, or None
CUBE = None    # the CanvasCube (the soul rides it in 3D; scene.compose asks it where to draw the soul)
FORCED_HITS = [(tm(20.75) + .14, 10)]   # scripted hits (time, damage): the spiral's shard
POOL = None    # PoolLevel of the ink rain (set by build)


def aim_at(t, g: "Geo"):
    """Where Ink Sans aims: the soul's (previously planned) position, else the centre."""
    if AIM is not None:
        p = AIM(t)
        if p is not None:
            x = min(max(p[0], g.ix0 + 10), g.ix1 - 10)
            y = min(max(p[1], g.iy0 + 10), g.iy1 - 10)
            return np.array([x, y], float)
    return np.array([g.cx, g.cy], float)


def ray_exit(g: "Geo", O, d):
    """Distance along d from O until the ray leaves the arena interior."""
    best = 1e9
    for lo, hi, o, v in ((g.ix0, g.ix1, O[0], d[0]), (g.iy0, g.iy1, O[1], d[1])):
        if abs(v) < 1e-9:
            continue
        t_hi = max((lo - o) / v, (hi - o) / v)
        best = min(best, t_hi)
    return best


def build(chart_path: Path | None = None) -> Timeline:
    notes, _ = parse_chart(chart_path or ROOT / "maidata.txt", OFFSET)
    tl = Timeline()
    times = sorted({round(n.t, 4) for n in notes})
    seed = 1000
    first_attack = tm(6)

    def nxt():
        nonlocal seed
        seed += 1
        return seed

    for n in notes:
        if n.t < first_attack - 1e-3:
            continue
        sec = section_at(n.t)
        if sec.get("tap") == "custom":
            continue                      # built by add_battle
        g = Geo(sec["box"])
        if n.kind == "slide" and sec.get("slide") in ("bg", "none"):
            continue                      # resting passage / title card: no attack
        if n.kind == "slide" and sec.get("slide") == "wave":
            for s in n.slides:
                tl.attacks.append(A.FloorWave(g.rain_x(s.start_key), g.rain_x(s.end_key),
                                              s.t_start, s.t_end, g.iy1, seed=nxt()))
                tl.swings.append(s.t_start)
            # the star itself drips too, aimed at where the soul is
            x = .15 * g.rain_x(n.key) + .85 * aim_at(n.t - .75, g)[0]
            tl.attacks.append(A.RainDrop(x, n.t, g.iy0, g.iy1, seed=nxt(), big=n.brk))
            continue
        if n.kind == "slide":
            for s in n.slides:
                paths = [np.array([g.ring(x, y) for x, y in p]) for p in slide_paths(s.start_key, s.segments)]
                long = s.duration > .5
                tl.attacks.append(A.BrushStroke(paths, s.t_star, s.t_start, s.t_end,
                                                width=12, trail=.55 if long else .26,
                                                seed=nxt(), brk=s.brk))
                tl.swings.append(s.t_start)
                if s.brk:
                    end = paths[0][-1]
                    tl.attacks.append(A.SplashRing(end, s.t_end, n=8, speed=130, seed=nxt(),
                                                   tele=.2))
                    tl.flashes.append((s.t_start, 20))
            # the star itself is also a tap in maimai; we leave it as the warning only
            continue
        if n.kind == "hold":
            add_hold(tl, n, sec, g, nxt)
            continue
        i = bisect.bisect_left(times, round(n.t, 4))
        gap_prev = n.t - times[i - 1] if i > 0 else 9
        gap_next = times[i + 1] - n.t if i + 1 < len(times) else 9
        n.isolated = min(gap_prev, gap_next) > .2
        m_start = tm(int((n.t - OFFSET) / MEAS + 1e-6))
        n.index_in_measure = sum(1 for tt in times if m_start - 1e-3 <= tt < n.t - 1e-3)
        add_tap(tl, n, sec, g, nxt)
    add_specials(tl, nxt, notes)
    add_battle(tl, nxt, notes)
    tl.finalize()
    return tl


def add_hold(tl, n: Note, sec, g: Geo, nxt):
    style = sec["hold"]
    if style == "none":
        return
    if style == "hbeam":
        frac = {1: .2, 8: .2, 2: .4, 7: .4, 3: .62, 6: .62, 4: .82, 5: .82}[n.key]
        y = g.iy0 + g.ih * frac
        right = n.key in (1, 2, 3, 4)
        ox = g.ix1 + 40 if right else g.ix0 - 40
        tl.attacks.append(A.Beam((ox, y), (-1 if right else 1, 0), n.t, n.t + n.dur,
                                 width=22, seed=nxt(), brk=n.brk, length=g.iw + 80 - 14))
        return
    ux, uy = key_vec(n.key)
    p, _ = g.edge(n.key)
    O = np.array((p[0] + ux * 40, p[1] + uy * 40))
    if O[1] > g.iy1 - 18:
        # keep ink wells out of the HUD: lower keys open beside the arena instead
        side = 1 if ux > 0 else -1
        O = np.array((g.cx + side * (g.iw / 2 + 40), g.iy1 - 22))
    target = aim_at(n.t - .45, g)
    d = target - O
    d /= np.linalg.norm(d)
    length = ray_exit(g, O, d) + 26
    if d[1] > 1e-6:
        length = min(length, (BOX_BOTTOM + 2 - O[1]) / d[1])
    tl.attacks.append(A.Beam(O, d, n.t, n.t + n.dur, width=24, seed=nxt(), brk=n.brk,
                             length=length))


def add_tap(tl, n: Note, sec, g: Geo, nxt):
    style = sec["tap"]
    if style == "none":
        return
    if n.brk:
        tl.attacks.append(A.SplashRing(g.key_point(n.key, .55), n.t, n=9 if n.group_size == 1 else 7,
                                       speed=140, seed=nxt(), phase=key_angle(n.key)))
        tl.flashes.append((n.t, 26))
        tl.swings.append(n.t)
        return
    if style == "spike":
        p, nn = g.edge(n.key)
        span = g.iw if nn[1] == 0 else g.ih
        L = (.62 if getattr(n, "isolated", False) else .36) * span
        tl.attacks.append(A.Spike(n.t, p, nn, L, seed=nxt(), big=getattr(n, "isolated", False)))
    elif style == "flick":
        if n.t >= TRIP_T[3] - 1e-3:          # from the 2nd group on the ink just rises
            return
        x = g.rain_x(n.key)
        tl.attacks.append(A.InkFlick((x, g.iy0 + 4), 1 if x < g.cx else -1, n.t, seed=nxt()))
        idx = getattr(n, "index_in_measure", 0)
        if idx % 3 == 0:                     # every triplet group also drops a heavy drip
            tl.attacks.append(A.RainDrop(g.rain_x((n.key + 3) % 8 + 1), n.t, g.iy0, g.iy1,
                                         seed=nxt(), big=True))
    elif style == "rain":
        x = g.rain_x(n.key)
        if n.group_size == 1:                # single drops lean toward where the soul is
            x = .15 * x + .85 * aim_at(n.t - .75, g)[0]
        tl.attacks.append(A.RainDrop(x, n.t, g.iy0, g.iy1, seed=nxt(), big=n.group_size > 1))
    elif style in ("drip", "slowdrip"):
        slow = style == "slowdrip"
        fall = .62 if slow else .42
        x = .55 * g.drip_x(n.key) + .45 * aim_at(n.t - fall - .3, g)[0]
        tl.attacks.append(A.Drip(x, n.t, g.iy0, g.iy1, fall=fall, seed=nxt(), big=slow))
    elif style == "dart":
        p, nn = g.edge(n.key)
        aim = aim_at(n.t - .25, g)
        v = aim - np.array(p)
        v = v / np.linalg.norm(v) * 210
        tl.attacks.append(A.Dart(np.array(p) + np.array(nn) * 6, v, n.t, seed=nxt()))
    elif style == "bone":
        add_bone(tl, n, g)


def add_bone(tl, n: Note, g: Geo):
    """Classic sliding bones with a gap. Single taps hang from the ceiling (upper keys)
    or rise from the floor (lower keys); a pair leaves one gap between them."""
    m = int((n.t - OFFSET) / MEAS + 1e-6)
    from_right = (m // 2) % 2 == 0
    speed = 230.0
    vx = -speed if from_right else speed
    x_ref = g.cx
    t_ref = n.t
    span = g.iw / 2 + 20
    t_from, t_to = t_ref - span / speed, t_ref + span / speed
    xr = x_ref
    top, bot = g.iy0, g.iy1
    ih = g.ih
    frac = {1: .30, 8: .30, 2: .40, 7: .40, 3: .40, 6: .40, 4: .30, 5: .30}[n.key]
    if n.key in (1, 2, 7, 8):
        tl.attacks.append(A.Bone(xr, vx, t_ref, top - 6, top + ih * frac, t_from, t_to))
    else:
        tl.attacks.append(A.Bone(xr, vx, t_ref, bot - ih * frac, bot + 6, t_from, t_to))


def add_specials(tl: Timeline, nxt, notes):
    """Hand-placed moments that the chart alone does not express."""
    # --- opening: summon the brush, paint two bars above the arena, bones on the 16ths
    m6 = [n for n in notes if tm(6) - 1e-3 <= n.t < tm(8) - 1e-3]
    hits6 = sorted({round(n.t, 4) for n in m6})          # 1 | 8 1 2/7 3/6 4/5
    six = [(n.t, n.key) for n in notes if tm(8.5) - 1e-3 <= n.t < tm(9) - 1e-3]
    g = Geo((140, 140))
    bars = [(g.x - 14, g.x + g.w + 14, g.y - 15), (g.x - 14, g.x + g.w + 14, g.y - 6)]
    # both pens paint at once: start on "8", finish together on the "2/7" pair; dab on "4/5"
    beats = [hits6[1], hits6[3], hits6[1], hits6[3], hits6[5]]
    tl.attacks.append(A.TopBones((g.ix0, g.iy0, g.ix1, g.iy1), bars, hits6[0], beats,
                                 six, tm(9), seed=nxt()))
    tl.extra_impacts.append((hits6[5], 1.5))
    for t, _ in six:
        tl.extra_impacts.append((t, 3.0))
    tl.extra_impacts.append((tm(9), 7.0))
    # --- the fallen ink pools on the floor and rises with every drop; on the
    #     last triplet it surges to the brim and the arena becomes a canvas
    g = Geo((360, 150))
    drops = [a for a in tl.attacks if isinstance(a, A.RainDrop) and tm(13) - .3 <= a.tl < tm(16)]
    cap = .42 * g.ih
    level = A.PoolLevel([d.tl for d in drops], cap / max(1, len(drops)), cap,
                        1e9, 1.0, g.ih + 4, cx=g.cx, half=g.iw / 2)
    level.floor_y = g.iy1
    # M15: every triplet note pours more ink in; the 6th fills the arena to the brim
    for k, frac in enumerate((.50, .57, .64, .71, .77)):
        level.rise.append((TRIP_T[k], frac * g.ih))
    level.rise.append((TRIP_T[5], g.ih + 6))
    level.freeze_t, level.engulf_t = TRIP_T[5], ENGULF
    # M14: the pool starts to slosh from side to side on every beat, climbing the
    # side where the soul is first; from M14.5 the swings are bigger and the
    # spray they throw off the wall is live
    side = 1 if aim_at(tm(14) - .3, g)[0] >= g.cx else -1
    for i, m in enumerate((14, 14.25, 14.5, 14.75)):
        amp = 16 if m < 14.5 else 24
        level.tilts.append((tm(m), side * amp * (1 if i % 2 == 0 else -1)))
    level.tilts.append((tm(15), 0.0))
    level.spray_hazard = lambda tk: tk >= tm(14.5) - 1e-3
    for d in drops:                          # each drop now splashes on the pool surface
        d.yb = level.surface(d.x, d.tl)
        d.pooled = d.yb < g.iy1 - 2
    for a in tl.attacks:
        if isinstance(a, A.FloorWave) and tm(13) <= a.ts < tm(16):
            a.surf = level.surface
        if isinstance(a, A.InkFlick) and tm(15) - 1e-3 <= a.th < tm(16):
            a.surf = lambda x, t: level.surface(x, t, hazard=False)
    tl.attacks.append(A.InkPool(level, tm(12.75), CANVAS_SET, seed=nxt()))
    global POOL
    POOL = level
    # --- the stretched arena's top edge soaks with ink (M13-M16)
    tl.attacks.append(A.SoakedEdge(tm(12.75), ENGULF - .1, seed=nxt()))
    tl.flashes.append((CANVAS_SET, 26))      # the canvas is set
    # --- M13.5: ink threads drop from the soaked edge on the off-beats, hang
    #     for a beat, then harden into bones and fall into the pool
    for k, m in enumerate((13.625, 13.875, 14.125, 14.375, 14.625, 14.875)):
        td = tm(m)
        x = float(np.clip(aim_at(td - .3, g)[0], g.ix0 + 12, g.ix1 - 12))
        tl.attacks.append(A.InkThread(x, td, td + BEAT, g.iy0 + 2, 54 + 8 * (k % 3), level.surface,
                                      seed=nxt()))
    # --- final note: one slow drop falls in the silence
    g = Geo((140, 140))
    t = tm(121)
    tl.attacks.append(A.Drip(g.cx + 26, t, g.iy0, g.iy1, fall=.9, seed=nxt(), big=True))


# --------------------------------------------------------------------- M17-M24
def add_battle(tl: Timeline, nxt, notes):
    """Phrase by phrase, every tap comes from OUTSIDE the canvas and turns to ink
    once it crosses into it (add_external). The first phrase after the title
    comes from one direction only (running), the second from another (climbing)
    and adds the binding spiral; then pincers, fans and the lodged spokes.
    Painted ink slashes are a rare accent (the two slides of M24.5)."""
    g = Geo((200, 150))
    C0 = np.array([g.cx, g.cy])
    RX, RY = g.iw / 2, g.ih / 2

    def chord(k, off, through=None, flip=False, L=260):
        ux, uy = key_vec(k)
        if through is None:
            ctr = C0 + np.array([ux * off * RX, uy * off * RY])
        else:
            ctr = np.array(through, float)
        v = np.array([-uy, ux]) * (-1 if flip else 1)
        return [ctr - v * L / 2, ctr + v * L / 2]

    def arc(k, r, span=100, flip=False):
        a = key_angle(k)
        a0, a1 = a - math.radians(span / 2), a + math.radians(span / 2)
        if flip:
            a0, a1 = a1, a0
        s = np.linspace(a0, a1, 40)
        return np.stack([C0[0] + np.sin(s) * r * RX, C0[1] - np.cos(s) * r * RY], 1)

    def add(path, t, **kw):
        tl.attacks.append(A.InkStroke(path, t, seed=nxt(), **kw))

    seg = [n for n in notes if tm(17) - 1e-3 <= n.t < tm(25) - 1e-3]
    times = sorted({round(n.t, 4) for n in seg})
    flip = False
    for t in times:
        group = [n for n in seg if abs(n.t - t) < 1e-3]
        m = (t - OFFSET) / MEAS
        if abs(t - tm(17)) < 1e-3:
            continue                  # the entry blots right after the title card: no warning possible
        pair = len(group) > 1
        flip = not flip
        for n in group:
            k = n.key
            if n.kind == "slide":
                continue                  # no painted slashes here any more (M20: the spiral,
                                          # M24.5: the cube; the break slide paints the ensō)
                for s in n.slides:
                    for p in slide_paths(s.start_key, s.segments):
                        path = [g.ring(x * .85, y * .85) for x, y in p]
                        tl.attacks.append(A.InkStroke(path, s.t_start, width=11,
                                                      lead=max(.42, s.t_start - s.t_star),
                                                      sweep=max(.06, s.duration), linger=.12, seed=nxt()))
                continue
            # every tap is an attack from outside the canvas (add_external); ink
            # slashes appear only as a rare accent: the two slides of M24.5
    add_external(tl, nxt, seg, g)
    add_slashes(tl, nxt, seg, g)
    # the break (1b>8) shatters the spokes; its long slide paints the ensō around the canvas
    brk = [n for n in seg if n.brk and n.kind == "slide"]
    for n in brk:
        s = n.slides[0]
        # the cube bursts into a ring of ink that does not dry: it comes alive,
        # rings the soul (which the blast has thrown clear) and closes in on it
        # until the one press of FIGHT cuts it all
        hunt_target = (lambda t: AIM(t) if AIM is not None and AIM(t) is not None else FIGHT_SOUL)
        tl.attacks.append(A.HuntingInk((g.cx, g.cy), n.t, hunt_target, PRESS_T, tm(28.5), lunges=CLOSE_IN,
                                       seed=nxt()))
        tl.extra_impacts.append((n.t, 8.0))
        tl.flashes.append((n.t, 40))
    add_rest(tl, nxt)
    # entering the canvas: the break pair 2b/7b lands as two great pale blots
    # (no warning is possible right after the title card, so they do not hurt)
    first = [n for n in seg if abs(n.t - tm(17)) < 1e-3]
    for n in first:
        tl.attacks.append(A.SplashRing(np.array(g.ring(*np.array(key_vec(n.key)) * .6)), n.t, n=9, speed=150,
                                       seed=nxt(), tele=.0, big=True, harmless=True))
    tl.flashes.append((tm(17), 30))


# --------------------------------------------------------------------- jump-slams
def add_slams(tl: Timeline, nxt, notes):
    """Ink Sans' jump-slams (actions.py). NOT scheduled at the moment (he stays
    still, as in the reference); kept for a later section that asks for one."""
    actions.SLAMS.clear()
    trip = sorted({round(n.t, 4) for n in notes if tm(15) - 1e-3 <= n.t < tm(16) - 1e-3})
    gr, gb = Geo((360, 150)), Geo((200, 150))

    def aimed(t, g):                     # land so that the brush comes down on the soul
        ax = float(np.clip(aim_at(t, g)[0], g.ix0 + 20, g.ix1 - 20))
        return ax - actions.TIP_DX

    specs = [
        # "...拉进地狱里吧。" - the brush is painted back, he leaps on the 4 and lands
        # on the break star: the brush smashes the arena wide open
        (actions.Slam(tm(12.375), tm(12.375) + .14, tm(12.75), peak=150), gr, POOL.surface, 9.0),
        # the triplet run ends with him leaping out of the frame; he crashes back
        # down as the title card ends, onto the canvas
        (actions.Slam(trip[9], trip[10], tm(17), x_land=aimed(tm(17) - .6, gb), peak=430, apex=19.5),
         gb, None, 9.0),
        # M21: the darkened phrase opens with a slam aimed at the soul
        (actions.Slam(tm(20.625), tm(20.625) + .11, tm(21), x_land=aimed(tm(21) - .5, gb), peak=150),
         gb, None, 8.0),
        # M24.75: he comes down on the centre of the eight spokes and shatters them
        (actions.Slam(tm(24.375), tm(24.5), tm(24.75), x_land=gb.cx - actions.TIP_DX, peak=140, hold=.6),
         gb, None, 8.0),
    ]
    for slam, g, surf, amp in specs:
        actions.SLAMS.append(slam)
        x = slam.tip_x
        tl.attacks.append(A.InkSlam(x, slam.land, g.iy0, g.iy1, seed=nxt()))
        for wall in (g.ix0 + 8, g.ix1 - 8):
            if abs(wall - x) > 30:
                tl.attacks.append(A.FloorWave(x, wall, slam.land + .02, slam.land + .40, g.iy1, seed=nxt(),
                                              warn=.01, surf=surf, height=20))
        tl.extra_impacts.append((slam.land, amp))
        tl.flashes.append((slam.land, 34))


# orb positions around the 200x150 canvas, by maimai key (kept clear of the HUD)
ORB_POS = {1: (452, 214), 2: (474, 288), 3: (474, 350), 4: (448, 374),
           5: (192, 374), 6: (166, 350), 7: (166, 288), 8: (188, 214)}


ORB_ABOVE = {1: (432, 206), 2: (472, 226), 3: (472, 226), 4: (432, 206),
             5: (208, 206), 6: (168, 226), 7: (168, 226), 8: (208, 206)}


def _wall_rows(g, n, aim_y, side_sign):
    """Rows for a wall of n bones: a few right on the soul, or a full wall with one
    gap a little away from it (alternating up / down like steps)."""
    rows = [g.iy0 + 10 + 15 * i for i in range(9)]
    if n >= 7:
        gy = min(max(aim_y + 32 * side_sign, rows[1]), rows[-2])
        gi = min(range(len(rows)), key=lambda i: abs(rows[i] - gy))
        use = [r for i, r in enumerate(rows) if i != gi][:n]
    else:
        use = sorted(rows, key=lambda r: abs(r - aim_y))[:n]
    return sorted(use)


# one ink slash in every measure of the battle (plus the two slides of M24.5)
SLASH_AT = [17.75, 18.75, 19.75, 20.125]        # none once the tunnel starts


def add_external(tl: Timeline, nxt, seg, g: "Geo"):
    """Attacks from outside the canvas (attacks.PenVolleys / BoneOrbs /
    HelixBind / SquareTunnel / CanvasCube), and one ink slash per measure."""
    taps = [n for n in seg if n.kind != "slide"]
    slash_t = [tm(m) for m in SLASH_AT]
    is_slash = lambda t: any(abs(t - s_) < 1e-3 for s_ in slash_t)
    # ---- M17-18 RUN: everything comes from the right, fast. On every strong beat
    #      a bar writes itself and fires a wall (the gap steps up and down like
    #      ledges); every other tap throws a fast bone at the soul's height
    inner = (g.ix0, g.iy0, g.ix1, g.iy1)
    strong = [tm(17.375), tm(18), tm(18.375), tm(19)]
    bars = []
    for k, t in enumerate(strong):
        use = _wall_rows(g, 8, aim_at(t - .4, g)[1], 1 if k % 2 else -1)
        bars.append(dict(side="R", paint=t - .3, fire=t, pops=[(t - .17 + .008 * i, y) for i, y in enumerate(use)]))
    tl.attacks.append(A.PenVolleys(g, bars, tm(17), tm(19), speed=620.0, seed=nxt()))
    run = []
    for n in taps:
        # (no single bones while a wall is sweeping through - they would plug its gap)
        if tm(17) + 1e-3 < n.t < tm(19) - 1e-3 and not any(-1e-3 < n.t - s_ < .36 for s_ in strong) \
                and not is_slash(n.t):
            y = float(np.clip(aim_at(n.t - .32, g)[1], g.iy0 + 10, g.iy1 - 10))
            run.append((n.t, (g.x + g.w + 34, y), (-1.0, 0.0)))
    tl.attacks.append(A.BoneShots(inner, run, speed=640.0, seed=nxt()))
    # ---- M19-20 CLIMB: everything comes from above - orbs over the arena fire
    #      down at the soul (the M19 downbeat belongs to the last wall)
    shots = []
    for n in taps:
        if tm(19) + 1e-3 < n.t < tm(20.54) - 1e-3 and not is_slash(n.t):
            shots.append((n.t, ORB_ABOVE[n.key], aim_at(n.t - .45, g)))
    # (not the last one: the spiral could not catch it, and a bone still flying
    #  after everything has been pulled tight and burst looks wrong)
    shots = sorted(shots, key=lambda s_: s_[0])[:-1]
    climb = A.BoneOrbs(g, shots, seed=nxt())
    tl.attacks.append(climb)
    # ...and the binding spiral: drawn up the canvas ring by ring on the eighths
    #    around the soul's column (a dense spring: a ring every sixteenth), it pulls
    #    tight on the three twelfths of M20.5
    cx = float(np.clip(aim_at(tm(19.875) - .3, g)[0], g.ix0 + 50, g.ix1 - 50))
    keys = [(tm(19.875 + k / 16), float(k)) for k in range(11)]      # a ring on every sixteenth
    tight = [(tm(20 + 13 / 24), .62), (tm(20 + 14 / 24), .32), (tm(20.625), .12)]
    helix = A.HelixBind(cx, g.iy1 - 10, g.iy0 + 10, 46, keys, tight, tm(20.75), seed=nxt())
    helix.bind_prey(climb.bones, (g.ix0, g.iy0, g.ix1, g.iy1))   # it ties up the bones inside, then bursts
    t_hit = FORCED_HITS[0][0]
    helix.hit = (t_hit, aim_at(t_hit, g))                       # ...and one shard hits the soul
    tl.attacks.append(helix)
    # ---- M21-22 TUNNEL (the camera zooms right into the canvas): square frames
    #      rush towards us out of a vanishing point that drifts round the middle,
    #      so the centre is never safe; on each strong beat the canvas, cut 3x3 by
    #      blades just before, drops away into the dark except one white cell (留白)
    #      that walks round the outer ring;
    #      the frames' cut-out wedges lead the soul from one white cell to the next
    C0 = np.array([g.cx, g.cy])
    ring = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    cw, ch = g.iw / 3, g.ih / 3

    def cell_centre(ij):
        return np.array([g.ix0 + (ij[0] + .5) * cw, g.iy0 + (ij[1] + .5) * ch])

    strong21 = [tm(21), tm(21.375), tm(22), tm(22.375)]
    a0 = aim_at(strong21[0] - .5, g)
    start = min(range(8), key=lambda i: np.linalg.norm(cell_centre(ring[i]) - a0))
    safes = [ring[(start + 1 + k) % 8] for k in range(4)]
    sixteenth = (tm(1) - tm(0)) / 16
    for t, sc in zip(strong21, safes):     # 切纸塌陷: cut on the sixteenths before, the pieces drop on the beat
        tl.attacks.append(A.GridCollapse((g.ix0, g.iy0, g.ix1, g.iy1), t, sc, sixteenth, (g.cx, g.cy),
                                         seed=nxt()))
    tunnel = sorted({round(n.t, 4) for n in taps if tm(21) - 1e-3 <= n.t < tm(22.75) - 1e-3
                     and not any(abs(n.t - s_) < 1e-3 for s_ in strong21)})
    bounds = strong21 + [tm(22.75)]
    frames = []
    for k in range(4):
        here = [t for t in tunnel if bounds[k] < t < bounds[k + 1]]
        p_from = cell_centre(safes[k])
        p_to = cell_centre(safes[k + 1]) if k + 1 < 4 else p_from
        for i, t in enumerate(here):
            u = (i + 1) / (len(here) + 1)
            target = p_from + (p_to - p_from) * u
            w_ = math.tau * (t - tm(21)) / MEAS
            vp = C0 + np.array([math.cos(w_) * 26, math.sin(w_) * 20])
            d = target - vp
            frames.append((t, math.degrees(math.atan2(d[1], d[0])), 18 * (1 if k % 2 else -1), False, tuple(vp)))
    tl.attacks.append(A.SquareTunnel(C0, frames, t_show=tm(20.75) + .1, seed=nxt()))
    # ---- M22.75-M24.75 THE CUBE: the camera pulls back out of the canvas - the
    #      soul was inside a rectangle all along. On M23 it grows depth and we see
    #      box and soul in 3D from one fixed angle (no turning yet): on every twelfth
    #      after the first a bone flies through it, three at a time from one side
    #      (the chart's key triplets), aimed at the soul, which dodges on the front face. Only on
    #      the M24 eighths does it start to turn - and every turn a slash cuts it;
    #      the two slides cut an X across its front, then it bursts.
    Y0, P0 = 28.0, -18.0                                    # the fixed 3D view
    steps = [(tm(22.75), 0.0, 0.0), (tm(23), Y0, P0)]
    side_of_key = {8: "-x", 4: "+y", 6: "-z", 2: "+x"}     # left / bottom / front / right
    bones = []
    # (the first twelfth is the box growing depth itself; a bone on each one after)
    for n in sorted((n for n in taps if tm(23) + 1e-3 <= n.t < tm(24) - 1e-3), key=lambda n: n.t):
        bones.append((n.t, side_of_key.get(n.key, "-x"), aim_at(n.t - .4, g)))
    slashes = []
    for mm, cut, yw, pt in ((24, (("+x", 0), ("-x", 1)), 90, 22), (24.125, (("+y", 0), ("-y", 1)), 180, -26),
                            (24.25, (("-z", 0), ("+z", 1)), 270, 14), (24.375, (("+x", 1), ("-x", 0)), 450, -12)):
        steps.append((tm(mm), Y0 + yw, float(pt)))
        slashes += [(tm(mm), f, dg) for f, dg in cut]
    steps.append((tm(24.5), Y0 + 720.0, P0))                # back to the 3D view for the X
    slashes += [(tm(24.5), "-z", 0), (tm(24.5), "-z", 1)]
    cube = A.CanvasCube((g.ix0, g.iy0, g.ix1, g.iy1), DEEP_T, tm(23), steps, bones, tm(24.75),
                        seed=nxt(), slashes=slashes)
    tl.attacks.append(cube)
    global CUBE
    CUBE = cube


def add_slashes(tl: Timeline, nxt, seg, g: "Geo"):
    """One ink slash per measure from M17 to M20, cut through where the soul is
    (a clear wash of ink first); none once the tunnel starts."""
    C0 = np.array([g.cx, g.cy])
    RX, RY = g.iw / 2, g.ih / 2
    for i, m in enumerate(SLASH_AT):
        t = tm(m)
        group = [n for n in seg if abs(n.t - t) < 1e-3 and n.kind != "slide"]
        if not group:
            continue
        k = group[0].key
        ux, uy = key_vec(k)
        aim = aim_at(t - .5, g)
        p = .6 * aim + .4 * (C0 + np.array([ux * .2 * RX, uy * .2 * RY]))
        v = np.array([-uy, ux]) * (1 if i % 2 else -1)
        tl.attacks.append(A.InkStroke([p - v * 140, p + v * 140], t, width=12, seed=nxt()))


def add_rest(tl: Timeline, nxt):
    """M25-M28.5: no turn is given. The soul walks for FIGHT the whole time
    (SOUL_SCRIPT) and presses it once, the moment it touches its edge (PRESSES): the
    button flashes (scene.compose) and one slash cuts the whole ring of living
    ink that had closed in on it (attacks.HuntingInk)."""
    actions.BRUSH_AWAY[:] = []
    actions.PARRIES[:] = []
    for t in PRESSES:
        tl.extra_impacts.append((t, 6.0))
