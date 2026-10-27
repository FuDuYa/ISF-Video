"""Section-by-section camera moves and global effects (applied after composition).

Edit this file to direct a section: camera keys, hit kicks and screen effects.
Times are built from chart measures (choreo.tm) so they stay on the beat.
"""

from __future__ import annotations

import math

import numpy as np

import choreo as C
import fx
from chart import parse_chart
from common import ROOT, smooth, clamp, out_cubic

tm = C.tm
_notes, _ = parse_chart(ROOT / "maidata.txt", C.OFFSET)


def _times(m0, m1):
    return sorted({round(n.t, 4) for n in _notes if tm(m0) - 1e-3 <= n.t < tm(m1) - 1e-3})


# ------------------------------------------------------------------ opening (7 - 10.7 s)
SUMMON, *CORNERS = _times(6, 8)[:6]          # 1 | 8 1 2/7 3/6 4/5
SIXTEENTHS = _times(8.5, 9)                   # 4 5 3 6 2 7 1 8
DOWNBEAT = tm(9)
BOX_C = (320.0, 318.0)
CHARGE = CORNERS[-1]

# M22.75: the deepest the camera goes into the canvas, on the soul - kept far
# enough from the rectangle's sides that none of them can be in the frame yet
DEEP = 3.6          # just far enough in that the trapping rectangle's edges are out of the frame
DEEP_T = C.DEEP_T


def _deep_focus(t):
    g = C.Geo((200, 150))
    sx, sy = SOUL(t) if SOUL is not None else (g.cx, g.cy)
    hx, hy = 320 / DEEP + 10, 240 / DEEP + 10
    return (min(max(sx, g.ix0 + hx), g.ix1 - hx), min(max(sy, g.iy0 + hy), g.iy1 - hy))


def _tunnel_push(z0=1.08, n=14):
    """Camera keys from M21 to DEEP_T: a steady push in, even in log zoom (no
    standing still and then lunging in), easing in at the start and settling at
    the end."""
    t0, t1 = tm(21), DEEP_T
    out = []
    for k in range(n + 1):
        u = k / n
        f = .55 * u + .45 * (u * u * (3 - 2 * u))
        out.append(dict(t=t0 + (t1 - t0) * u, focus=(320, 313), zoom=z0 * (DEEP / z0) ** f,
                        pitch=0, yaw=0, roll=0, ease="linear" if k else "smooth"))
    return out


def _sink_focus(t):
    """The descent: the camera leans a quarter of the way after the sliding arena."""
    dx, dy, _, _ = C.arena_state(t)
    return (320 + .25 * dx, 240 + .2 * (dy - C.CY))


KEYS = [
    dict(t=7.0, focus=(320, 240), zoom=1.0, pitch=0, yaw=0, roll=0),
    # the brush paints with the camera held still
    dict(t=CHARGE, focus=(320, 240), zoom=1.0, pitch=0, yaw=0, roll=0),
    # charge: slow push in, tilting back, a creeping roll
    dict(t=SIXTEENTHS[0] - .02, focus=BOX_C, zoom=1.30, pitch=16, yaw=0, roll=-4, ease="in"),
    # sixteenths: the card falls flat again while it is being hit
    dict(t=SIXTEENTHS[-1] + .04, focus=BOX_C, zoom=1.22, pitch=0, yaw=0, roll=0, ease="smooth"),
    # downbeat: snap back to the plain battle view
    dict(t=DOWNBEAT + .14, focus=(320, 240), zoom=1.0, pitch=0, yaw=0, roll=0, ease="out"),
    # resting passage (M9-M12): a slow pseudo-3D breathing drift with the phrases
    dict(t=tm(10), focus=(320, 236), zoom=1.035, pitch=-3, yaw=4, roll=.8),
    dict(t=tm(11), focus=(320, 244), zoom=1.015, pitch=3, yaw=-4, roll=-.8),
    dict(t=tm(12), focus=(320, 236), zoom=1.04, pitch=-2, yaw=3, roll=.6),
    dict(t=tm(12.75), focus=(320, 240), zoom=1.06, pitch=0, yaw=0, roll=0),
    dict(t=tm(13), focus=(320, 240), zoom=1.0, pitch=0, yaw=0, roll=0, ease="out"),
    # ink rain (M13-M16): we lean over the stretched canvas, looking down onto it
    dict(t=tm(14), focus=(320, 296), zoom=1.05, pitch=9, yaw=-4, roll=0),
    dict(t=tm(15), focus=(320, 302), zoom=1.07, pitch=10, yaw=4, roll=0),
    # triplet run: push in and slowly roll while the ink rises
    dict(t=C.ENGULF, focus=(320, 306), zoom=1.12, pitch=6, yaw=0, roll=3.0),
    # the arena soaks into paper: the sheet turns to face us squarely
    dict(t=C.CANVAS_SET, focus=(320, 314), zoom=1.24, pitch=0, yaw=0, roll=0, ease="out"),
    # the two floor waves race apart: the roll swings back
    dict(t=tm(16.5), focus=(320, 312), zoom=1.08, pitch=3, yaw=0, roll=-2, ease="out"),
    dict(t=tm(17), focus=(320, 240), zoom=1.0, pitch=0, yaw=0, roll=0, ease="smooth"),
    # M17-M24 battle on the canvas
    dict(t=tm(18), focus=(320, 286), zoom=1.05, pitch=6, yaw=-5, roll=-1.5),
    dict(t=tm(19), focus=(320, 280), zoom=1.02, pitch=0, yaw=0, roll=0),
    dict(t=tm(19.5), focus=(320, 286), zoom=1.05, pitch=-5, yaw=0, roll=0),     # climbing: looking up
    dict(t=tm(19.875), focus=(320, 300), zoom=1.06, pitch=-3, yaw=0, roll=0),
    dict(t=tm(20.5), focus=(320, 312), zoom=1.14, pitch=0, yaw=0, roll=0, ease="in"),  # the spiral rises
    dict(t=tm(20.75), focus=(320, 306), zoom=1.08, pitch=0, yaw=0, roll=0, ease="out"),
    # tunnel: the paper spreads over the screen while we push right into the
    # canvas - all the way inside the rectangle, whose edges we cannot see. The
    # push starts the moment the 3x3 begins (M21) and goes on steadily (even in
    # log zoom, so it reads as one steady move, easing in and settling) until the
    # rectangle's edges have just left the frame (DEEP_T); a moment's pause there
    *_tunnel_push(),
    dict(t=tm(22.75), focus=(320, 313), zoom=DEEP, pitch=0, yaw=0, roll=0, ease="linear"),
    # ...then pull back: its edges slide in from outside the frame - the soul
    # was trapped in a rectangle all along
    dict(t=tm(23), focus=(320, 313), zoom=1.2, pitch=0, yaw=0, roll=0, ease="out"),
    dict(t=tm(24), focus=(320, 313), zoom=1.24, pitch=0, yaw=0, roll=0),
    dict(t=tm(24.5), focus=(320, 313), zoom=1.3, pitch=0, yaw=0, roll=0, ease="in"),   # it comes back square
    dict(t=tm(24.75) + .3, focus=(320, 300), zoom=1.0, pitch=0, yaw=0, roll=0, ease="out"),  # burst
    # the soul heads across the sheet for FIGHT with the living ink closing round
    # it: follow it down to the button's right edge, pushing in slowly - all the
    # way in only on the cut, then pull back (same framing on the soul as when it
    # walked onto the button's left end)
    dict(t=C.WALK_T, focus=(290, 320), zoom=1.04, pitch=0, yaw=0, roll=0),
    dict(t=tm(26.5), focus=(245, 335), zoom=1.16, pitch=0, yaw=0, roll=0),
    dict(t=C.PRESS_T - .1, focus=(204, 404), zoom=1.36, pitch=0, yaw=0, roll=0),
    dict(t=C.PRESS_T, focus=(170, 430), zoom=1.7, pitch=0, yaw=0, roll=0, ease="in"),
    dict(t=C.PRESS_T + .2, focus=(168, 432), zoom=1.74, pitch=0, yaw=0, roll=0, ease="linear"),
    dict(t=tm(28.5), focus=(320, 240), zoom=1.0, pitch=0, yaw=0, roll=0, ease="smooth"),   # (no jolt as it starts)
    # the descent: a slight lean down as we sink, then after the arena
    dict(t=C.RISE[0], focus=(320, 240), zoom=1.0, pitch=0, yaw=0, roll=0),
    dict(t=C.ARRIVE - .05, focus=(320, 250), zoom=1.03, pitch=-5, yaw=0, roll=0, ease="smooth"),
    dict(t=C.ARRIVE + .3, focus=lambda t: _sink_focus(t), zoom=1.04, pitch=0, yaw=0, roll=0, ease="out"),
    dict(t=C.NEW_END - .2, focus=lambda t: _sink_focus(t), zoom=1.04, pitch=0, yaw=0, roll=0, ease="linear"),
]

KICKS = [dict(t=CORNERS[4], zoom=.02, decay=14)]
for i, t in enumerate(SIXTEENTHS):
    s = 1 if i % 2 == 0 else -1
    KICKS.append(dict(t=t, zoom=.07, roll=4 * s, yaw=5 * s, decay=20))
KICKS.append(dict(t=DOWNBEAT, zoom=-.10, roll=3, pitch=-6, decay=9))
# paired drops hitting the floor (M13-M16)
for n in _notes:
    if tm(13) - 1e-3 <= n.t < tm(17) - 1e-3 and n.group_size > 1 and n.kind != "slide":
        KICKS.append(dict(t=n.t, zoom=.018, decay=16))

# M17-18 run: a quick step on every eighth; M20.5 the spiral pulls tight in three
# jolts and bursts; M21-22 each tunnel gate thumps; the faces of the cube are hit
for i in range(16):
    KICKS.append(dict(t=tm(17 + i / 8), zoom=.012, pitch=1.8 * (1 if i % 2 else -1), roll=.6 * (1 if i % 2 else -1),
                      decay=18))
for k in (13, 14, 15):
    KICKS.append(dict(t=tm(20 + k / 24), zoom=.03, decay=14))
KICKS.append(dict(t=tm(20.75), zoom=-.05, decay=9))
for m in (21, 21.375, 22, 22.375):                 # the cut paper drops away (留白): the floor gives
    KICKS.append(dict(t=tm(m), zoom=.06, pitch=5, decay=10))
for k in range(12):                                  # each fold / slap of the paper box (折纸立方体)
    KICKS.append(dict(t=tm(23 + k / 12), zoom=.02, roll=1.2 * (1 if k % 2 else -1), decay=16))
for k in (1, 2, 3):                                  # the net scored, the top flap creasing up (sixteenths)
    KICKS.append(dict(t=tm(22.75 + k / 16), zoom=.015, roll=.8 * (1 if k % 2 else -1), decay=18))
KICKS.append(dict(t=tm(24), zoom=.06, pitch=4, decay=10))                # the lid shuts
for m in (24.125, 24.25, 24.375):                    # the box tumbling like a die lands each turn: a thump
    KICKS.append(dict(t=tm(m) + .1, zoom=.045, pitch=3, decay=12))
KICKS.append(dict(t=tm(20.75) + .14, zoom=.05, roll=-3, decay=10))     # the shard hits the soul
# battle (M17-M24): pairs kick, the break blows it open
for n in _notes:
    if tm(17) - 1e-3 <= n.t < tm(24) - 1e-3 and n.group_size > 1:
        KICKS.append(dict(t=n.t, zoom=.022, decay=16))
for n in _notes:                                     # the descent: pairs kick, so does every piston stab
    if C.ARRIVE + .05 <= n.t < tm(47) - 1e-3 and n.group_size > 1 and not n.brk:
        KICKS.append(dict(t=n.t, zoom=.02, roll=.8 * (1 if int(n.t * 100) % 2 else -1), decay=16))
KICKS.append(dict(t=tm(24.75), zoom=-.12, roll=-3, decay=7))

INVERT = [(t, 1 / 60) for t in SIXTEENTHS] + [(DOWNBEAT, 2 / 60)]
LIFT = [(t + 1 / 60, 1 / 60, 70) for t in SIXTEENTHS] + [(DOWNBEAT + 2 / 60, 3 / 60, 50)]
GLITCH = [(SIXTEENTHS[3], .06, .8), (SIXTEENTHS[6], .06, 1.0), (DOWNBEAT, .09, 1.2),
          (tm(24.75), .08, 1.0)]
INVERT.append((tm(24.75), 1 / 60))
for t in C.CONFIRMS:                                 # each confirmed bar smashes a drop
    KICKS.append(dict(t=t, zoom=.03, roll=1.0, decay=16))
KICKS.append(dict(t=C.AIM_OPEN, zoom=.04, decay=12))   # FIGHT: the aim meter comes up
for t in C.CURTAIN_CUTS:                             # every piece of the ink over him cut away
    KICKS.append(dict(t=t, zoom=.015, roll=.6 * (1 if int(t * 1000) % 2 else -1), decay=18))
KICKS.append(dict(t=C.REVEAL, zoom=-.05, pitch=-2, decay=8))   # the last of the ink is gone: he is there
for j in range(4):                                   # the 5-4-5-4: the piece is torn out, a side a note
    KICKS.append(dict(t=C.SINK0 + j * C.MEAS / 16, zoom=.02, pitch=-2.5, decay=14))
KICKS.append(dict(t=C.ARRIVE, zoom=.07, pitch=4, decay=9))   # it lands
for t in (tm(38.875), tm(47)):                       # the breaks: the arena snaps into a new shape
    KICKS.append(dict(t=t, zoom=-.08, roll=2.5, decay=8))
    GLITCH.append((t, .06, .9))
for t, hard in C.TRANSFORMS:                         # every slam of the canvas to a new side / shape
    sgn = 1 if int(t * 10) % 2 else -1
    KICKS.append(dict(t=t + .05, zoom=.05 * hard, roll=2.2 * hard * sgn, yaw=3 * hard * sgn, decay=10))
    LIFT.append((t + .05, 1 / 60, int(40 * hard)))   # (a frame where the black lifts: the slam)

# triplet run (M15): every note kicks the camera, every triplet group flashes
TRIPLETS = _times(15, 16)
for i, t in enumerate(TRIPLETS):
    soft = t >= C.ENGULF - 1e-3              # while it soaks into paper: gentler
    KICKS.append(dict(t=t, zoom=.008 if soft else .014, roll=(.8 if soft else 1.6) * (1 if i % 2 else -1),
                      decay=22))
    if i in (0, 3, 6):
        LIFT.append((t, 2 / 60, 46))
    if i in (0, 3):
        GLITCH.append((t, .04, .55))
KICKS.append(dict(t=C.CANVAS_SET, zoom=.03, decay=10))     # the canvas is set

# the arena stretching wide on the break star
KICKS.append(dict(t=tm(12.75), zoom=-.05, roll=1.5, decay=10))


# ------------------------------------------------------------------ close-up cuts
# Hard cuts to a close shot and back (integer zoom keeps the pixels square).
# focus: a point, "soul" (the planned soul position) or "pool" (soul x on the ink surface)
SOUL = None          # set by render.py: t -> (x, y)
# Cuts happen only inside the twelfth-note (triplet) runs.
# M15: while the ink rises, the shot cuts to the title word by word - "Ink!" on
# the 2nd+3rd notes of the first group, "Sans" and "Fight" on the 2nd and 3rd of
# the second; from the 7th note on (arena engulfed, soaking into a canvas)
# the camera stays on the battle.
CUTS = [
    (TRIPLETS[1], TRIPLETS[3], None, 4, -3),            # title: Ink!
    (TRIPLETS[4], TRIPLETS[5], None, 4, 0),             # title: Sans
    (TRIPLETS[5], TRIPLETS[6], None, 4, 3),             # title: Fight
]


def cut_at(t):
    for t0, t1, focus, zoom, roll in CUTS:
        if t0 <= t < t1:
            if focus in ("soul", "pool") and SOUL is not None:
                sx, sy = SOUL(t0)
                if focus == "pool" and C.POOL is not None:
                    sy = C.Geo((360, 150)).iy1 - C.POOL(t, hazard=True)
                focus = (sx, sy)
            elif not isinstance(focus, tuple):
                focus = (320, 318)
            return dict(focus=focus, zoom=float(zoom), pitch=0.0, yaw=0.0, roll=float(roll))
    return None


TITLE_CUTS = 3          # the first cuts (inside the M15 triplets) show the title, word by word


# the rest (M25.75-M28): as the living ink rings the soul and slowly closes in,
# ink closes in over the whole screen too, just as slowly and smoothly - creeping
# from the edges round the soul; blots seep onto the lens and run down; the one
# FIGHT slash drives it all back
CRISIS = (C.CLOSE_IN[0] - .3, C.PRESS_T + .5)
_CR_FAR, _CR_NEAR = 500.0, 66.0                      # clear radius (screen px): from past the screen's corners / at the press


def _crisis_level(t):
    import attacks as A
    return A.close_progress(t, CRISIS[0], C.PRESS_T)


def _crisis(out, t, p):
    after = t - C.PRESS_T
    lv = _crisis_level(min(t, C.PRESS_T))
    if after > 0:                                    # the slash drives it all back (smoothly, not in one jump)
        lv *= 1 - smooth(after / .45)
    if lv <= .001:
        return out
    radius = _CR_FAR - (_CR_FAR - _CR_NEAR) * lv
    sx, sy = fx.project_point(p, *SOUL(t))
    splats = []
    for k, L in enumerate(C.CLOSE_IN[2:], start=2):  # blots on the lens from the third lunge on
        if t < L:
            continue
        rng = np.random.default_rng(900 + k)
        pL = fx.camera_at(L, KEYS, KICKS)
        lx, ly = fx.project_point(pL, *SOUL(L))
        rL = _CR_FAR - (_CR_FAR - _CR_NEAR) * _crisis_level(L)
        for j in range(2):
            ang = rng.uniform(0, math.tau)
            dist = rL + rng.uniform(50, 130)
            age = (t - L) + (3 * after if after > 0 else 0)
            splats.append((lx + math.cos(ang) * dist * 1.2, ly + math.sin(ang) * dist,
                           rng.uniform(14, 30), age, 900 + k * 10 + j))
    gain = smooth((t - CRISIS[0]) / .8) * (1 - smooth((t - (CRISIS[1] - .4)) / .4))
    return fx.crisis_ink(out, t, (sx, sy), radius, 0.0, splats, gain)


_SLICES = None


def _dismantle_slices():
    """解: while the shut paper box tumbles, every piece cut to bits cuts the whole
    screen with it - a hairline across everything, the picture slipping along it;
    on each turn a burst of bigger cuts through the middle."""
    global _SLICES
    if _SLICES is None and C.CUBE is not None and hasattr(C.CUBE, "cuts"):
        rng = np.random.default_rng(2424)
        box = C.CUBE
        end = box.slash_t - .06                      # all healed before the box is back down for the X
        raw = []
        for key, tc in sorted(box.cuts.items(), key=lambda e: e[1]):
            p = (box.C[0] + rng.uniform(-60, 60), box.C[1] + rng.uniform(-45, 45))
            raw.append((tc, p, rng.uniform(0, math.pi), rng.uniform(6, 10) * rng.choice([-1, 1]), .24))
        for tk in box.tumbles:                       # on every turn, a burst of bigger cuts through the middle
            for j in range(4):
                p = (box.C[0] + rng.uniform(-40, 40), box.C[1] + rng.uniform(-40, 40))
                raw.append((tk + .012 * j, p, rng.uniform(0, math.pi), rng.uniform(10, 16) * rng.choice([-1, 1]), .3))
        _SLICES = [(tc, p, a, sh, min(life, end - tc)) for tc, p, a, sh, life in raw if end - tc > .06]
    return _SLICES or []


def apply(frame: np.ndarray, t: float) -> np.ndarray:
    import titlecard
    if titlecard.active(t):
        return titlecard.draw(t)
    for i, (t0, t1, _focus, _zoom, roll) in enumerate(CUTS[:TITLE_CUTS]):
        if t0 <= t < t1:
            return titlecard.fragment(t, i, TITLE_CUTS, t0, t1, roll)
    p = cut_at(t) or fx.camera_at(t, KEYS, KICKS)
    import scene
    lp = scene.LAST_PLANE
    out = fx.warp(frame, p, lp[1] if lp is not None and abs(lp[0] - t) < 1e-6 else None)
    # charge vignette: the edges of the world darken toward the arena
    if CHARGE <= t < DOWNBEAT + .3:
        v = .65 * smooth((t - CHARGE) / (SIXTEENTHS[0] - CHARGE)) * (1 - smooth((t - DOWNBEAT) / .3))
        out = fx.vignette(out, v)
    # sixteenths: speed lines, double image, glitch
    if SIXTEENTHS[0] <= t < DOWNBEAT + .12:
        out = fx.speed_lines(out, t, 1.0 if t < DOWNBEAT else 1.6)
        k = sum(1 for s in SIXTEENTHS if s <= t)
        out = fx.ghost(out, 8 if k % 2 else -8, .38)
    # triplet run: ink creeps in from the edges, double image on every note
    # (it clears while the arena soaks into paper)
    if TRIPLETS[0] - .05 <= t < C.ENGULF + .15:
        g = smooth((t - TRIPLETS[0] + .05) / .25) * (1 - smooth((t - C.ENGULF) / .15))
        out = fx.vignette(out, .45 * g)
        k = sum(1 for s in TRIPLETS if s <= t)
        out = fx.ghost(out, 6 if k % 2 else -6, .28 * g)
    sl = _dismantle_slices()
    if sl and C.CUBE is not None and C.CUBE.t_shut <= t < C.CUBE.slash_t:
        bx, by = fx.project_point(p, *C.CUBE.C)              # only round the box (big, but not the whole screen)
        hud_y = fx.project_point(p, 320, 388)[1]
        out = fx.dismantle(out, t, sl, (bx, by, 190 * p["zoom"], 150 * p["zoom"], hud_y))
    if CRISIS[0] <= t < CRISIS[1] and SOUL is not None:
        out = _crisis(out, t, p)
    for t0, dur, s in GLITCH:
        if t0 <= t < t0 + dur:
            out = fx.glitch(out, t, s)
    for t0, dur in INVERT:
        if t0 <= t < t0 + dur:
            out = fx.invert(out)
    for t0, dur, luma in LIFT:            # afterglow frame: the black lifts to grey
        if t0 <= t < t0 + dur:
            out = np.maximum(out, luma).astype(np.uint8)
    return out
