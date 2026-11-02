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
from common import ROOT, clamp, key_angle, key_vec, out_cubic, smooth

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
CANVAS = (lambda: (CANVAS_SET, SECTION_END))   # from here on the arena is a canvas
# the rest after the cube (M25-M28.5): the canvas wraps the buttons too. The burst
# throws the soul a little left of the middle; from there it heads for FIGHT the
# whole time (arrow keys held, 8 directions - no turn is given) while the ink of
# the burst comes alive, rings it and closes in on it beat by beat. Touching a
# button's edge is enough to trigger it: the soul grazes FIGHT's right edge only
# on the M28 downbeat (the hardest hit of the long hold), just as the ring is
# about to touch it: one press, and one slash cuts the whole ring
REST_RECT = (-220, 226, 860, 700)              # the sheet runs on past the screen (scene.overscan)
# where the soul touches FIGHT: walking straight down-left from KNOCK_TO, the
# first spot where the heart overlaps fight1.png (at 32,432) - by one pixel, on
# the middle of its right edge (the step before, it is still clear of it)
FIGHT_SOUL = (149, 451)
MENU_SOUL = (47, 453)                           # the soul's spot on the FIGHT button (UT's menu cursor)
PRESS_T = tm(28)
DEEP_T = tm(22.75) - .26     # the camera is deepest inside the canvas: the rectangle is there from now on (cinema);
                             # it holds there (a beat to take it in) until M22.75, then pulls back
KNOCK_DUR = .4
KNOCK_TO = (270, 330)                           # thrown just left of the middle
WALK_T = tm(24.75) + .42
_WALK_STEPS = int(PRESS_T * 60) - int(round(WALK_T * 60))
# from that close it walks much slower than UT's 2 px a frame (about 0.59), so
# that it still touches the button only on PRESS_T
WALK_SPEED = max(abs(KNOCK_TO[0] - FIGHT_SOUL[0]), abs(KNOCK_TO[1] - FIGHT_SOUL[1])) / _WALK_STEPS
WALK_ARRIVE = (int(round(WALK_T * 60)) + _WALK_STEPS) / 60
KNOCK_T = PRESS_T + .12                         # the slash has gone all the way round: the blow knocks the cut ring out
# after the cut it snaps back onto FIGHT's menu spot - fast, then slowing - just
# after the blow (by then the ring is wide enough for it to dart across inside it)
SETTLE = (KNOCK_T + .12, .25)                   # (start, duration) of going back onto it
# ...but the ring it is knocked out round (and the camera) follow it over smoothly,
# just in time for the M29 press: a ring dragged along at the soul's snap would
# stall the drops on one side against it (a twitch). The ring is wide by then:
# the soul is well inside it all the way
RING_SETTLE = (PRESS_T + .1, tm(29) - .04 - (PRESS_T + .1))


def ring_centre(t):
    """The middle of the cut ring after the cut: the soul's spot, eased over from the
    button's edge to the menu spot more slowly than the soul snaps there."""
    u = smooth(clamp((t - RING_SETTLE[0]) / RING_SETTLE[1]))
    return (FIGHT_SOUL[0] + (MENU_SOUL[0] - FIGHT_SOUL[0]) * u, FIGHT_SOUL[1] + (MENU_SOUL[1] - FIGHT_SOUL[1]) * u)


SOUL_SCRIPT_REST = [dict(kind="fly", t0=tm(24.75), dur=KNOCK_DUR, to=KNOCK_TO, hold=WALK_T),
                    dict(kind="walk", t0=WALK_T, to=FIGHT_SOUL, hold=SETTLE[0], speed=WALK_SPEED),
                    dict(kind="fly", t0=SETTLE[0], dur=SETTLE[1], to=MENU_SOUL, hold=tm(32.75))]
SOUL_SCRIPT = list(SOUL_SCRIPT_REST)      # (build() puts the dodge out of the paper box in front)
CLOSE_IN = [tm(25.75 + k / 4) for k in range(9)]   # the ring lunges in on every beat up to the press
# after the cut (M28-M33.5): the cut ring is not destroyed - the cut knocks every
# drop flying, and each swings round in a wide loop and comes back at the soul (now
# eased back onto FIGHT, the menu spot). FIGHT is pressed again on the M29 hold and
# UT's aim meter comes up; every note from M29.125 to M30.75 confirms a bar and
# smashes one drop as it gets back (so there are as many drops as notes). Then the player cuts the pure black over the paper (M31-M32.9375):
# piece by piece it comes off - it was a curtain of black ink, and on the M33
# downbeat the rest falls: Ink Sans was hiding behind it all along (he stands
# there through M33; the old placeholder takes over from M34)
SECTION_END = tm(48)                            # the canvas lasts through the descent (M32.75-M48)

_CHART_T = sorted({round(n.t, 4) for n in parse_chart(ROOT / "maidata.txt", OFFSET)[0]})
AIM_OPEN = tm(29)                               # FIGHT pressed again: the aim meter
EIGHTH = MEAS / 8
# while the cut ring wheels round the soul on FIGHT the camera follows the ring's
# middle, framing it left of the middle and a little down (cinema), so that the
# whole wide ring is in shot round it - the sheet runs on past the screen (scene.overscan)
SWIRL_OFF = (100, -15)                          # the camera's focus from the ring's middle
SWIRL_FOCUS = (MENU_SOUL[0] + SWIRL_OFF[0], MENU_SOUL[1] + SWIRL_OFF[1])


def swirl_focus(t):
    x, y = ring_centre(t)
    return (x + SWIRL_OFF[0], y + SWIRL_OFF[1])


def cam_zoom(t):
    """The camera's zoom (without its jolts) - the knocked ring is laid out on screen."""
    import cinema
    import fx
    return fx.camera_at(t, cinema.KEYS, [])["zoom"]


CONFIRMS = [t for t in _CHART_T if tm(29.125) - 1e-3 <= t < tm(31) - 1e-3]   # every note: a bar, a drop smashed
# the cut ring flies out from the blow all the way to the M28.5 tap - and there the
# first drop already recoils (no idle turn in between): each drop recoils for an
# eighth, then swoops at the soul for four - round, then in - and is smashed on its
# note, so three or four are swooping in at once, each from its own way
LAUNCH = [t - 4 * EIGHTH for t in CONFIRMS]
WINDUP = [t - EIGHTH for t in LAUNCH]
LAND_T = WINDUP[0]
CURTAIN_START = PRESS_T                         # the black over the paper is a curtain of ink from here on
REVEAL = tm(32.75)                              # the last piece is gone as the 5-4-5-4 sixteenths begin: Ink Sans is there
CURTAIN_CUTS = [t for t in _CHART_T if tm(31) - 1e-3 <= t < REVEAL - 1e-3]   # every note of M31-M32.6875: one more bar
CURTAIN_PIECE_CUTS = CURTAIN_CUTS[:-1]          # ...each cuts away a piece of the ink over him;
PARRY_T = CURTAIN_CUTS[-1]                      # the last one is blocked by his brush, blinking in
FIGHT_SELECT = (WALK_ARRIVE, tm(33.75))        # (the buttons rise away with the sheet in M33)
PRESSES = [PRESS_T, AIM_OPEN]                   # the cut (its edge just touched), then FIGHT for the aim meter
# Ink Sans is gone while the camera is right inside the canvas (out of shot) - when
# it pulls back he is not there; he is back while it is pushed in on the FIGHT cut
# (out of shot again), so he is simply there once it pulls back after it
INK_GONE = (DEEP_T, PRESS_T + .15)                  # (the camera starts pulling back from PRESS_T + .1)

# the stars' flow over the battle (backdrop.StarRain): one set, one speed; on
# every turn they all ease to a stop, turn, and ease back up
STAR_FLOWS = [(tm(17), "left"), (tm(18.5625), "down"), (tm(21), "up"), (tm(23), "down"), (tm(24), "up")]
STAR_TURN = .5
CLIMB_T = STAR_FLOWS[1][0]           # the stars turn down: from here on everything comes from above

# the paper moves with the world like a treadmill (a little faster than the
# stars): left through the run, down through the climb; it eases in after the
# title card, turns with the stars and stops for good when they turn at M21
# (the canvas spreads and the tunnel begins)
PAPER_SPEED = 520.0
PAPER_STOP = STAR_FLOWS[2][0]
EDGE_PUSH = []                                  # (t, px): a wall drunk at the left edge pushes it out
_WAYS = {"left": (-1.0, 0.0), "right": (1.0, 0.0), "up": (0.0, -1.0), "down": (0.0, 1.0)}


def paper_velocity(t):
    """(vx, vy) px/s of the paper under the canvas."""
    if not (STAR_FLOWS[0][0] <= t <= PAPER_STOP):
        return 0.0, 0.0
    i = max(0, bisect.bisect_right([f[0] for f in STAR_FLOWS], t) - 1)
    g = smooth((t - STAR_FLOWS[0][0]) / .25)
    if i > 0:
        g *= smooth((t - STAR_FLOWS[i][0]) / STAR_TURN)
    if i + 1 < len(STAR_FLOWS):
        g *= smooth((STAR_FLOWS[i + 1][0] - t) / STAR_TURN)
    dx, dy = _WAYS[STAR_FLOWS[i][1]]
    return dx * PAPER_SPEED * g, dy * PAPER_SPEED * g


_PV_T = np.arange(STAR_FLOWS[0][0] - .05, PAPER_STOP + .05, 1 / 240)
_PV = np.array([paper_velocity(t) for t in _PV_T])
_PV_X = np.concatenate([[0.0], np.cumsum(_PV[:-1, 0]) / 240])
_PV_Y = np.concatenate([[0.0], np.cumsum(_PV[:-1, 1]) / 240])


def paper_offset(t):
    """How far (dx, dy) px the paper has moved by t."""
    return float(np.interp(t, _PV_T, _PV_X)), float(np.interp(t, _PV_T, _PV_Y))


A.PAPER_OFFSET = paper_offset


# --------------------------------------------------------------------- the descent
# M32.75-M48. On the 5-4-5-4 sixteenths the whole sheet - the full width of the
# screen, the soul on it - is yanked down a little each time, while the buttons
# and the status bar painted on it stay where they were: they come away from it.
# Through M33 the camera sinks with the sheet: the buttons, the status bar and Ink
# Sans rise away, and the sheet comes up to fill the whole screen, carrying the
# soul up to the middle; it lands on M34. Then the sheet is the battle, the whole
# height of the screen: it folds, jumps from side to side and shakes, and the ink
# on it is thrown about by every move (add_sink). Everything on it is worked out
# in its own frame (the planner sees it standing still: LOCAL) and moved with it
# on screen (attacks.ArenaFrame, render.soul_at, scene.sink_paper). No stars here.
SINK0 = tm(32.75)
RISE = (tm(33), tm(33.8))
ARRIVE = tm(34)
NEW_END = tm(48)
LOCAL = False                                   # plan.py: everything in the arena's own frame
FULL, HALF_W, BAND_H = 640, 300, 480            # the arena: the whole screen wide / half of it; the screen's height
_DX0, _DY0 = 0, 226                             # at first it is the sheet under the buttons (its top where the sheet's is)
SOUL_RESETS = [(SINK0, (47, 453 - _DY0)),       # (plan.py: from here the soul is in the arena's frame...
               (NEW_END, (320, 313))]           # ...and back in the old box after it)
LEFT_HALF = (HALF_W - FULL) / 2                 # lx of the sheet's left half (its left edge where the sheet's was)
JUMP = FULL - HALF_W                            # dx that carries the left half over to the right half of the screen
FOLDS = [tm(m) for m in (34.125, 34.25, 34.375)]   # the band folds its right end under, a third at a time
JUMPS = [(tm(35), 1), (tm(35.5), -1), (tm(36), 1)]   # (landing, way): the half jumps across on the pairs
SLOSH_V = 116.0      # px/s the walls' gaps move at: the soul (120) has to keep running, turning on the M35.5 pair
SLOSH_GAP = 34       # px: their gaps (the whole heart is 16: 9 px to spare each side)
NEEDLES = [tm(36.5 + k / 16) for k in range(6)]     # the 2-7-1-8-2-7
RING_DRAWS = [tm(36.375 + k / 16) for k in range(4)]    # four rings of ink round the soul, from the walls' last two notes...
RING_FIRES = [tm(36.625 + k / 16) for k in range(4)]    # ...then their beads fly, one ring on each of the last four
CENTRE_T = tm(36.875)                           # the hold: it folds over to the other half of the screen
BREAK_T = tm(38.875)                            # the break: folded back over, slamming down on it, for the climax
# M37 the call (ink shooting up past the soul, a stroke across, the 24ths); M38
# - where its mirrored answer was - no attacks: the sheet folds itself up small
# round the soul, note by note, flies off across the screen spinning, and opens
# out again on the break on the other side, near the edge (pack_geometry); the soul
# walks to where the run will start from meanwhile (GOALS): when it opens out the
# soul is at the bottom of the screen, ready to run
CALL_M = (37,)
LANE_SP = 52         # px between the beads on M37's lines of ink (the lanes between them: 42 px, the heart 16)
LANE_STEP = 6.0      # px each line's beads are off the last's (the lanes move down, a step a shot)
TURN_T = tm(37.875)                             # the hold: the one slow turn, onto M38's first note
# the climax (M39-M41 so far): from the moment each round starts a sea of ink
# spikes shot in from off the sheet rises behind the soul and drives it along the
# sheet as fast as it can run, right through the bar's trill; on its pairs the page
# is folded over three times, hard - it lies across the screen, slams over, stands
# up again somewhere else - and the soul starts the next round somewhere else
# entirely, running another way on screen: up, then down, then to the right (each
# round sets off from about where the last one ended on the sheet). Little bubbles
# of ink are shot at it head-on through each sea
CORE_M = (39, 40, 41, 43, 44, 45)
CUT_M = 42                                      # between: the sheet cut down round the soul (add_sink)
PAIR_TURN = .09                                 # s each fold takes (it lands on the note)


def core_trill(m):
    return [tm(m + (6 + 3 * i) / 48) for i in range(6)]


def core_pairs(m):
    return [tm(m + (24 + 6 * j) / 48) for j in range(4)]


# (start, how long, the fold line's angle on screen (deg; through the sheet's middle),
#  where the sheet's middle lands on screen (None: where it is))
# after the rings the half is flung over to the other side of the screen (onto
# M37); through M37-M38 it is flung over on the notes - lying across the screen,
# standing up, from side to side, top to bottom (one slow turn on M37's hold);
# the break flings it back to the right half. Then each round's pairs: the folds
# are chosen so that on screen the rounds run up, down, right | left, up, down
# (RUN_WAY: which way each runs on the sheet itself - up it for the first three,
# back down it for the last three). M42's pairs: the cut-down sheet opens out
# again, then one fold (UNFOLD_T, add_sink)
_E = lambda m: tm(m) - PAIR_TURN
FOLD_KEYS = [(CENTRE_T, EIGHTH, 90, (150, 240))]
_PAIR_FOLDS = {39: ((45, (320, 150)), (0, (320, 330)), (-45, (150, 240))),     # -> M40 standing on the left
               40: ((-45, (320, 330)), (0, (320, 150)), (90, (320, 330))),     # -> M41 lying along the bottom
               41: ((45, (490, 240)), (90, (150, 240))),                       # -> on the left; then it opens out
               43: ((-45, (320, 330)), (0, (320, 150)), (90, (320, 330))),     # -> M44 lying along the bottom
               44: ((45, (150, 240)), (0, (490, 240)), (90, (150, 240))),      # -> M45 standing on the left
               45: ((0, (490, 240)),)}                                         # -> over to the right, upside down:
                                                                               # the soul up top, where the slide
                                                                               # starts; then it opens out (OPEN2_T)
for _m, _folds in _PAIR_FOLDS.items():
    _p = core_pairs(_m)
    FOLD_KEYS += [(_p[j] - PAIR_TURN, PAIR_TURN, deg, target) for j, (deg, target) in enumerate(_folds)]
# M41's last pair: the half opens out into the whole sheet, the whole screen (a
# warning on it); M42 the cut; on its first pair the cut-down sheet is whole again
# (UNFOLD_T), on its second it folds back in half (REFOLD_T) and M43's run starts
OPEN_T = tm(41.75)
# (a move that is never seen: while the half lies still on M37 its own frame is put
# back over the left of the screen, and sheet_state makes up for it on screen - so
# that when it opens out the whole sheet lies inside the picture it is drawn in)
SHIFT_T = tm(37.5)
UNFOLD_T = tm(42.75)                            # the cut-down sheet opens out again (CircleCut)
OPEN2_T = tm(45.625)                            # M45's second pair: the half opens out into the whole sheet again -
                                                # the canvas that soaks through and lies down into a pool (plunge)
REFOLD_T = tm(42.875)
FOLD_KEYS.sort(key=lambda f: f[0])
RUN_WAY = (-1, -1, -1, 1, 1, 1)                 # -1: up the sheet (the spikes come from its bottom)
CUT_AWAY = (125, 55)                            # px from the soul to the middle of M42's cut (add_sink)
GATE_DY = 36                                    # px between the second wave's gates (add_sink): with
GATE_AT = 26                                    # the first this far ahead; a row ~12 px thick (what hurts),
                                                # the heart 14, a 40 px gap - it can weave ~40 px from one
                                                # gap to the next flat out, and they are 32 apart
# (plan.py: in these it runs along the line the second wave's gates are laid out on
# - (start, end, y -> x) - looking a little ahead, like a player who sees the gaps
# coming; add_sink fills it in)
LINES = []


def core_run(k):
    """Round k's run: (its start - the sea comes on at once, from the fold it lands
    on -, the sixteenths it comes in on, through the bar's trill)."""
    m = CORE_M[k]
    t0 = {39: BREAK_T, 43: REFOLD_T}.get(m) or core_pairs(m - 1)[2]
    s16 = MEAS / 16
    trill = core_trill(m)
    return t0, [t0 + j * s16 for j in range(int(round((trill[-1] - t0) / s16)) + 1)]


# (plan.py: in these the soul runs flat out the sea's way - like a player holding the key)
RUNS = [(core_run(k)[0], core_run(k)[1][-1] + .1, RUN_WAY[k]) for k in range(len(CORE_M))]
# (plan.py: in these it walks to a spot on the sheet, nothing else to mind - the
# start of the first run: when the page lands for the last time it is at the
# bottom of the screen)
PACK_SPOT = (150, 400)                          # (M38: where the soul waits, shut in the packet)
GOALS = [(tm(37.8), BREAK_T, PACK_SPOT)]
_GOALS0 = list(GOALS)                           # (add_sink adds one: M42, it keeps to the middle of the cut)
# (plan.py: between one run and the next it stays where it got to - only moving
# to keep clear - so the next run starts from where the last one ended)
HOLDS = [(RUNS[k][1], RUNS[k + 1][0]) for k in range(len(RUNS) - 1)] + [(RUNS[-1][1], NEW_END)]
FOLD_LANDS = [t0 + d for t0, d, _, target in FOLD_KEYS if target is not None]


def _reflect(deg):
    a = math.radians(deg)
    c2, s2 = int(round(math.cos(2 * a))), int(round(math.sin(2 * a)))
    return np.array([[c2, s2], [s2, -c2]])


def sheet_state(t, pack=True):
    """The page's folds so far - a render transform (scene): the planner never sees
    them. (M, b, fold): the finished ones make up p -> M p + b on screen; fold: the
    one under way - (its line's point and angle, how far over (rad), how far it
    has moved the sheet so far) - or None. Each comes over faster and faster and
    slams down on its note."""
    M, b, cur = np.eye(2, dtype=int), np.zeros(2), None
    shifted = packed = False
    for t0, d, deg, target in FOLD_KEYS:
        if t < t0:
            break
        if not shifted and t0 >= SHIFT_T and t >= SHIFT_T and not LOCAL:   # (the unseen move: made up for)
            b, shifted = b - M @ np.array([-JUMP, 0.0]), True
        if pack and not packed and t0 >= BREAK_T and t >= BREAK_T and not LOCAL:   # (M38's flight: it has landed)
            b, packed = b + np.asarray(PACK_MOVE), True
        x, y, w, h = arena_box(t0)
        P = M @ np.array([x + w / 2, y + h / 2]) + b      # the sheet's middle on screen as it starts
        D = (np.asarray(target, float) - P) if target is not None else np.zeros(2)
        u = clamp((t - t0) / d)
        if u >= 1:
            F = _reflect(deg)
            M, b = F @ M, P + F @ (b - P) + D
        else:
            cur = (P, deg, math.pi * u ** 2, D * u ** 2)
    if not shifted and t >= SHIFT_T and not LOCAL:
        b = b - M @ np.array([-JUMP, 0.0])
    if pack and not packed and t >= BREAK_T and not LOCAL:
        b = b + np.asarray(PACK_MOVE)
    return M, b, cur


# M38: the sheet folds itself up round the soul like a letter - really folding, a
# half at a time, turning over in depth and showing its back as it comes down over
# the soul (which is shut in); the first lifts on the hold's start, hangs
# trembling, and slams down on the next note; three more snap down on the notes
# after, each one landing with a jolt. The packet winds up, hops across the bottom
# of the screen on the 24ths - a quarter turn a hop, leaving ghosts - and bursts
# open on the break on the right half (PACK_MOVE further over). A render overlay
# (fx.paper_pack) in the screen's frame of the half as it starts: x 0-300, y 0-480,
# cut into 4 x 4 cells.
PACK_MOVE = (JUMP, 0.0)
PACK_CELL = (75.0, 120.0)
# (start, landing, crease axis ('h': the line y = c, 'v': x = c), c, side that turns over (-1: the lower x/y))
PACK_FOLDS = [(TURN_T, tm(38.125), "h", 240.0, -1),
              (tm(38.25) - .09, tm(38.25), "v", 150.0, 1),
              (tm(38.375) - .09, tm(38.375), "h", 360.0, -1),
              (tm(38.5) - .09, tm(38.5), "v", 75.0, -1)]
PACK_C0 = np.array([112.5, 420.0])              # the packet's middle when it is folded
PACK_HOPS = [tm(38.625 + k / 24) for k in range(4)]
# it bursts open, a fold after another (the last first - one at a time: two at once
# would tear it), the biggest slapping down flat on the break
PACK_OPENS = {3: (tm(38.75), tm(38.785)), 2: (tm(38.775), tm(38.81)), 1: (tm(38.80), tm(38.84)),
              0: (tm(38.83), BREAK_T)}
# the packet's knocks (for fx.pack_bursts): (t, how hard) - each fold landing, each
# hop landing, each fold flung open
PACK_KNOCKS = ([(t1, 1.0) for _, t1, *_ in PACK_FOLDS] + [(h, .35) for h in PACK_HOPS]
               + [(PACK_OPENS[i][0] + .012, .6) for i in (3, 2, 1, 0)])


def _pack_theta(i, t):
    """How far over fold i is at t (0 flat ... pi down)."""
    t0, t1, _, _, _ = PACK_FOLDS[i]
    if t < t0:
        th = 0.0
    elif i == 0 and t < t1 - .07:                      # up quickly, then hanging there, trembling
        th = math.pi * .55 * (1 - (1 - clamp((t - t0) / .06)) ** 3) + .05 * math.sin((t - t0) * 70) * clamp((t - t0) / .06)
    elif i == 0:
        u = clamp((t - (t1 - .07)) / .07)
        th = math.pi * (.55 + .45 * u * u)
    else:
        u = clamp((t - t0) / (t1 - t0))
        th = math.pi * u ** 2.2
    lo, hi = PACK_OPENS[i]                             # bursting open: the last first, one after another
    if t >= lo:
        u = clamp((t - lo) / (hi - lo))
        th = math.pi * (1 - u) ** (1.3 if i == 0 else 2.0)   # (flung open; the last slaps down flat on the break)
    return th


def _pack_place(t):
    """The packet as a whole: (where its middle is, turn, squash x, squash y)."""
    T, rot, sx, sy = PACK_C0.copy(), 0.0, 1.0, 1.0
    for t0, t1, *_ in PACK_FOLDS[1:]:                  # each fold lands with a jolt
        b = t - t1
        if 0 <= b < .07:
            k = math.sin(math.pi * b / .07) * .07
            sx, sy = sx * (1 + k), sy * (1 - k)
    w0 = PACK_FOLDS[-1][1]
    if w0 <= t < PACK_HOPS[0] - (PACK_HOPS[1] - PACK_HOPS[0]):   # winding up: squashed, drawn back, trembling
        u = clamp((t - w0) / .05)
        sx, sy = 1 + .22 * u, 1 - .16 * u
        T = T + np.array([-14.0 * u + 2 * math.sin(t * 90), 0.0])
    step = PACK_HOPS[1] - PACK_HOPS[0]
    P = [PACK_C0 + np.asarray(PACK_MOVE) * k / 4 for k in range(5)]
    for k, h in enumerate(PACK_HOPS):                  # a hop on each 24th, a quarter turn each
        if h - step <= t < h:
            u = (t - (h - step)) / step
            e = 1 - (1 - u) ** 2
            T = P[k] + (P[k + 1] - P[k]) * e + np.array([0.0, -38.0 * math.sin(math.pi * u)])
            rot = math.pi / 2 * (k + e)
            st = 1 + .18 * math.sin(math.pi * u)
            sx, sy = (st, 1 / st) if k % 2 else (1 / st, st)
    if t >= PACK_HOPS[-1]:
        T, rot = P[4].copy(), 0.0
    return T, rot, sx, sy


def pack_geometry(t):
    """M38's folding up, at t: the cells as they lie on screen, back to front - each
    (its corners on screen (4 x 2), front showing?, the rect of the screen it
    shows when face up) - and the packet's middle. None outside M38."""
    if not (TURN_T <= t < BREAK_T) or LOCAL:
        return None
    cw, ch = PACK_CELL
    cells = []
    for i in range(4):
        for j in range(4):
            x0, y0 = j * cw, i * ch
            P = np.array([[x0, y0, 0.0], [x0 + cw, y0, 0.0], [x0 + cw, y0 + ch, 0.0], [x0, y0 + ch, 0.0]])
            cells.append(dict(P=P, src=(x0, y0, x0 + cw, y0 + ch), layer=0.0))
    nl = 1.0
    for f, (t0, t1, ax, c, side) in enumerate(PACK_FOLDS):
        th = _pack_theta(f, t)
        k = 1 if ax == "h" else 0                     # (the coordinate across the crease)
        moving = [cl for cl in cells if side * (cl["flat"][:, k].mean() - c) > 0] if all("flat" in cl for cl in cells) \
            else [cl for cl in cells if side * (cl["P"][:, k].mean() - c) > 0]
        mids = {id(cl) for cl in moving}
        for cl in cells:                              # (where each lies once this fold is down: for the next)
            fl = cl.get("flat", cl["P"][:, :2].copy()).copy()
            if id(cl) in mids:
                fl[:, k] = 2 * c - fl[:, k]
            cl["flat_next"] = fl
        if th > 0:
            for cl in moving:                          # (turned over about the crease, towards us)
                P = cl["P"].copy()
                d = P[:, k] - c
                z = P[:, 2]
                P[:, k] = c + d * math.cos(th) + side * z * math.sin(th)
                P[:, 2] = -side * d * math.sin(th) + z * math.cos(th) - .01
                cl["P"] = P
                lay = cl["layer"] / max(nl, 1) * .5        # (the stack keeps its order until past upright)
                cl["layer"] = nl + (lay if th < math.pi / 2 else .5 - lay)
            nl += 1
        for cl in cells:
            cl["flat"] = cl.pop("flat_next")
    T, rot, sx, sy = _pack_place(t)
    cr, sr = math.cos(rot), math.sin(rot)
    f_ = 900.0
    out = []
    for cl in sorted(cells, key=lambda c_: c_["layer"]):
        P = cl["P"]
        z = P[:, 2]
        cen = PACK_C0
        q = cen + (P[:, :2] - cen) * (f_ / (f_ + z))[:, None]       # (in depth: nearer, bigger)
        q = (q - PACK_C0) * [sx, sy]
        q = np.stack([cr * q[:, 0] - sr * q[:, 1], sr * q[:, 0] + cr * q[:, 1]], 1) + T
        e1, e2 = q[1] - q[0], q[3] - q[0]
        front = (e1[0] * e2[1] - e1[1] * e2[0]) > 0
        n = np.cross(P[1] - P[0], P[3] - P[0])
        out.append((q, front, cl["src"], abs(n[2]) / max(1e-9, np.linalg.norm(n))))   # (and how flat it lies)
    return out, T


def sheet_centre(t):
    """Where the sheet's middle is on screen (folds and all)."""
    M, b, cur = sheet_state(t)
    x, y, w, h = arena_box(t)
    c = M @ np.array([x + w / 2, y + h / 2]) + b
    if cur is not None:
        c = cur[0] + cur[3]
    pg = pack_geometry(t)
    if pg is not None:                                     # (folded up and flying: the packet)
        q = np.concatenate([g[0] for g in pg[0]])
        c = (q.min(0) + q.max(0)) / 2
    return c


def _arena_keys():
    s16, b = MEAS / 16, lambda m: tm(m)
    K = []                                       # (t, dx, dy, w, h, lx, dur, ease)
    for j in range(4):                           # the sheet is yanked down on each sixteenth
        K.append((SINK0 + j * s16, _DX0, _DY0 + 6 * (j + 1), FULL, BAND_H, 0, .05, "out"))
    K.append((b(33), 0, 0, FULL, BAND_H, 0, ARRIVE - b(33) - .02, "smooth"))   # carried up into the band
    K.append((ARRIVE, 0, 0, FULL, BAND_H - 16, 0, .05, "out"))                 # lands
    K.append((ARRIVE + .07, 0, 0, FULL, BAND_H, 0, .22, "out"))
    for j, t in enumerate(FOLDS):                # folds: the right end goes under, the left edge stays put
        w = FULL + (HALF_W - FULL) * (j + 1) / len(FOLDS)
        K.append((t, 0, 0, w, BAND_H, (w - FULL) / 2, .06, "out"))
    x = 0.0
    for t, way in JUMPS:                         # a jump: it draws back, then snaps across and lands on the note
        K.append((t - .2, x - 14 * way, 0, HALF_W, BAND_H, LEFT_HALF, .1, "smooth"))
        x = JUMP if way > 0 else 0.0
        K.append((t - .085, x + 10 * way, 0, HALF_W, BAND_H, LEFT_HALF, .085, "in"))
        K.append((t, x, 0, HALF_W, BAND_H, LEFT_HALF, .12, "out"))
    # (from here on the page's folds move it: FOLD_KEYS) - but before M42 the half
    # opens out to the whole sheet (its left edge stays put), and after it folds back
    K.append((SHIFT_T, 0, 0, HALF_W, BAND_H, LEFT_HALF, 1e-6, "out"))
    K.append((OPEN_T - .004, 0, 0, FULL, BAND_H, 0, .004, "out"))
    K.append((REFOLD_T - .1, 0, 0, HALF_W, BAND_H, LEFT_HALF, .06, "out"))
    K.append((OPEN2_T - .004, 0, 0, FULL, BAND_H, 0, .004, "out"))
    return K


ARENA_KEYS = _arena_keys()


def _ease(kind, u):
    u = clamp(u)
    if kind == "out":
        return 1 - (1 - u) ** 3
    if kind == "in":
        return u * u
    return u * u * (3 - 2 * u)


def arena_state(t):
    """(dx, dy, w, h, lx) of the arena at t: its offset on screen, its size, and
    which part of the sheet it is (lx: its middle, from the middle of the sheet)."""
    st = [_DX0, _DY0, float(FULL), float(BAND_H), 0.0]
    for tk, dx, dy, w, h, lx, dur, ease in ARENA_KEYS:
        if t < tk:
            break
        e = _ease(ease, (t - tk) / dur)
        st = [a + (b_ - a) * e for a, b_ in zip(st, (dx, dy, w, h, lx))]
    return tuple(st)


def arena_offset(t):
    """How far (dx, dy) the arena - and everything on it - is moved on screen."""
    if LOCAL or not (SINK0 <= t < NEW_END):
        return 0, 0
    dx, dy = arena_state(t)[:2]
    return int(round(dx)), int(round(dy))


def arena_size(t):
    w, h = arena_state(t)[2:4]
    return int(round(w / 2)) * 2, int(round(h / 2)) * 2


def arena_local(t):
    """The arena in its own frame (what the planner sees): (x, y, w, h) - the
    screen's frame once it fills the screen (dx = dy = 0, the whole width)."""
    w, h = arena_size(t)
    lx = arena_state(t)[4]
    return 320 - w // 2 + int(round(lx)), (480 - h) // 2, w, h


def arena_box(t):
    x, y, w, h = arena_local(t)
    ox, oy = arena_offset(t)
    return x + ox, y + oy, w, h


def rise(t):
    """How far the rest of the sheet, the buttons and Ink Sans have risen (px)."""
    if t < RISE[0]:
        return 0.0
    u = clamp((t - RISE[0]) / (RISE[1] - RISE[0]))
    return 560.0 * u ** 1.8


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
    dict(m=28, box=(200, 150), tap="none", hold="none", slide="none", head="head_black_eyes.png",
         body="body.png"),
    # the descent (M32.75-M48): the arena is a torn-out piece of the sheet, carried
    # down to a new place and sliding left and right under fire (add_sink)
    dict(m=32.75, box=(220, 150), tap="custom", hold="custom", slide="custom", head="head_black_eyes.png",
         body="body.png"),
    # (old placeholder from here on, until the next part is directed)
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
    BOX_TRANSITION before it starts; sections may set their own lead/dur).
    During the descent it is the moving piece of paper (arena_box)."""
    if SINK0 <= t < NEW_END:
        return arena_box(t)
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
    add_sink(tl, notes)
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
        tl.attacks.append(A.HuntingInk((g.cx, g.cy), n.t, hunt_target, PRESS_T, CONFIRMS[-1] + .5, lunges=CLOSE_IN,
                                       n=len(CONFIRMS), seed=nxt(), confirms=CONFIRMS, launches=LAUNCH,
                                       windups=WINDUP, t_land=LAND_T, home=MENU_SOUL,
                                       soul_path=ring_centre, zoom=cam_zoom, t_knock=KNOCK_T))
        # FIGHT on the M29 hold: UT's aim meter, painted on the paper; one bar
        # per drop, sliding in as its drop flies and stopped (confirmed) on its
        # note as the drop gets there - and that drop is smashed
        # it stays up through M31-M32.9375 - a bar for every piece of ink cut off him -
        # and is gone the moment the last one is (M33)
        tl.attacks.append(A.FightTarget((320, 316), 268, 54, AIM_OPEN, CONFIRMS + CURTAIN_CUTS, REVEAL - .14,
                                        slides=[min(2 * EIGHTH, t - AIM_OPEN) for t in CONFIRMS]
                                        + [.3] * len(CURTAIN_CUTS), fade=.12))
        tl.extra_impacts.append((n.t, 8.0))
        tl.flashes.append((n.t, 40))
    add_rest(tl, nxt)
    # entering the canvas: the break pair 2b/7b lands as two great pale blots
    # (no warning is possible right after the title card, so they do not hurt)
    first = [n for n in seg if abs(n.t - tm(17)) < 1e-3]
    for n in first:
        tl.attacks.append(A.SplashRing(np.array(g.ring(*np.array(key_vec(n.key)) * .6)), n.t, n=9, speed=150,
                                       seed=nxt(), tele=.0, big=True, harmless=True, scroll=True))
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
    strong = [tm(17.375), tm(18), tm(18.375)]      # (the M19 downbeat is after the turn: from above)
    bars = []
    for k, t in enumerate(strong):
        use = _wall_rows(g, 8, aim_at(t - .4, g)[1], 1 if k % 2 else -1)
        bars.append(dict(side="R", paint=t - .3, fire=t, pops=[(t - .17 + .008 * i, y) for i, y in enumerate(use)]))
    walls = A.PenVolleys(g, bars, tm(17), tm(19), speed=620.0, seed=nxt(), trail=True)
    tl.attacks.append(walls)
    # each wall is drunk at the left edge all at once: it pushes the paper out there
    EDGE_PUSH[:] = [(min(b.t_out for b in bar["bones"]), 12.0) for bar in walls.bars]
    # the paper itself moves: old pale marks ride past on it
    tl.attacks.append(A.PaperRun(inner, paper_offset, STAR_FLOWS[0][0], PAPER_STOP, seed=1718))
    run = []
    for n in taps:
        # (no single bones while a wall is sweeping through - they would plug its gap)
        if tm(17) + 1e-3 < n.t < CLIMB_T - 1e-3 and not any(-1e-3 < n.t - s_ < .36 for s_ in strong) \
                and not is_slash(n.t):
            y = float(np.clip(aim_at(n.t - .32, g)[1], g.iy0 + 10, g.iy1 - 10))
            run.append((n.t, (g.x + g.w + 34, y), (-1.0, 0.0)))
    shots_run = A.BoneShots(inner, run, speed=640.0, seed=nxt(), trail=True, through=True)   # (out through the far side)
    tl.attacks.append(shots_run)
    # ink gathers where each single bone will form, just before it launches - the
    # warning (the walls have their self-writing bar). Own seed: nxt() untouched
    tl.attacks.append(A.InkGather([dict(kind="pt", x=b.S[0], y=b.S[1], t=b.tg) for b in shots_run.bones],
                                  tm(17), seed=1717))
    # ---- CLIMB (from the stars' turn down, M18.5625, to M20.5): everything comes
    #      from above - orbs over the arena fire down at the soul (not while the
    #      last wall from the right is still sweeping through: it would plug its gap)
    shots = []
    for n in taps:
        if CLIMB_T - 1e-3 <= n.t < tm(20.54) - 1e-3 and not is_slash(n.t) \
                and not any(-1e-3 < n.t - s_ < .36 for s_ in strong):
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
    # ---- M22.75-M24.75 折纸立方体: the camera pulls back out of the canvas - the
    #      soul was inside a rectangle all along. On the M22.75 sixteenths the paper
    #      round it is scored into a box net and the top flap creases up (30/60/90:
    #      the back wall). On the twelfth triplets the bottom, left and right flaps,
    #      each rising while its strip of floor is washed pale, slap down in three
    #      panels (inked black: the hazard) and stand up as walls; on the last
    #      triplet the lid comes down from the camera and the warning goes up once -
    #      the pieces of the box that will be cut are marked. The lid shuts on the
    #      M24 downbeat; the shut box lifts towards us and tumbles like a die, 120 deg
    #      an eighth (a full turn in three), the marked pieces cut to bits in a
    #      random order as it turns (cut floor cells: holes of ink); it drops back,
    #      the two slides cut an X across its floor, and on the break it bursts.
    tw = lambda m, k: tm(m + k / 12)
    slams = {"bottom": (tm(22.75) + .02, [tw(23, k) for k in range(3)]),          # left to right
             "left": (tw(23.25, 0) - .42, [tw(23.25, k) for k in range(3)]),       # top to bottom
             "right": (tw(23.5, 0) - .42, [tw(23.5, 2 - k) for k in range(3)])}   # bottom to top
    box = A.OrigamiBox((g.ix0, g.iy0, g.ix1, g.iy1), DEEP_T, tm(22.75), [tm(22.75 + k / 16) for k in (1, 2, 3)],
                       slams, [tw(23.75, k) for k in range(3)], tm(24), [tm(24.125), tm(24.25), tm(24.375)],
                       tm(24.5), tm(24.75), soul_at_warn=aim_at(tw(23.75, 0), g), seed=nxt())
    tl.attacks.append(box)
    # the moment the warning goes up the soul makes for the one clear cell (arrow
    # keys held, UT speed) and stays there through the cuts, the X and the burst
    sx0, sy0, sx1, sy1 = box.cell_xy(*box.safe)
    safe_xy = (float(box.C[0] + (sx0 + sx1) / 2), float(box.C[1] + (sy0 + sy1) / 2))
    SOUL_SCRIPT[:] = [dict(kind="walk", t0=tw(23.75, 0) + .05, to=safe_xy, hold=tm(24.75))] + SOUL_SCRIPT_REST
    global CUBE
    CUBE = box


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
    # the last bar's cut: his brush blinks off his back and blocks it (the ink
    # over him is all gone by then)
    shield = A.InkShieldStrikes((314, 150), [PARRY_T], seed=nxt())
    tl.attacks.append(shield)
    actions.BRUSH_AWAY[:] = [shield.brush_away]
    for t in PRESSES:
        tl.extra_impacts.append((t, 6.0 if t == PRESS_T else 3.0))
    for t in CONFIRMS:                                  # each confirmed bar smashes a drop
        tl.extra_impacts.append((t, 2.5))


# --------------------------------------------------------------------- the descent's attacks
def add_sink(tl: Timeline, notes):
    """M34-M47: the band is the battle. Every move it makes throws the ink on it
    about - attacks made in its own frame and moved with it on screen
    (attacks.ArenaFrame):
      - M34: it lands; on the three pairs its right end folds under, a third at a
        time (it is the sheet's left half now) - the folds squeeze the ink on it
        against its right edge, and on the run it sloshes across, a step a note;
      - M35-M36.4375: on each pair it jumps to the other side of the screen and the
        ink, left behind, sloshes back across it the other way, a step a note
        (attacks.SloshWave: one gap in the wall, shown while it lies pooled);
      - M36.5: the 2-7-1-8-2-7 - rings of ink round the soul, whose beads then fly
        at it from every side (attacks.InkRings);
      - M36.875: it glides to the middle."""
    atts = []
    s16 = MEAS / 16
    w, h = HALF_W, BAND_H
    x0, y0 = 320 - w // 2 + int(round(LEFT_HALF)), (480 - h) // 2
    b = BORDER
    rect = (x0 + b, y0 + b, x0 + w - b, y0 + h - b)         # the half's paper, in its own frame
    lo, hi = rect[1] + 22, rect[3] - 22

    def aim(t):
        p = AIM(t) if AIM is not None else None
        return p if p is not None else ((rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2)

    def inside(x, y, r=rect, m=34):
        return float(np.clip(x, r[0] + m, r[2] - m)), float(np.clip(y, r[1] + m, r[3] - m))

    full = (b, b, FULL - b, BAND_H - b)                  # the whole sheet, in its own frame (M33-M34)
    # M33, going down: ink in the world it falls past shoots up through the sheet on
    # every note - one rod at the soul, one beside it (a side in turn) - and the
    # slide is a stroke painted across the sheet as it lands
    rods = []
    for k, t in enumerate(t for t in _CHART_T if tm(33) - 1e-3 <= t < tm(34) - 1e-3 and abs(t - tm(33.625)) > 1e-3):
        sx = aim(t - .3)[0]
        rods.append((t, inside(sx, 0, full, 10)[0]))
        rods.append((t, inside(sx + (46 if k % 2 == 0 else -46), 0, full, 10)[0]))
    atts.append(A.InkRod(full, rods))
    t_s = tm(33.875)                                    # (the slide's star is on M33.625, it slides from M33.875)
    sx, sy = aim(t_s - .1)
    th = math.radians(25)
    d, nrm = np.array([math.cos(th), math.sin(th)]), np.array([math.sin(th), -math.cos(th)])
    Q = np.array([sx, sy]) + nrm * 55                   # (it passes above the soul)
    atts.append(A.InkStroke([tuple(Q + d * u * 700) for u in np.linspace(-1, 1, 9)], t_s, width=14,
                            lead=t_s - tm(33.625), sweep=EIGHTH, seed=5400))
    # the folds: each end slams down wet - where it lands is inked
    for j, t in enumerate(FOLDS):
        wa = FULL + (HALF_W - FULL) * j / len(FOLDS)
        wb = FULL + (HALF_W - FULL) * (j + 1) / len(FOLDS)
        atts.append(A.FoldFlap(t, wb, wa - wb, y0, y0 + h, slam=True))
    # the sloshing: a wall on every quarter - two to each half bar - each across in
    # four steps, a step a sixteenth, so that one is always on the move and there is
    # never more than one; on the pairs, where the half jumps, they turn round with it
    # (the next comes from the other side at once). Each lies pooled at the edge it
    # starts from for two sixteenths first (pale, its gap showing - but only on paper
    # that is not black: never pale over ink that still hurts), then it is black as it
    # slams on. Nothing of it shows off the paper. The gaps run along one line, as fast as the soul can go (SLOSH_V,
    # it runs 120) and only just wide enough for it (SLOSH_GAP: 9 px to spare each
    # side), so it has to keep running flat out: the first bar one way, turning on the
    # pair in the middle (M35.5), the second back. Its mark stops hurting fast (dry),
    # but its streaks stay on a while (linger)
    halves = [(tm(34.5), -1)] + [(t, way) for t, way in JUMPS]
    t_a, t_turn = FOLDS[-1] + .1, tm(35.5)               # (the first shows as the last fold is down: it sets off)
    sy0 = aim(t_a - .1)[1]
    D = -1 if sy0 >= (lo + hi) / 2 else 1                 # (the first bar towards the further edge)

    def track(t):
        if t < t_turn:
            return sy0 + D * SLOSH_V * (t - t_a)
        return sy0 + D * SLOSH_V * (t_turn - t_a) - D * SLOSH_V * (t - t_turn)

    for h, (th, way) in enumerate(halves):
        for j in range(2):
            go = th + 4 * j * s16
            gy = float(np.clip(track(go + s16), lo, hi))   # (it is over the middle of the sheet a step on)
            t_pool = FOLDS[-1] - .02 if h == j == 0 else go - 2 * s16
            atts.append(A.SloshWave(rect, way, t_pool, [go] + [go + k * s16 for k in range(4)], gy,
                                    seed=4400 + 2 * h + j, dry=2.6, linger=2.2, gap=SLOSH_GAP, under=True, hz=0.0))
    # the 2-7-1-8-2-7, after the walls: three rings of ink round the soul, brushed
    # on the first three notes, the outermost first (hollow, grey, turning - all
    # three there at once), then firing their beads in at it from every side one by
    # one on the last three, the innermost first (so that each ring's shots land on
    # their own)
    cx_, cy_ = aim(RING_DRAWS[0] - .05)
    atts.append(A.InkRings((cx_, cy_), (146, 112, 78, 44), RING_DRAWS, RING_FIRES[::-1], n=8, seed=5460))
    # M37, the call (its mirrored answer on M38 is gone: there the page is only
    # flung over): ink shooting up either side of the soul on the two eighths; the
    # slide's star shows the stroke it will be (pale), and on the slide it is
    # painted across the sheet, black, while on the 24ths ink shoots up through it,
    # stepping across
    for k, m in enumerate(CALL_M):
        bt = lambda f: tm(m + f)
        atts.append(A.InkRod(rect, [(bt(.125), inside(aim(bt(.125) - .3)[0] - 40, 0, rect, 8)[0]),
                                    (bt(.25), inside(aim(bt(.25) - .3)[0] + 40, 0, rect, 8)[0])]))
        sx, sy = aim(bt(.6))
        d, nrm = np.array([1, -1]) / math.sqrt(2), np.array([-1, -1]) / math.sqrt(2)
        Q = np.array([sx, sy]) - nrm * 60                   # (it passes below and left of the soul)
        path = [tuple(Q + d * u * 380 + nrm * 14 * (1 - u * u)) for u in np.linspace(-1, 1, 9)]
        atts.append(A.InkStroke(path, bt(.625), width=14, lead=bt(.625) - bt(.375), sweep=EIGHTH, seed=5350 + k))
        sx = aim(bt(.55))[0]                               # the 24ths: four rods stepping across past the soul
        atts.append(A.InkRod(rect, [(tm(m + .625 + j / 24), inside(sx + 60 - 40 * j, 0, rect, 8)[0]) for j in range(4)]))
        # and like the rings before it, but straight: three lines of ink down either side
        # of the sheet, brushed on (the left side on the first eighth, the right on the
        # second), firing their beads across it one line at a time on the six notes
        # after, the sides by turns - the outermost first; each line's beads a little off
        # the last's, so the lanes between them move, round where the soul is
        ly = aim(bt(.3))[1]
        fires = [bt(f) for f in (.375, .5, .625, .625 + 1 / 24, .625 + 2 / 24, .75)]
        lines = []
        for q in range(6):
            side = -1 if q % 2 == 0 else 1                   # -1: on the left, firing right
            depth = q // 2                                   # (0: the outermost)
            x = rect[0] + 16 + 22 * depth if side < 0 else rect[2] - 16 - 22 * depth
            off = ly + LANE_SP / 2 + LANE_STEP * (q - 2.5)   # (the lane: round the soul, moving down)
            ys = [y for y in off + LANE_SP * np.arange(-9, 10) if rect[1] + 8 <= y <= rect[3] - 8]
            lines.append((x, -side, (bt(.125) if side < 0 else bt(.25)) + .03 * depth, fires[q], ys))
        atts.append(A.InkLanes(rect, lines, seed=5380 + k))
    # the climax: from each round's start through the bar's trill a sea of ink spears
    # rises behind the soul (up the sheet for three rounds, back down it for three -
    # on screen: up, down, right, left, up, down) and comes on after it as fast as it
    # can run; little bubbles of ink are shot at it head-on. On the pairs the page is
    # folded over (choreo.sheet_state), slamming down - ink thrown off its edges
    # (M42's cut is round a spot well away from the soul - CUT_AWAY px further onto the
    # opened-out sheet: it shows as the sheet opens, and the soul has to get into it
    # before the blades come round, then keeps to it - stepping out of the slashes'
    # way and back - so that M43's run sets off from there, the same every plan)
    # (across the sheet towards its middle, and only a little down it: M43-M45 run on
    # down from there, three rounds of it)
    soul0 = np.array(aim(OPEN_T + .1), float)
    sx_ = 1.0 if rect[0] + FULL / 2 >= soul0[0] else -1.0
    cc = np.clip(soul0 + [sx_ * CUT_AWAY[0], CUT_AWAY[1]], [rect[0] + 90, rect[1] + 90],
                 [rect[0] + FULL - 90, rect[1] + 135])
    GOALS[:] = _GOALS0 + [(OPEN_T + .1, REFOLD_T, (float(cc[0]), float(cc[1])))]
    line_end = None                                  # (the second wave: where the last round's line ended)
    LINES.clear()
    for k, m in enumerate(CORE_M):
        trill = core_trill(m)
        t_show, hits = core_run(k)
        sy = aim(t_show)[1]
        way = RUN_WAY[k]                             # -1: it rises from the bottom, driving the soul up
        lo_f, hi_f = rect[1] + 52, rect[3] - 52
        gates, darts, dart_aim = [], [], aim
        if k >= 3:
            # the second wave: gates thrust in ahead of it on the pairs and the trill's
            # eighths (both sides at once, one gap - to the left and the right of its
            # way by turns, a gate every GATE_DY px, as many as it gets through), and on
            # the trill's notes the front spear's point snaps off and flies at it. All of
            # it laid out along one line it can run (flat out, weaving through the gaps)
            # from where the last round's line ended - not from the last plan, which
            # would shift it about from one pass to the next (plan.py)
            P = cc.copy() if line_end is None else line_end
            x_s = float(np.clip(P[0], rect[0] + 50, rect[2] - 50))
            sy = float(P[1])
            rng_g = np.random.default_rng(5180 + k)
            sgn = rng_g.choice((-1, 1))
            eighths = [t for t in (tm(m - .25 + j / 8) for j in range(6))
                       if t >= t_show - 1e-3 and any(abs(t - n) < 1e-3 for n in _CHART_T)]
            way_pts = [(t_show, x_s)]                    # (t, x) the line goes through: the gaps, as it gets there
            for tg in eighths:
                j = len(gates)
                ahead = GATE_AT + GATE_DY * j
                t_pass = t_show + ahead / 118
                if t_pass > hits[-1] + .08:              # (it runs on a moment past the last note)
                    break
                if tg > t_pass - .2:
                    continue
                off = sgn * 8 if j == 0 else sgn * (-1) ** j * 16    # (the first: just off where it starts; then
                                                                    # the other side, and back)
                gx = float(np.clip(x_s + off + rng_g.uniform(-2, 2), rect[0] + 30, rect[2] - 30))
                gates.append((tg, sy + way * ahead, gx))
                way_pts.append((t_pass, gx))
            tw, xw = zip(*way_pts)
            uw = [118 * (t - t_show) for t in tw]        # (how far along it each is)

            def dart_aim(t, tw=tw, xw=xw, sy=sy, way=way, t_show=t_show):
                return float(np.interp(t, tw, xw)), sy + way * 118 * (t - t_show)

            def line_x(y, uw=uw, xw=xw, sy=sy, way=way):
                return float(np.interp(way * (y - sy), uw, xw))

            line_end = np.array(dart_aim(hits[-1] + .1))   # (where it stops: RUNS)
            darts = list(trill[:-1])                     # (the last would come after it has stopped)
            LINES.append((t_show, hits[-1] + .1, line_x))
        fronts = [float(np.clip(sy - way * 30 + way * 118 * (t - t_show), lo_f, hi_f)) for t in hits]
        sea = A.SpikeSea(rect, t_show, hits, fronts, way, hits[-1] + .1, seed=5100 + k, gates=gates, darts=darts,
                         aim=dart_aim)
        atts.append(sea)
        # bubbles: on every sixteenth from the sea's showing a pair wells up on the edge
        # the soul is running for and fires at it head-on - one at where it will be by
        # then, one well away from that. (The second wave: the gates are in their way -
        # they break on them, all but those that come through a gap - and the one at
        # the soul comes close by it, not at it: it is in a gap, with no room to step aside)
        s16 = MEAS / 16
        rng_b = np.random.default_rng(5160 + k)
        y_in = rect[1] + 10 if way < 0 else rect[3] - 10
        rows = []
        for r in range(1, int((trill[-1] - t_show) / s16) + 1):
            tf = t_show + r * s16
            meet = tf + abs(y_in - dart_aim(tf)[1]) / (A.InkBubbles.V + 118)
            xa = dart_aim(meet)[0] + (rng_b.uniform(-6, 6) if k < 3 else (-1) ** r * 20 + rng_b.uniform(-3, 3))
            xa = float(np.clip(xa, rect[0] + 16, rect[2] - 16))
            xs = [x for x in np.linspace(rect[0] + 16, rect[2] - 16, 40) if abs(x - xa) > 70]
            rows.append((tf, [xa, float(rng_b.choice(xs))]))
        atts.append(A.InkBubbles(rect, -way, rows, trill[-1] + .2, sea=sea, seed=5150 + k))
    # M42, between the rounds: the sheet is cut down round the soul - on the first
    # four notes a blade goes round it in a circle, smaller each time, and all of
    # the sheet outside drops away into the dark - and all the while it is slashed
    # through, every which way, inside what is left (the slide: one long sweep). On
    # the first pair it opens out whole again; on the second it is folded over and
    # the next run starts
    cut_ts = [tm(CUT_M + f) for f in (.125, .25, .375, .5)]
    full = (rect[0], rect[1], rect[0] + FULL, rect[3])          # (the whole sheet, opened out)
    atts.append(A.FoldFlap(OPEN_T - A.FoldFlap.DUR, rect[2], FULL - (rect[2] - rect[0]), rect[1], rect[3], unfold=True))
    atts.append(A.FoldFlap(REFOLD_T - A.FoldFlap.DUR, rect[2], FULL - (rect[2] - rect[0]), rect[1], rect[3]))
    atts.append(A.FoldFlap(OPEN2_T - A.FoldFlap.DUR, rect[2], FULL - (rect[2] - rect[0]), rect[1], rect[3], unfold=True))
    rng_c = np.random.default_rng(4242)
    # the drum roll in 32nds (41.875-42.125): on each a burst of the sheet outside the
    # circle is cut to scraps and thrown off - in no order round it, most of the
    # sheet gone before the circle's first cut
    bursts = [tm(41.875 + j / 32) for j in range(8)]
    # the warning: in the middle of what is going to go (upright on screen)
    M_, b_, _ = sheet_state(OPEN_T + .01)
    ox_, oy_ = arena_offset(OPEN_T + .01)
    to_scr = lambda p: M_ @ (np.asarray(p, float) + [ox_, oy_]) + b_
    to_loc = lambda q: M_.T @ (np.asarray(q, float) - b_) - [ox_, oy_]
    c_s, s_s = to_scr(cc), to_scr(soul0)               # (beyond the soul, if it fits clear of it there)
    w_s = np.clip(s_s + (s_s - c_s) / max(1e-6, float(np.linalg.norm(s_s - c_s))) * 100, [150, 130], [490, 350])
    if np.linalg.norm(w_s - s_s) < 80:
        w_s = np.clip(np.array([640, 480]) - c_s, [150, 130], [490, 350])
    sign = [to_loc(w_s + v) for v in ((0, -70), (78, 62), (-78, 62))]
    bang = [to_loc(w_s + v) for v in ((0, -26), (0, 22))], to_loc(w_s + (0, 42))
    slashes = []
    notes42 = [tm(CUT_M + f) for f in (.125, .25, .375, .5, .5 + 1 / 24, .5 + 2 / 24, .625)]
    for j, th in enumerate(notes42):
        for _ in range(3 if j >= 4 else 2):
            slashes.append((th, rng_c.uniform(0, math.pi), rng_c.uniform(16, 52) * rng_c.choice((-1, 1)),
                            rng_c.uniform(6, 10), rng_c.uniform(-.5, .5)))
    slashes.append((tm(CUT_M + .5), None, None, 12, EIGHTH))   # (the slide: one long sweep round)
    for j in range(1, 13):                                   # the drums rattling in 48ths under the 24ths: short ones
        slashes.append((tm(CUT_M + .375 + j / 48), rng_c.uniform(0, math.pi), rng_c.uniform(18, 40) * rng_c.choice((-1, 1)),
                        rng_c.uniform(5, 8), rng_c.uniform(-.6, .6), rng_c.uniform(46, 84)))
    atts.append(A.CircleCut(full, cc, list(zip(cut_ts, (170, 128, 98, 76))), slashes, UNFOLD_T,
                            lambda t: aim(t), bursts=bursts, warn=(OPEN_T, sign, bang), seed=4243))
    for j, t in enumerate(FOLD_LANDS + [BREAK_T]):          # (the break: where M38's flight lands)
        atts.append(A.FoldBurst(t, rect, seed=5170 + j))
    tl.attacks.append(A.ArenaFrame(atts, arena_offset))
    # streaks of ink rushing up over the paper the whole way: it is still going down
    # (in the world, not the arena's frame - they smear as it whips about under them)
    tl.attacks.append(A.SpeedInk(RISE[0], NEW_END, lambda t: (arena_offset(t)[0] + (JUMP if t >= SHIFT_T and not LOCAL else 0),
                                                             arena_offset(t)[1]), seed=4471))
