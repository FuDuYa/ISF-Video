"""Ink Sans' big body moves.

Only purposeful, full-body motion: a jump-slam (crouch with the brush raised,
leap, fall, brush driven down into the arena on landing, then a hop back to
his spot). Between moves he stands still with his brush held down.

choreo.build() fills SLAMS (it knows where the soul is); scene.draw_sans asks
pose(t) for the offsets and the body frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from common import clamp, in_quad, out_cubic, smooth

HOME = 320          # Ink Sans' origin x (scene.json)
TIP_DX = 80         # where the brush of body_anime2 enters the arena, relative to his origin


def _even(v: float) -> int:
    return int(round(v / 2.0)) * 2


@dataclass
class Slam:
    crouch: float           # anticipation starts (brush raised over the shoulder)
    off: float              # feet leave the ground
    land: float             # brush hits the arena (on the beat)
    x_land: float = HOME    # origin x where he comes down
    peak: float = 150.0     # jump height (px); large values leave the frame
    apex: float | None = None
    hold: float = .30       # stays in the landing pose
    hop: float = .32        # then hops back to HOME (only if he landed elsewhere)

    def __post_init__(self):
        if self.apex is None:
            self.apex = self.off + (self.land - self.off) * .5
        self.end = self.land + self.hold + (self.hop if abs(self.x_land - HOME) > 1 else 0.0)

    @property
    def tip_x(self) -> float:
        return self.x_land + TIP_DX

    def pose(self, t: float):
        X = self.x_land - HOME
        if t < self.off:                                   # crouch: the body sinks onto the legs
            u = smooth((t - self.crouch) / (self.off - self.crouch))
            s = _even(10 * u)
            return dict(dx=0, dy=0, body="body.png", by=s, hy=s, ly=0)
        if t < self.land:                                  # in the air
            if t < self.apex:
                y = -self.peak * out_cubic((t - self.off) / (self.apex - self.off))
            else:
                y = -self.peak * (1 - in_quad((t - self.apex) / (self.land - self.apex)))
            x = X * smooth((t - self.off) / (self.land - self.off))
            body = "body_anime1.png" if t >= self.land - .05 else "body.png"
            return dict(dx=_even(x), dy=_even(y), body=body, by=0, hy=-2, ly=2)
        tau = t - self.land
        if tau < self.hold:                                # landed: squash, brush planted
            s = _even(12 * (1 - out_cubic(tau / .18))) if tau < .18 else 0
            return dict(dx=_even(X), dy=0, body="body_anime2.png", by=s, hy=s, ly=0)
        u = clamp((tau - self.hold) / self.hop)            # hop back to his spot
        return dict(dx=_even(X * (1 - smooth(u))), dy=_even(-22 * math.sin(math.pi * u)),
                    body="body_anime2.png", by=0, hy=0, ly=0)


SLAMS: list[Slam] = []




# ---------------------------------------------------------------- parries
@dataclass
class Parry:
    """An iaido-like parry for one strike landing at `t`: he sinks and bends with
    the brush held low, draws it across in one cut (the swing frame), and ends
    leaning back with the brush flung up over his shoulder - the flick of ink
    from that cut is the shield (attacks.InkShieldStrikes)."""
    t: float

    def __post_init__(self):
        self.start, self.end = self.t - .16, self.t + .14

    def pose(self, t: float):
        a = t - self.t
        if a < -.07:                                       # crouch, bending forward, brush low
            u = smooth((a + .16) / .09)
            s = _even(10 * u)
            return dict(dx=0, dy=0, body="body_anime2.png", by=s, hy=s, ly=0, rot=9 * u)
        if a < 0:                                          # the draw: one cut across
            return dict(dx=0, dy=0, body="body_anime1.png", by=8, hy=10, ly=0, rot=3.0)
        u = smooth(a / .14)                                 # follow-through: leaning back, brush up
        return dict(dx=0, dy=0, body="body.png", by=_even(2 * (1 - u)), hy=0, ly=0, rot=-7 * (1 - u))


PARRIES: list[Parry] = []

# ---------------------------------------------------------------- the brush blinks away
# (leaves, returns): the brush is gone from his back in between (it is blocking
# strikes in front of him, attacks.InkShieldStrikes); a white flash on each change.
BRUSH_AWAY: list[tuple[float, float]] = []
FLASH = .04


def brush_state(t: float):
    """'gone' while the brush is away, 'flash' just before it leaves / as it returns."""
    for t0, t1 in BRUSH_AWAY:
        if t0 - FLASH <= t < t0 or t1 <= t < t1 + FLASH:
            return "flash"
        if t0 <= t < t1:
            return "gone"
    return None


def pose(t: float):
    """Offsets/body for Ink Sans at t, or None when he is just standing."""
    for s in SLAMS:
        if s.crouch <= t < s.end:
            return s.pose(t)
    for p in PARRIES:
        if p.start <= t < p.end:
            return p.pose(t)
    return None
