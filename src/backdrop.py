"""Background mood layer: ink and painting imagery behind Ink Sans.

Used in the chart's resting passages instead of attacks. Every element is cued
by a note: holds paint giant dry-brush strokes, slides paint the slide's path
huge across the screen, taps flash an ink/painting object (ink bottle, palette,
brush, splat) blooming out of a spreading ink cloud. All luma stays low so the
battle stays readable; everything is drawn on a 2-px grain like the sprites.
"""

from __future__ import annotations

import bisect
import math

import numpy as np
from PIL import Image, ImageDraw

import choreo as C
from chart import parse_chart
from common import ROOT, W, H, clamp, key_vec, out_cubic, smooth, sprite

HW, HH = W // 2, H // 2
SCREEN_C = (320.0, 236.0)
HUD_TOP = 394


def _noise(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros((HH, HW))
    amp, total = 1.0, 0.0
    for cell in (40, 20, 10, 5):
        g = rng.random((HH // cell + 2, HW // cell + 2))
        im = Image.fromarray((g * 255).astype(np.uint8)).resize(
            ((HW // cell + 2) * cell, (HH // cell + 2) * cell), Image.Resampling.BICUBIC)
        out += amp * np.asarray(im, float)[:HH, :HW] / 255
        total += amp
        amp *= .55
    return out / total


NOISE = _noise(77)
_YY, _XX = np.mgrid[0:HH, 0:HW]


class Canvas:
    def __init__(self):
        self.img = Image.new("L", (HW, HH))
        self.d = ImageDraw.Draw(self.img)
        self.extra = []          # numpy layers to max-combine (half res)

    def line(self, pts, w, v):
        self.d.line([(x / 2, y / 2) for x, y in pts], fill=int(v), width=max(1, int(w / 2)))

    def poly(self, pts, v, outline=None):
        self.d.polygon([(x / 2, y / 2) for x, y in pts], fill=int(v),
                       outline=None if outline is None else int(outline))

    def ellipse(self, cx, cy, rx, ry, v, outline=None, width=1):
        self.d.ellipse((cx / 2 - rx / 2, cy / 2 - ry / 2, cx / 2 + rx / 2, cy / 2 + ry / 2),
                       fill=None if v is None else int(v),
                       outline=None if outline is None else int(outline), width=width)


# ------------------------------------------------------------------ elements
class DryStroke:
    """A giant dry-brush stroke: separate bristle streaks with ragged ends."""

    def __init__(self, path, ta, tb, width, luma, seed, hold=.5, fade=1.2):
        pts = np.array(path, float)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.concatenate([[0], np.cumsum(seg)])
        n = max(2, int(cum[-1] / 6))
        s = np.linspace(0, cum[-1], n)
        self.p = np.stack([np.interp(s, cum, pts[:, 0]), np.interp(s, cum, pts[:, 1])], 1)
        d = np.gradient(self.p, axis=0)
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        self.nrm = np.stack([-d[:, 1], d[:, 0]], 1)
        rng = np.random.default_rng(seed)
        nb = 11
        self.off = np.linspace(-.5, .5, nb) * width + rng.uniform(-2, 2, nb)
        self.bw = rng.choice([2, 4, 4, 6], nb)
        self.start = rng.uniform(0, .12, nb)
        self.stop = rng.uniform(.75, 1.0, nb)
        self.gaps = rng.random((nb, n)) < .06
        self.lum = rng.uniform(.6, 1.0, nb) * luma
        self.ta, self.tb, self.hold, self.fade = ta, tb, hold, fade
        self.t0, self.t1 = ta, tb + hold + fade

    def draw(self, cv: Canvas, t):
        u = smooth((t - self.ta) / max(1e-6, self.tb - self.ta))
        g = 1 - smooth((t - self.tb - self.hold) / self.fade)
        n = len(self.p)
        head = int(u * (n - 1))
        for b in range(len(self.off)):
            i0 = int(self.start[b] * n)
            i1 = min(head, int(self.stop[b] * n))
            if i1 <= i0:
                continue
            pts = self.p[i0:i1 + 1] + self.nrm[i0:i1 + 1] * self.off[b]
            # break the bristle where it ran dry
            run = [pts[0]]
            for k in range(1, len(pts)):
                if self.gaps[b, i0 + k]:
                    if len(run) > 1:
                        cv.line(run, self.bw[b], self.lum[b] * g)
                    run = [pts[k]]
                else:
                    run.append(pts[k])
            if len(run) > 1:
                cv.line(run, self.bw[b], self.lum[b] * g)


class Bloom:
    """Ink dropped in water: a cloud spreading through fixed noise."""

    def __init__(self, center, t0, radius, luma, life=1.4, grow=.55):
        self.c = center
        self.r = radius
        self.luma = luma
        self.ts, self.life, self.grow = t0, life, grow
        self.t0, self.t1 = t0, t0 + life

    def draw(self, cv: Canvas, t):
        age = t - self.ts
        R = self.r * out_cubic(age / self.grow) / 2 + 1
        g = 1 - smooth((age - self.life * .45) / (self.life * .55))
        cx, cy = self.c[0] / 2, self.c[1] / 2
        x0, x1 = int(max(0, cx - R * 1.5)), int(min(HW, cx + R * 1.5))
        y0, y1 = int(max(0, cy - R * 1.5)), int(min(HH, cy + R * 1.5))
        if x0 >= x1 or y0 >= y1:
            return
        d = np.hypot(_XX[y0:y1, x0:x1] - cx, _YY[y0:y1, x0:x1] - cy) / R
        f = NOISE[y0:y1, x0:x1] * .9 + (1 - d)
        layer = np.zeros((HH, HW), np.uint8)
        # thin bright tide-line at the wet edge, faint wash inside, grain in the wash
        wash = (f > .64) & (NOISE[y0:y1, x0:x1] > .47)
        v = np.where(wash, self.luma * .35, 0.0)
        rim = (f > .64) & (f < .69)
        v = np.where(rim, self.luma * 1.3, v) * g
        layer[y0:y1, x0:x1] = np.clip(v, 0, 255).astype(np.uint8)
        cv.extra.append(layer)


class Ripple:
    """A drop of ink falls into the dark ink sea behind the arena: a small crown
    splash, then perspective rings spreading on the surface."""

    def __init__(self, center, t_hit, luma=90, size=1.0, seed=0):
        self.c = np.array(center, float)
        self.th, self.luma, self.size = t_hit, luma, size
        rng = np.random.default_rng(seed)
        self.crown = rng.uniform(-1, 1, (7, 2))
        self.t0, self.t1 = t_hit - .34, t_hit + 1.8

    def draw(self, cv: Canvas, t):
        cx, cy = self.c
        tau = t - self.th
        if tau < 0:                                   # the falling drop
            u = (tau + .34) / .34
            y = cy - 240 * (1 - u * u)
            cv.line([(cx, y - 18 * u), (cx, y)], 2, self.luma * .6)
            cv.ellipse(cx, y, 5, 7, self.luma)
            return
        s = self.size
        for k in range(3):                            # rings on the surface
            a = tau - k * .16
            if a <= 0:
                continue
            r = (14 + 150 * (1 - math.exp(-a * 2.2))) * s
            g = (1 - smooth(a / 1.5)) * (1 - .25 * k)
            cv.ellipse(cx, cy, 2 * r, .56 * r, None, outline=self.luma * g, width=1)
        if tau < .45:                                 # crown splash
            for (dx, dy) in self.crown:
                px = cx + dx * 40 * tau * s * 2.2
                py = cy - (150 + 60 * dy) * tau + 420 * tau * tau
                if py < cy + 2:
                    cv.ellipse(px, py, 4, 4, self.luma * (1 - tau / .45))


class Motes:
    """A depth field of ink particles streaming out of the arena towards the
    camera, as if we were travelling into the canvas."""

    def __init__(self, t0, t1, n=70, seed=3, vp=(320.0, 318.0), speed=1.1):
        rng = np.random.default_rng(seed)
        self.ang = rng.uniform(0, math.tau, n)
        self.z0 = rng.uniform(0, 1, n)
        self.l = rng.uniform(50, 110, n)
        self.vp, self.speed = vp, speed
        self.t0, self.t1 = t0, t1

    def pos(self, t):
        z = 1 - ((self.z0 + (t - self.t0) * self.speed * .45) % 1)    # 1 far -> 0 near
        r = 18 / (z + .06)
        x = self.vp[0] + np.cos(self.ang) * r
        y = self.vp[1] + np.sin(self.ang) * r * .8
        return x, y, z

    def draw(self, cv: Canvas, t):
        g = smooth((t - self.t0) / .8) * (1 - smooth((t - self.t1 + .5) / .5))
        x1, y1, z = self.pos(t)
        x0, y0, _ = self.pos(t - 1 / 30)
        for k in range(len(z)):
            if z[k] > .97 or abs(x1[k] - x0[k]) > 200:
                continue
            near = 1 - z[k]
            v = self.l[k] * min(1, near * 1.4) * g
            w = 2 if near < .7 else 4
            cv.line([(x0[k], y0[k]), (x1[k], y1[k])], w, v)


class StarField:
    """Stars (sprite/star.png, shrunk) streaming out of the arena towards the
    camera - the fight now feels like travelling through the canvas."""

    HALF_SIZES = (4, 6, 8, 10, 13, 16)       # half-res sizes -> 8..32 px on screen
    ANGLES = 12

    def __init__(self, t0, t1, n=26, seed=11, vp=(320.0, 318.0), speed=.55, flows=None):
        from common import Sprite
        rng = np.random.default_rng(seed)
        self.t0, self.t1, self.vp, self.speed = t0, t1, vp, speed
        self.ang = rng.uniform(0, math.tau, n)
        self.z0 = rng.uniform(0, 1, n)
        # flow mode: stars stream across the screen in a direction that changes
        # with every phrase [(t_start, angle_deg, px/s)]; depth sets size & speed
        self.flows = flows
        self.fx0 = rng.uniform(0, W + 80, n)
        self.fy0 = rng.uniform(0, H + 80, n)
        self.depth = rng.uniform(.25, 1.0, n)
        self.spin = rng.uniform(-2.5, 2.5, n)
        self.rot0 = rng.uniform(0, math.tau, n)
        src = Image.open(ROOT / "sprite" / "star.png").convert("RGBA")
        self.sprites = []
        for hs in self.HALF_SIZES:
            row = []
            for k in range(self.ANGLES):
                rot = src.rotate(k * 360 / self.ANGLES / 5, resample=Image.Resampling.BICUBIC, expand=True)
                small = rot.resize((hs, hs), Image.Resampling.LANCZOS)
                a = np.asarray(small.getchannel("A")) > 110
                v = np.asarray(small.convert("L")).astype(float)
                v = (np.round(v / 255 * 3) / 3 * 200 + 40).astype(np.uint8)     # 4 grey steps
                v, a = v.repeat(2, 0).repeat(2, 1), a.repeat(2, 0).repeat(2, 1)
                row.append(Sprite(v, a))
            self.sprites.append(row)

    def draw_full(self, frame: np.ndarray, t: float):
        from common import paste
        g = smooth((t - self.t0) / .6) * (1 - smooth((t - self.t1 + .6) / .6))
        if g <= 0:
            return
        if self.flows:
            return self._draw_flow(frame, t, g, paste)
        z = 1 - ((self.z0 + (t - self.t0) * self.speed) % 1)       # 1 far -> 0 near
        order = np.argsort(-z)
        for k in order:
            zk = z[k]
            if zk > .96:
                continue
            r = 30 / (zk + .09)
            x = self.vp[0] + math.cos(self.ang[k]) * r
            y = self.vp[1] + math.sin(self.ang[k]) * r * .78
            near = 1 - zk
            si = min(len(self.HALF_SIZES) - 1, int(near ** 1.6 * len(self.HALF_SIZES)))
            ai = int(((self.rot0[k] + self.spin[k] * t) / math.tau * self.ANGLES)) % self.ANGLES
            spr = self.sprites[si][ai]
            paste(frame, spr, x - spr.w / 2, y - spr.h / 2, gain=g * (.25 + .45 * near))

    def _offset(self, t):
        """Distance travelled (x, y) by a depth-1 star, integrated over the phrases."""
        ox = oy = 0.0
        for i, (ts, ang, sp) in enumerate(self.flows):
            te = self.flows[i + 1][0] if i + 1 < len(self.flows) else 1e9
            dt = max(0.0, min(t, te) - ts)
            ox += math.cos(math.radians(ang)) * sp * dt
            oy -= math.sin(math.radians(ang)) * sp * dt
        return ox, oy

    def _draw_flow(self, frame, t, g, paste):
        ox, oy = self._offset(t)
        Wp, Hp = W + 80, H + 80
        for k in np.argsort(self.depth):
            d = self.depth[k]
            x = (self.fx0[k] + ox * d) % Wp - 40
            y = (self.fy0[k] + oy * d) % Hp - 40
            si = min(len(self.HALF_SIZES) - 1, int(d ** 1.4 * len(self.HALF_SIZES)))
            ai = int(((self.rot0[k] + self.spin[k] * t) / math.tau * self.ANGLES)) % self.ANGLES
            spr = self.sprites[si][ai]
            paste(frame, spr, x - spr.w / 2, y - spr.h / 2, gain=g * (.2 + .5 * d))


class StarRain:
    """The phase-4 stars as in the reference: sprite/star.png at one size, in its
    own flat grey, one set of stars over the whole screen (behind Ink Sans, the
    arena and the HUD), all moving together at one speed V. flows = [(t, dir)],
    dir "left" / "right" / "up" / "down": on each turn every star eases down to
    a stop together by t, they all turn, and ease back up to the same speed (TURN
    s each way). The first flow is already at full speed."""

    TURN = .5
    SIZE = 36                                    # ~80 px at 1080p
    ANGLES = 18                                  # rotation steps over the star's 72 degrees
    WAYS = {"left": (-1.0, 0.0), "right": (1.0, 0.0), "up": (0.0, -1.0), "down": (0.0, 1.0)}

    def __init__(self, t0, t1, flows, V=400.0, n=40, seed=12, windows=None):
        from common import Sprite
        rng = np.random.default_rng(seed)
        self.t0, self.t1, self.V = t0, t1, V
        self.flows = flows
        self.windows = windows or [(t0, t1)]
        # spread evenly (jittered grid) so no part of the sky is empty
        cols = int(math.ceil(math.sqrt(n * (W + 60) / (H + 60))))
        rows = int(math.ceil(n / cols))
        cells = rng.permutation(cols * rows)[:n]
        self.x0 = (cells % cols + rng.uniform(.15, .85, n)) * (W + 60) / cols
        self.y0 = (cells // cols + rng.uniform(.15, .85, n)) * (H + 60) / rows
        self.spin = rng.uniform(-.8, .8, n)
        self.rot0 = rng.uniform(0, math.tau, n)
        src = Image.open(ROOT / "sprite" / "star.png").convert("RGBA")
        self.sprites = []
        for k in range(self.ANGLES):
            rot = src.rotate(k * 72 / self.ANGLES, resample=Image.Resampling.BICUBIC, expand=True)
            small = rot.resize((self.SIZE, self.SIZE), Image.Resampling.LANCZOS)
            a = np.asarray(small.getchannel("A")) > 120
            v = np.asarray(small.convert("L")).astype(np.uint8)
            self.sprites.append(Sprite(v, a))
        # integrate the (eased) velocity once
        self.dt = 1 / 240
        ts = np.arange(t0, t1 + .5, self.dt)
        v = np.array([self._vel(t) for t in ts])
        self.ts = ts
        self.off = np.concatenate([np.zeros((1, 2)), np.cumsum(v[:-1], 0) * self.dt])

    def _vel(self, t):
        fl = self.flows
        i = max(0, bisect.bisect_right([f[0] for f in fl], t) - 1)
        g = 1.0
        if i > 0:                                        # speeding up again after the turn
            g *= smooth((t - fl[i][0]) / self.TURN)
        if i + 1 < len(fl):                              # every star slowing to a stop before the next
            g *= smooth((fl[i + 1][0] - t) / self.TURN)
        dx, dy = self.WAYS[fl[i][1]]
        return (dx * self.V * g, dy * self.V * g)

    def draw_full(self, frame: np.ndarray, t: float):
        from common import paste
        g = max(smooth((t - a) / .3) * (1 - smooth((t - b) / .3)) for a, b in self.windows)
        if g <= 0:
            return
        ox = float(np.interp(t, self.ts, self.off[:, 0]))
        oy = float(np.interp(t, self.ts, self.off[:, 1]))
        for k in range(len(self.x0)):
            x = (self.x0[k] + ox) % (W + 60) - 30
            y = (self.y0[k] + oy) % (H + 60) - 30
            ai = int((self.rot0[k] + self.spin[k] * t) / math.tau * 5 * self.ANGLES) % self.ANGLES
            spr = self.sprites[ai]
            paste(frame, spr, x - spr.w / 2, y - spr.h / 2, gain=g)


# ------------------------------------------------------------------ schedule
def ring_point(k, r=1.0, rx=250, ry=180):
    ux, uy = key_vec(k)
    return (SCREEN_C[0] + ux * rx * r, SCREEN_C[1] + uy * ry * r)


def build():
    notes, _ = parse_chart(ROOT / "maidata.txt", C.OFFSET)
    els = []
    # ---- after the bones shatter (M9 - M12): no attacks, only mood
    a, b = C.tm(9), C.tm(13)
    els.append(Motes(a, b + .3))
    seed = 500
    for n in notes:
        if not (a - 1e-3 <= n.t < b - 1e-3):
            continue
        seed += 1
        if n.kind == "hold":
            p0 = ring_point(n.key, 1.25)
            p1 = ring_point((n.key + 3) % 8 + 1, 1.25)
            els.append(DryStroke([p0, p1], n.t, n.t + n.dur, 90, 46, seed, hold=.3, fade=1.0))
            els.append(Bloom(ring_point(n.key, 1.0), n.t, 120, 34, life=n.dur + .8, grow=n.dur))
        elif n.kind == "slide":
            for s in n.slides:
                for path in C.slide_paths(s.start_key, s.segments):
                    pts = [(SCREEN_C[0] + x * 260, SCREEN_C[1] + y * 190) for x, y in path]
                    last = s.t_start > b - .4          # the break star at the end: a bright wipe
                    els.append(DryStroke(pts, s.t_start, s.t_end, 70 if not last else 120,
                                         52 if not last else 120, seed, hold=.15 if last else .4,
                                         fade=.35 if last else 1.1))
        else:
            ux, _ = key_vec(n.key)
            x = 320 + ux * 210 + (18 if n.key in (2, 3, 6, 7) else -18) * np.sign(ux)
            y = 250 + (40 if n.key in (3, 4, 5, 6) else 0)
            els.append(Bloom((x, y + 6), n.t + .03, 90, 34, life=1.6, grow=.6))
            els.append(Ripple((x, y), n.t, seed=seed))
    els.sort(key=lambda e: e.t0)
    return els


class Backdrop:
    def __init__(self):
        self.els = build()
        # after the title card the battle resumes with the reference's star rain:
        # the first phrase streams sideways (running, everything comes from the
        # right), then vertical. One set of stars, one speed: on every phrase they all slow to a stop
        # together, turn, and speed up again (from M21 the paper spreads over
        # the screen and covers them; they are gone after the burst)
        # the turn to "down" sits on the 16th note at M18.5625 (where the old turn at
        # M19 used to start slowing down), so the whole slow-down/turn comes 0.5 s earlier
        flows = C.STAR_FLOWS                     # (the paper moves with them in the run and climb)
        burst = C.tm(24.75)
        sky = StarRain(C.tm(17), burst + .4, flows, V=400.0, n=40, seed=13, windows=[(C.tm(17), burst)])
        # the descent: they are back as we sink - streaming up past us - and turn
        # with the phrases while the arena slides about
        dflows = [(C.RISE[0], "up"), (C.tm(38.875), "left"), (C.tm(42), "right"), (C.tm(45.875), "up")]
        deep = StarRain(C.RISE[0], C.NEW_END + .4, dflows, V=400.0, n=40, seed=29, windows=[(C.RISE[0], C.NEW_END)])
        self.full = [sky, deep]
        self.t0s = [e.t0 for e in self.els]
        self.maxlen = max((e.t1 - e.t0 for e in self.els), default=0)

    def draw(self, frame: np.ndarray, t: float):
        for f in self.full:
            if f.t0 <= t <= f.t1:
                f.draw_full(frame, t)            # behind Sans, the arena and the HUD
        lo = bisect.bisect_left(self.t0s, t - self.maxlen - .01)
        hi = bisect.bisect_right(self.t0s, t)
        act = [e for e in self.els[lo:hi] if e.t0 <= t <= e.t1]
        if not act:
            return
        cv = Canvas()
        for e in act:
            e.draw(cv, t)
        layer = np.asarray(cv.img)
        for ex in cv.extra:
            layer = np.maximum(layer, ex)
        big = layer.repeat(2, 0).repeat(2, 1)
        big[HUD_TOP:] = 0          # keep the HUD rows clean
        np.maximum(frame, big, out=frame)
