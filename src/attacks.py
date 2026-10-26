"""Ink attack primitives.

Every attack is a pure function of time: draw(ctx) paints the visible art and the
hazard mask for ctx.t. Nothing is accumulated between frames, so any frame can be
rendered independently (and in parallel).

Art is drawn on a 320x240 half-resolution canvas and enlarged 2x with nearest
neighbour, so every bullet shares the 2-pixel grain of the Ink Sans sprites.
Coordinates passed to the ctx helpers are in 640x480 world pixels.

Luma conventions (black background):
  255/225   live ink (hazard)
  60..95    warning marks (harmless)
  30..55    dried ink residue (harmless)
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from common import clamp, smooth, out_cubic, in_quad, sprite

HW, HH = 320, 240


class Ctx:
    """Drawing layers for one frame. Normally half resolution (2-px grain, like
    the sprites); `fine` draws at full resolution - used while the camera is
    zoomed deep into the canvas so lines do not turn into big blocks."""

    def __init__(self, t: float, box: tuple[int, int, int, int], fine: bool = False):
        self.t = t
        self.box = box
        self.k = 1 if fine else 2
        size = (640 // self.k, 480 // self.k)
        self.size = size
        self.art = Image.new("L", size)   # the arena / canvas
        self.top = Image.new("L", size)   # free overlay (beams, brushes)
        self.haz = Image.new("L", size)   # hazard mask (clipped later)
        self.sprites = []                 # (Sprite, x, y, gain, mask) pasted at full res on top
        self.ox = 0.0                     # offset added to every primitive (marks riding the moving paper)
        self.oy = 0.0
        self.d = {"art": ImageDraw.Draw(self.art), "top": ImageDraw.Draw(self.top),
                  "haz": ImageDraw.Draw(self.haz)}

    # world -> layer helpers ----------------------------------------------
    def circle(self, layer, x, y, size, fill):
        """Filled disc; `size` is the world-pixel diameter."""
        k = self.k
        r = max(size, 1.0) / (2 * k)
        x, y = (x + self.ox) / k, (y + self.oy) / k
        if r < 1.25:     # tiny discs become solid pixel blocks, not '+' shapes
            s = 0 if r < .75 else 1
            ix, iy = int(x - s / 2), int(y - s / 2)
            self.d[layer].rectangle((ix, iy, ix + s, iy + s), fill=fill)
            return
        self.d[layer].ellipse((x - r, y - r, x + r - .01, y + r - .01), fill=fill)

    def ring(self, layer, x, y, r, width, fill):
        """Circle outline of world radius `r`."""
        k = self.k
        x, y, r = (x + self.ox) / k, (y + self.oy) / k, r / k
        self.d[layer].ellipse((x - r, y - r, x + r, y + r), outline=fill, width=max(1, int(round(width / k))))

    def poly(self, layer, pts, fill):
        k = self.k
        self.d[layer].polygon([((px + self.ox) / k, (py + self.oy) / k) for px, py in pts], fill=fill)

    def line(self, layer, pts, width, fill):
        k = self.k
        self.d[layer].line([((px + self.ox) / k, (py + self.oy) / k) for px, py in pts], fill=fill,
                           width=max(1, int(round(width / k))))

    def rect(self, layer, x0, y0, x1, y1, fill):
        k = self.k
        x0, x1, y0, y1 = x0 + self.ox, x1 + self.ox, y0 + self.oy, y1 + self.oy
        self.d[layer].rectangle((x0 / k, y0 / k, x1 / k - .01, y1 / k - .01), fill=fill)

    def dot(self, layer, x, y, fill):
        x, y = int((x + self.ox) // self.k), int((y + self.oy) // self.k)
        if 0 <= x < self.size[0] and 0 <= y < self.size[1]:
            self.d[layer].point((x, y), fill=fill)

    def paper(self) -> np.ndarray:
        """The paper mask at this layer's resolution (cached per frame)."""
        if getattr(self, "_pm", None) is None:
            import scene as S
            self._pm = S.paper_mask(self.t, self.box)[0][::self.k, ::self.k]
        return self._pm

    def layer(self, name) -> np.ndarray:
        """A layer at full resolution (640x480)."""
        a = np.asarray(getattr(self, name))
        return a.repeat(2, 0).repeat(2, 1) if self.k == 2 else a


PAPER_OFFSET = None      # set by choreo: t -> (dx, dy) px the paper has moved by t (run, climb)


def scroll_d(t, t_mark):
    """How far (dx, dy) a mark left on the paper at t_mark has been carried by t."""
    if PAPER_OFFSET is None:
        return 0.0, 0.0
    (x1, y1), (x0, y0) = PAPER_OFFSET(t), PAPER_OFFSET(t_mark)
    return x1 - x0, y1 - y0


def speed_trail(c, tail, d, length, width=8):
    """A tapered afterimage behind something flying along d from `tail`: a white
    streak over the dark, a pale grey one over the paper (the free layer reads
    inverted there). Only fills pixels nothing else has drawn on."""
    k = c.k
    Wl, Hl = c.size
    img = Image.new("L", (Wl, Hl))
    dr = ImageDraw.Draw(img)
    n = 6
    for i in range(n):
        f0, f1 = i / n, (i + 1) / n
        p0, p1 = tail - d * length * f0, tail - d * length * f1
        dr.line([(p0[0] / k, p0[1] / k), (p1[0] / k, p1[1] / k)], fill=int(255 * (1 - f0) ** 1.3),
                width=max(1, round(width * (1 - .7 * f0) / k)))
    a = np.asarray(img).astype(np.float32) / 255
    on = a > .04
    if not on.any():
        return
    pm = c.paper()
    top = np.asarray(c.top).copy()
    free = on & (top == 0)
    top[free & ~pm] = np.clip(170 * a[free & ~pm], 1, 255).astype(np.uint8)       # white over the dark
    top[free & pm] = np.clip(12 + 46 * a[free & pm], 1, 255).astype(np.uint8)      # pale grey on paper
    c.top.paste(Image.fromarray(top))


def close_progress(t, t0, t1):
    """0 -> 1 from t0 to t1: steady in the middle, easing in and out at the ends -
    a slow, even closing in (the living ink in the rest, and the screen's ink)."""
    u = clamp((t - t0) / max(1e-6, t1 - t0))
    return .45 * u + .55 * smooth(u)


def fade_gray(value: float, gain: float) -> int:
    return int(max(0, min(255, round(value * clamp(gain)))))


def jagged_circle(cx, cy, r, rng_vals, n=14):
    pts = []
    for i in range(n):
        a = i / n * math.tau
        rr = r * (0.78 + 0.44 * rng_vals[i % len(rng_vals)])
        pts.append((cx + math.cos(a) * rr, cy + math.sin(a) * rr))
    return pts


class Attack:
    t0 = 0.0
    t1 = 0.0
    shake = 0.0          # screen-shake amplitude at impact
    impact = None        # time of impact for shake/flash

    def active(self, t):
        return self.t0 <= t <= self.t1

    def draw(self, c: Ctx):
        raise NotImplementedError


# ======================================================================
class Spike(Attack):
    """An ink tendril that stabs inward from the arena edge on the beat."""

    TELE = 0.30

    def __init__(self, th, base, normal, length, width=20, seed=0, big=False):
        self.th, self.base, self.n = th, np.array(base, float), np.array(normal, float)
        self.L, self.wb = length, width * (1.35 if big else 1.0)
        self.big = big
        rng = np.random.default_rng(seed)
        self.jit = rng.uniform(-1, 1, 8)
        self.blot = rng.uniform(0, 1, 14)
        self.t0, self.t1 = th - self.TELE, th + 1.5
        self.impact = th
        self.shake = 3.0 if big else 0.0

    def shape(self, l):
        b, n = self.base, self.n
        p = np.array([-n[1], n[0]])
        w = self.wb / 2
        j = self.jit
        pts = [b + p * w,
               b + p * (w * .55) + n * l * .30 + p * j[0] * 2,
               b + p * (w * .75) + n * l * .42,        # barb
               b + p * (w * .35) + n * l * .52 + p * j[1] * 2,
               b + n * l + p * j[2] * 2,                # tip
               b - p * (w * .35) + n * l * .60 + p * j[3] * 2,
               b - p * (w * .70) + n * l * .48,        # barb
               b - p * (w * .50) + n * l * .34 + p * j[4] * 2,
               b - p * w]
        return [tuple(q) for q in pts]

    def draw(self, c):
        tau = c.t - self.th
        bx, by = self.base
        if tau < 0:
            u = (tau + self.TELE) / self.TELE
            blink = 1.0 if int(c.t * 30) % 2 == 0 or u < .6 else .55
            c.circle("art", bx, by, 4 + 8 * out_cubic(u), fade_gray(95 * blink, 1))
            for k in range(1, 6):
                q = self.base + self.n * self.L * k / 6
                if k / 6 < u + .2:
                    c.circle("art", q[0], q[1], 4, 58)
            return
        if tau < .24:
            ext = out_cubic(tau / .05) * (1 - smooth((tau - .13) / .10))
            l = self.L * ext
            if l > 2:
                pts = self.shape(l)
                c.poly("art", pts, 225)
                inner = self.shape(l * .82)
                mid = np.array(inner)
                cen = self.base + self.n * l * .3
                mid = cen + (mid - cen) * .55
                c.poly("art", [tuple(q) for q in mid], 255)
                if l > 5:
                    c.poly("haz", pts, 255)
        if tau > .10:
            g = 1 - smooth((tau - .25) / 1.2)
            if g > 0:
                c.poly("art", jagged_circle(bx, by, 7 if not self.big else 11, self.blot),
                       fade_gray(26, g))


# ======================================================================
class BrushStroke(Attack):
    """A brush paints a maimai slide path across the arena.

    Each sample of the path is painted when the brush head passes it, stays live
    for `trail` seconds, then dries to dark residue. During the warning window
    the path is shown as a dotted guide, like a slide arrow.
    """

    def __init__(self, paths, t_tele, ts, te, width=12, trail=.28, seed=0, brk=False,
                 brush=True):
        self.ts, self.te, self.tt = ts, te, t_tele
        self.width = width * (1.45 if brk else 1)
        self.trail = trail
        self.brk = brk
        self.show_brush = brush
        rng = np.random.default_rng(seed)
        self.paths = []
        for path in paths:
            pts = np.array(path, float)
            seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            cum = np.concatenate([[0], np.cumsum(seg)])
            total = cum[-1]
            n = max(2, int(total / 3))
            s = np.linspace(0, total, n)
            xs = np.interp(s, cum, pts[:, 0])
            ys = np.interp(s, cum, pts[:, 1])
            samples = np.stack([xs, ys], 1)
            tpaint = ts + (s / max(total, 1e-6)) * (te - ts)
            wv = self.width * (0.82 + 0.36 * rng.random(n))
            wv = np.convolve(wv, np.ones(3) / 3, mode="same")
            drips = rng.random(n) < .07
            driplen = rng.uniform(6, 16, n)
            self.paths.append({"p": samples, "tp": tpaint, "w": wv, "drip": drips,
                               "dl": driplen, "total": total})
        self.t0 = t_tele
        self.t1 = te + trail + 1.8
        self.impact = ts if brk else None
        self.shake = 3.5 if brk else 0.0

    def head(self, t):
        path = self.paths[0]
        u = clamp((t - self.ts) / max(self.te - self.ts, 1e-6))
        idx = u * (len(path["p"]) - 1)
        i = int(idx)
        j = min(i + 1, len(path["p"]) - 1)
        f = idx - i
        pos = path["p"][i] * (1 - f) + path["p"][j] * f
        a = max(0, i - 2)
        b = min(len(path["p"]) - 1, i + 2)
        d = path["p"][b] - path["p"][a]
        if np.linalg.norm(d) < 1e-6:
            d = path["p"][-1] - path["p"][0]
        d = d / (np.linalg.norm(d) + 1e-9)
        return pos, d

    def draw(self, c):
        t = c.t
        for path in self.paths:
            p, tp, wv = path["p"], path["tp"], path["w"]
            age = t - tp
            n = len(p)
            # warning guide (dots) for unpainted samples
            if t < self.te:
                g = smooth((t - self.tt) / .12)
                blink = .65 + .35 * (int(t * 24) % 2)
                for i in range(0, n, 4):
                    if age[i] < 0:
                        c.circle("art", p[i][0], p[i][1], 4, fade_gray(84 * blink, g))
            # residue first, then live ink on top
            for i in range(n - 1):
                a = age[i]
                if a < 0:
                    continue
                if a >= self.trail:
                    dry = a - self.trail
                    if dry < .14:
                        val = 205 - (205 - 26) * dry / .14
                    else:
                        val = 26 * (1 - smooth((dry - .14) / 1.3))
                    if val < 3:
                        continue
                    w = wv[i] * (1.0 if dry < .14 else .8)
                    c.line("art", [p[i], p[i + 1]], w, int(val))
                    if path["drip"][i] and dry > .05:
                        dl = path["dl"][i] * smooth(dry / .9)
                        c.line("art", [(p[i][0], p[i][1]), (p[i][0], p[i][1] + dl)], 2,
                               int(val * .9))
                        c.circle("art", p[i][0], p[i][1] + dl, 4, int(val * .9))
            for i in range(n - 1):
                a = age[i]
                if 0 <= a < self.trail:
                    w = wv[i]
                    c.line("art", [p[i], p[i + 1]], w, 215)
                    c.circle("art", p[i][0], p[i][1], w, 215)
                    c.line("art", [p[i], p[i + 1]], max(2, w * .45), 255)
                    c.line("haz", [p[i], p[i + 1]], w, 255)
                    c.circle("haz", p[i][0], p[i][1], w, 255)
        if self.show_brush and self.ts - .18 <= t <= self.te + .12:
            pos, d = self.head(t)
            g = smooth((t - (self.ts - .18)) / .12) * (1 - smooth((t - self.te) / .12))
            draw_brush(c, pos, d, g)


def draw_brush(c: Ctx, tip, direction, gain=1.0, scale=1.0):
    """A small floating pixel paintbrush whose bristle tip sits on `tip`."""
    if gain <= .02:
        return
    d = np.array(direction, float)
    # The handle trails behind the motion, tilted upward so it reads as held.
    back = -d + np.array([0.35, -0.9])
    back /= np.linalg.norm(back) + 1e-9
    side = np.array([-back[1], back[0]])
    tip = np.array(tip, float)
    s = scale

    def P(a, b):
        q = tip + back * a * s + side * b * s
        return (q[0], q[1])

    g = lambda v: fade_gray(v, gain)
    # bristle tuft: dark ink-soaked body with a white wet tip
    c.poly("top", [P(-2, 0), P(5, -6), P(12, -8), P(16, -6), P(16, 6), P(12, 8), P(5, 6)], g(60))
    c.poly("top", [P(-2, 0), P(4, -4), P(9, -4), P(9, 4), P(4, 4)], g(250))
    c.poly("top", [P(9, -5), P(14, -6), P(14, -2), P(9, -1)], g(120))
    c.poly("top", [P(16, -7), P(23, -7), P(23, 7), P(16, 7)], g(200))                  # ferrule
    c.poly("top", [P(18, -7), P(20, -7), P(20, 7), P(18, 7)], g(120))
    c.poly("top", [P(23, -5), P(54, -4), P(56, 0), P(54, 4), P(23, 5)], g(90))          # handle
    c.poly("top", [P(24, -2), P(52, -1), P(52, 1), P(24, 2)], g(150))
    c.poly("top", [P(54, -4), P(60, -4), P(60, 4), P(54, 4)], g(235))                   # end cap


# ======================================================================
class Beam(Attack):
    """An ink geyser: a well of ink opens outside the arena and pours a straight beam."""

    def __init__(self, origin, direction, t_on, t_off, width=26, tele=.45, seed=0,
                 brk=False, length=900):
        self.O = np.array(origin, float)
        self.length = length
        d = np.array(direction, float)
        self.d = d / np.linalg.norm(d)
        self.t_on, self.t_off = t_on, max(t_off, t_on + .12)
        self.width = width * (1.25 if brk else 1)
        self.tele = tele
        rng = np.random.default_rng(seed)
        self.blot = rng.uniform(0, 1, 16)
        self.flow = rng.uniform(0, 1, (24, 3))
        self.t0, self.t1 = t_on - tele, self.t_off + .45
        self.impact = t_on
        self.shake = 4.0 if brk else 2.0

    def width_at(self, t):
        if t < self.t_on:
            return 0.0
        w = self.width * out_cubic((t - self.t_on) / .06)
        w *= 1 - smooth((t - self.t_off) / .16)
        w *= 1 + .07 * math.sin(t * 47)
        return w

    def draw(self, c):
        t = c.t
        O, d = self.O, self.d
        p = np.array([-d[1], d[0]])
        # ink well
        grow = out_cubic((t - self.t0) / (self.tele * .8))
        shrink = 1 - smooth((t - self.t_off - .1) / .3)
        r = 17 * grow * shrink
        recoil = -d * 5 * math.exp(-max(0, t - self.t_on) * 10) if t >= self.t_on else 0
        Oc = O + recoil
        if r > 1:
            c.poly("top", jagged_circle(Oc[0], Oc[1], r * 1.25, self.blot, 16), 96)
            c.poly("top", jagged_circle(Oc[0], Oc[1], r, self.blot[::-1], 16), 26)
            if t >= self.t_on - .1:
                c.circle("top", Oc[0], Oc[1], r * 1.6, 200)
                c.circle("top", Oc[0], Oc[1], r * .9, 255)
        # warning
        if t < self.t_on:
            on = int(t * 20) % 2 == 0
            g = smooth((t - self.t0) / .1)
            for k in range(2, int(self.length / 12)):
                q = O + d * k * 12
                c.circle("top", q[0], q[1], 4, fade_gray(80 if on else 50, g))
            return
        w = self.width_at(t)
        if w < 1:
            return
        L = self.length
        far = O + d * L
        c.line("top", [Oc, far], w, 215)
        c.line("top", [Oc, far], max(2, w * .55), 255)
        # ragged splashing end
        for i in range(5):
            a = self.blot[i] * math.tau + t * 9
            q = far + d * (4 + 8 * self.blot[i + 5]) + p * (self.blot[i + 10] - .5) * w * 1.2
            c.circle("top", q[0], q[1], 4 + 6 * self.blot[i], 215)
        # flowing droplets along the edges give the beam a liquid surface
        for i, (u, side, sz) in enumerate(self.flow):
            s = (u * (L - r) + (t - self.t_on) * 700) % (L - r)
            q = Oc + d * (s + r) + p * (1 if side > .5 else -1) * (w / 2 + 1 + sz * 3)
            c.circle("top", q[0], q[1], 3 + 3 * sz, 215)
        if w > 4 and t <= self.t_off + .1:
            c.line("haz", [Oc + d * r * .5, far], w, 255)


# ======================================================================
class Drip(Attack):
    """A drop swells at the arena ceiling, falls and splashes into droplets."""

    def __init__(self, x, t_land, y_top, y_bot, fall=.42, r=7, seed=0, big=False):
        self.x, self.tl, self.yt, self.yb = x, t_land, y_top, y_bot
        self.fall, self.r = fall, r * (1.4 if big else 1)
        self.tr = t_land - fall
        rng = np.random.default_rng(seed)
        n = 6 if big else 4
        ang = np.linspace(-2.6, -0.54, n) + rng.uniform(-.15, .15, n)
        spd = rng.uniform(110, 190, n) * (1.3 if big else 1)
        self.drops = [(math.cos(a) * s, math.sin(a) * s) for a, s in zip(ang, spd)]
        self.blot = rng.uniform(0, 1, 14)
        self.t0, self.t1 = self.tr - .30, t_land + 2.2
        self.impact = t_land
        self.shake = 2.5 if big else 0.0

    def draw(self, c):
        t, x, r = c.t, self.x, self.r
        if t < self.tr:
            u = (t - self.t0) / .30
            rr = r * out_cubic(u)
            c.circle("art", x, self.yt, rr * 2.8, fade_gray(55 + 20 * u, 1))
            c.circle("art", x, self.yt + rr * .5, rr * 2, fade_gray(90 + 120 * u, 1))
            return
        if t < self.tl:
            u = in_quad((t - self.tr) / self.fall)
            y = self.yt + (self.yb - self.yt - r) * u
            tail = 6 + 16 * u
            c.poly("art", [(x - r * .9, y - r * .2), (x, y - r - tail), (x + r * .9, y - r * .2)], 200)
            c.circle("art", x, y, r * 2, 235)
            c.poly("art", [(x - r * .45, y), (x, y - r - tail * .6), (x + r * .45, y)], 255)
            c.circle("haz", x, y, r * 2, 255)
            c.poly("haz", [(x - r * .6, y), (x, y - r - tail * .6), (x + r * .6, y)], 255)
            return
        tau = t - self.tl
        # puddle residue
        g = 1 - smooth((tau - .2) / 1.9)
        c.poly("art", [(px, self.yb - 3 + (py - self.yb) * .35) for px, py in
                       jagged_circle(x, self.yb, r * 2.0, self.blot)], fade_gray(28, g))
        if tau < .08:
            c.circle("art", x, self.yb - 4, r * 3 * (1 - tau / .08) + 2, 255)
        for vx, vy in self.drops:
            px = x + vx * tau
            py = self.yb - 5 + vy * tau + 520 * tau * tau
            if py > self.yb + 2 or tau > .7:
                continue
            c.circle("art", px, py, 6, 235)
            c.circle("haz", px, py, 6, 255)


# ======================================================================
class Dart(Attack):
    """An ink blob fired in a straight line, with a short trailing tail."""

    def __init__(self, start, vel, t_launch, r=7, seed=0, life=2.5, tele=.25):
        self.S, self.V = np.array(start, float), np.array(vel, float)
        self.tlch, self.r = t_launch, r
        self.tele = tele
        self.t0, self.t1 = t_launch - tele, t_launch + life
        rng = np.random.default_rng(seed)
        self.blot = rng.uniform(0, 1, 12)

    def draw(self, c):
        t = c.t
        if t < self.tlch:
            u = (t - self.t0) / self.tele
            on = int(t * 30) % 2 == 0
            c.poly("art", jagged_circle(self.S[0], self.S[1], 4 + 7 * out_cubic(u), self.blot),
                   80 if on else 60)
            return
        tau = t - self.tlch
        pos = self.S + self.V * tau
        sp = np.linalg.norm(self.V)
        dirv = self.V / (sp + 1e-9)
        for k, (rr, val) in enumerate([(.45, 120), (.6, 165), (.8, 205)]):
            q = pos - dirv * self.r * (3.2 - k) * 1.1
            c.circle("art", q[0], q[1], self.r * 2 * rr, val)
        c.circle("art", pos[0], pos[1], self.r * 2, 230)
        c.circle("art", pos[0] - self.r * .3, pos[1] - self.r * .3, self.r * .8, 255)
        c.circle("haz", pos[0], pos[1], self.r * 2, 255)
        if tau < .1:
            c.poly("art", jagged_circle(self.S[0], self.S[1], 10 * (1 - tau / .1) + 2, self.blot),
                   200)


# ======================================================================
class SplashRing(Attack):
    """Ink gathers into a point, then bursts into a ring of droplets."""

    def __init__(self, center, t, n=10, speed=150, r=5, seed=0, tele=.30, phase=0.0,
                 life=1.4, big=False, harmless=False, scroll=False):
        self.C = np.array(center, float)
        self.th, self.n, self.speed, self.r = t, n, speed, r
        self.tele, self.phase, self.life = tele, phase, life
        rng = np.random.default_rng(seed)
        self.blot = rng.uniform(0, 1, 16)
        self.jit = rng.uniform(-.08, .08, n)
        self.t0, self.t1 = t - tele, t + life + 1.5
        self.impact = t
        self.shake = 5.0 if big else 3.0
        self.big = big
        self.harmless = harmless          # decoration only: pale, no hazard
        self.scroll = scroll              # (harmless only) carried off by the paper as it slides

    def draw(self, c):
        if not (self.scroll and self.harmless):
            return self._draw(c)
        c.ox, c.oy = scroll_d(c.t, self.th)
        try:
            return self._draw(c)
        finally:
            c.ox = c.oy = 0.0

    def _draw(self, c):
        t = c.t
        cx, cy = self.C
        tau = t - self.th
        if tau < 0:
            u = (tau + self.tele) / self.tele
            rad = 46 * (1 - out_cubic(u)) + 4
            for i in range(self.n):
                a = self.phase + i / self.n * math.tau + .6 * (1 - u)
                c.circle("art", cx + math.cos(a) * rad, cy + math.sin(a) * rad, 4,
                         fade_gray(60 + 60 * u, 1))
            c.circle("art", cx, cy, 6 + 10 * u, fade_gray(90 + 100 * u, 1))
            return
        g = 1 - smooth((tau - .3) / 1.5)
        hv = 105 if self.harmless else 235
        c.poly("art", jagged_circle(cx, cy, 16 if self.big else 12, self.blot), fade_gray(28, g))
        if tau < .1:
            rr = (26 if self.big else 18) * (1 - tau / .1) + 4
            c.circle("art", cx, cy, rr * 2, 120 if self.harmless else 255)
        if tau > self.life or (self.harmless and tau > .45):
            return
        dist = self.speed * (tau - .25 * tau * tau)
        for i in range(self.n):
            a = self.phase + i / self.n * math.tau + self.jit[i]
            ca, sa = math.cos(a), math.sin(a)
            px, py = cx + ca * dist, cy + sa * dist
            q = (cx + ca * (dist - 8), cy + sa * (dist - 8))
            c.circle("art", q[0], q[1], self.r * 1.2, 150 if not self.harmless else 70)
            c.circle("art", px, py, self.r * 2, fade_gray(hv, 1 - tau / .45) if self.harmless else hv)
            if not self.harmless:
                c.circle("haz", px, py, self.r * 2, 255)


# ======================================================================
class Bone(Attack):
    """A plain white bone sliding horizontally, spanning y_a..y_b."""

    def __init__(self, x_ref, vx, t_ref, y_a, y_b, t_from, t_to, width=10):
        self.xr, self.vx, self.tr = x_ref, vx, t_ref
        self.ya, self.yb = min(y_a, y_b), max(y_a, y_b)
        self.wd = width
        self.t0, self.t1 = t_from, t_to

    def draw(self, c):
        x = self.xr + self.vx * (c.t - self.tr)
        put_bone(c, x, self.ya, self.yb)


# ======================================================================
def draw_bone(c: Ctx, layers, p0, p1, val=255, width=6, knob=8, spread=3):
    """A white bone between p0 and p1 (any direction), UT style knobs at both ends."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = np.linalg.norm(d)
    if L < 2:
        return
    u = d / L
    q = np.array([-u[1], u[0]])
    for layer in layers:
        v = val if layer != "haz" else 255
        c.line(layer, [p0 + u * 3, p1 - u * 3], width, v)
        for e in (p0 + u * 3, p1 - u * 3):
            for sgn in (-1, 1):
                k = e + q * sgn * spread
                c.circle(layer, k[0], k[1], knob, v)


_PEN = {}


def pen_sprite(frame: int, mirrored: bool = False):
    """pen0/pen1 enlarged 2x with nearest neighbour (keeps the pixel grain)."""
    key = (frame, mirrored)
    if key not in _PEN:
        s = sprite("pen1.png" if frame else "pen0.png")
        from common import Sprite
        v, al = s.v.repeat(2, 0).repeat(2, 1), s.a.repeat(2, 0).repeat(2, 1)
        if mirrored:
            v, al = v[:, ::-1].copy(), al[:, ::-1].copy()
        _PEN[key] = Sprite(v, al)
    return _PEN[key]


# ---------------------------------------------------------------- bone sprite
_BONES = {}
CAP = 7          # rows of the knobbly end caps in sprite/bone/*.png (10 x 30)


def bone_sprite(length: int, kind: str = "white", angle: float = 0.0):
    """The user's UT bone (sprite/bone/<kind>.png), its shaft stretched to `length`
    pixels (caps unchanged), optionally rotated (degrees, nearest neighbour)."""
    length = max(2 * CAP + 2, int(round(length / 2)) * 2)
    ang = int(round(angle / 7.5)) * 7.5 % 180
    key = (length, kind, ang)
    if key not in _BONES:
        from common import Sprite, ROOT
        im = Image.open(ROOT / "sprite" / "bone" / f"{kind}.png").convert("RGBA")
        a = np.asarray(im)
        mid = a[CAP:CAP + 1].repeat(length - 2 * CAP, 0)
        full = np.concatenate([a[:CAP], mid, a[-CAP:]], 0)
        img = Image.fromarray(full)
        if ang:
            img = img.rotate(ang, resample=Image.Resampling.NEAREST, expand=True)
        arr = np.asarray(img)
        _BONES[key] = Sprite(np.asarray(img.convert("L")).copy(), arr[..., 3] > 127)
    return _BONES[key]


def put_bone(c, x, y0, y1, kind="white", gain=1.0, hazard=True):
    """A vertical bone from y0 to y1 centred on x (world px)."""
    top, bot = min(y0, y1), max(y0, y1)
    spr = bone_sprite(bot - top, kind)
    c.sprites.append((spr, int(round(x - spr.w / 2)), int(round(top)), gain, None))
    if hazard:
        c.rect("haz", x - 4, top + 1, x + 4, top + spr.h - 1, 255)


_MELT = {}


def melted_bone(length, angle, m, seed=0):
    """A bone losing its shape: the outline softens (knobs merge, ends round off),
    frays at the edges and greys into ink as m goes 0 -> 1."""
    from scipy import ndimage as ndi
    from common import Sprite
    mq = int(round(clamp(m) * 8))
    aq = int(round(angle / 7.5)) * 7.5 % 180
    key = (length, aq, mq, seed % 4)
    if key in _MELT:
        return _MELT[key]
    base = bone_sprite(length, "white", aq)
    if mq == 0:
        _MELT[key] = base
        return base
    mm = mq / 8
    P = 4
    a = np.pad(base.a.astype(float), P)
    v = np.pad(base.v.astype(float), P)
    soft = ndi.gaussian_filter(a, .4 + 2.2 * mm)
    soft = ndi.zoom(soft, (1 + .6 * mm, 1), order=1)      # the softened shape sags like wet ink
    v = ndi.zoom(v, (1 + .6 * mm, 1), order=0)
    a = ndi.zoom(a, (1 + .6 * mm, 1), order=0)
    alpha = soft > (.40 + .14 * mm)
    noise = np.random.default_rng(seed % 4 + 7).random(alpha.shape)
    alpha &= noise > .55 * mm * mm * (1 - soft)            # frays from the edges in
    vv = np.where(a > 0, v, 220) * (1 - mm) + 196 * mm
    spr = Sprite(np.clip(vv, 0, 255).astype(np.uint8), alpha)
    _MELT[key] = spr
    return spr


class TopBones(Attack):
    """Ink Sans summons his brush (pen0/pen1), paints two bars just above the
    arena on the beat, the bars charge, turn into two rows of bones that stab
    down column by column on the sixteenth notes, and shatter on the downbeat."""

    PEN_SCALE = 2
    PEN_TIP = (36 * 2 + 1, 105 * 2 + 1)   # bristle tip inside the 2x pen sprite

    def __init__(self, inner, bars, t_summon, beats, sixteenths, t_shatter, seed=0):
        self.inner = inner                    # ix0, iy0, ix1, iy1
        self.bars = bars                      # [(x0, x1, y), (x0, x1, y)]
        self.ts = t_summon
        self.beats = beats                    # stroke1 start, stroke1 end, stroke2 start, stroke2 end, tap
        self.t_bones = sixteenths[0][0]
        self.t_shatter = t_shatter
        rng = np.random.default_rng(seed)
        self.jit = rng.uniform(.7, 1.3, 600)
        self.blot = rng.uniform(0, 1, 32)
        ix0, iy0, ix1, iy1 = inner
        floor = iy1 - 19          # bones stop one soul-height above the floor
        col_of = {8: 0, 7: 1, 6: 2, 5: 3, 4: 4, 3: 5, 2: 6, 1: 7}
        col_time = {col_of[k]: t for t, k in sixteenths}
        colw = (ix1 - ix0) / 8
        self.bones = []
        for row, (x0, x1, y) in enumerate(bars):
            sp = 20
            xs = np.arange(x0 + 6 + row * 10, x1 - 3, sp)
            for x in xs:
                inside = ix0 + 4 <= x <= ix1 - 4
                col = int(min(7, max(0, (x - ix0) // colw)))
                t_hit = col_time.get(col, 9e9) if inside else 9e9
                self.bones.append(dict(x=float(x), y=float(y), l0=8 + 3 * row,
                                       l1=(floor - 6 * row) - y, t=t_hit + .003 * (int(x) % 4),
                                       frag=rng.uniform(-1, 1, 6)))
        self.center = np.array([(ix0 + ix1) / 2, (iy0 + iy1) / 2])
        self.t0 = t_summon - .2
        self.t1 = t_shatter + .9

    # -------------------------------------------------------------- pen paths
    def pens(self, t):
        """Two brushes, summoned together: the left one paints the upper bar
        left->right, the right one (mirrored) paints the lower bar right->left.
        Returns [(tip, frame, mirrored)]; frame 1 (pen1) only while painting."""
        (ax0, ax1, ay), (bx0, bx1, by) = self.bars
        s1a, s1b, s2a, s2b, tap = self.beats
        hover = np.array([0.0, 2 * math.sin(t * 7)])
        ch = clamp((t - tap) / (self.t_bones - tap))
        shake = np.array([(2 if int(t * 60) % 2 else -2) * (ch > .45), 0.0])
        out = []
        for side in (0, 1):
            if side == 0:
                start, end, a0, a1 = np.array([ax0, ay]), np.array([ax1, ay]), s1a, s1b
            else:
                start, end, a0, a1 = np.array([bx1, by]), np.array([bx0, by]), s2a, s2b
            frame = 0
            if t < a0:                                   # waiting at the start of its bar
                p = start - np.array([0, 10]) + hover
            elif t < a1:                                 # painting
                u = smooth((t - a0) / (a1 - a0))
                p = start + (end - start) * u
                frame = 1
            else:                                        # stays where the stroke ended
                lift = smooth((t - a1) / .2)
                p = end - np.array([0, 10 * lift]) + hover * lift * (1 - ch) + shake
                k = t - (tap - .12)                       # dab on the tap beat, in place
                if 0 <= k < .3:
                    p = p + np.array([0, 10 * math.sin(clamp(k / .3) * math.pi)])
            out.append((p, frame, side == 1))
        return out

    def bar_progress(self, t, row):
        s1a, s1b, s2a, s2b, _ = self.beats
        a, b = (s1a, s1b) if row == 0 else (s2a, s2b)
        return smooth((t - a) / (b - a))

    def draw_bar(self, c, row, val, t):
        x0, x1, y = self.bars[row]
        u = self.bar_progress(t, row)
        if u <= 0:
            return
        if row == 0:
            xa, xb = x0, x0 + (x1 - x0) * u
        else:
            xa, xb = x1 - (x1 - x0) * u, x1
        n = max(1, int((xb - xa) / 4))
        for j in range(n):
            p = xa + (xb - xa) * j / n
            q = xa + (xb - xa) * (j + 1) / n
            k = row * 300 + j
            w = 8 * self.jit[k % 600]
            c.line("top", [(p, y + (self.jit[(k + 7) % 600] - 1) * 2), (q, y)], w, val)
        # dry-brush streaks at the ends
        end = xb if row == 0 else xa
        for k in range(3):
            c.line("top", [(end, y - 3 + 3 * k), (end + (8 + 6 * self.jit[k]) * (1 if row == 0 else -1), y - 3 + 3 * k)],
                   2, int(val * .8))

    def draw(self, c):
        t = c.t
        s1a, s1b, s2a, s2b, tap = self.beats
        if t < self.t_bones:
            # summon blots, one on each side
            (ax0, _, ay), (_, bx1, by) = self.bars
            for (px, py), until in (((ax0, ay - 14), s1a), ((bx1, by - 14), s2a)):
                g = out_cubic((t - self.ts + .15) / .2) * (1 - smooth((t - until) / .3))
                if g > .02:
                    c.poly("top", jagged_circle(px, py, 16 * g, self.blot[:14]), 70)
                    c.poly("top", jagged_circle(px, py, 9 * g, self.blot[14:28]), 20)
            val = 180
            if t > tap:
                ch = clamp((t - tap) / (self.t_bones - tap))
                val = int(180 + 70 * ch * (.5 + .5 * math.sin((t - tap) * (4 + 24 * ch * ch) * math.tau)))
            for row in (0, 1):
                self.draw_bar(c, row, val, t)
            # splats where strokes end and where the brush taps
            for tt, (px, py) in ((s1b, (self.bars[0][1], self.bars[0][2])),
                                 (s2b, (self.bars[1][0], self.bars[1][2])),
                                 (tap, (self.bars[0][1] - 4, self.bars[0][2])),
                                 (tap, (self.bars[1][0] + 4, self.bars[1][2]))):
                age = t - tt
                if 0 <= age < .3:
                    for k in range(9):
                        a = k / 9 * math.tau + tt
                        r = 5 + 34 * out_cubic(age / .3)
                        c.circle("top", px + math.cos(a) * r, py + math.sin(a) * r * .7, 5,
                                 fade_gray(200, 1 - age / .3))
            # charging: bone silhouettes flicker inside the bars
            if t > tap:
                ch = clamp((t - tap) / (self.t_bones - tap))
                if int(t * 60) % max(1, int(6 - 5 * ch)) == 0:
                    for b in self.bones:
                        put_bone(c, b["x"], b["y"] - 6, b["y"] + b["l0"] + 4, gain=(40 + 160 * ch) / 255,
                                 hazard=False)
            # the two pens (user sprite; the right one mirrored), revealed tip-first
            if t >= self.ts:
                reveal = clamp((t - self.ts) / .16)
                for tip, frame, mirrored in self.pens(t):
                    spr = pen_sprite(frame, mirrored)
                    tx = (spr.w - 1 - self.PEN_TIP[0]) if mirrored else self.PEN_TIP[0]
                    mask = None
                    if reveal < 1:
                        rows = np.arange(spr.h)[:, None]
                        mask = np.broadcast_to(rows >= spr.h * (1 - reveal), (spr.h, spr.w))
                    c.sprites.append((spr, tip[0] - tx, tip[1] - self.PEN_TIP[1], 1.0, mask))
            return
        # ---------------- bones
        tb = t - self.t_bones
        if tb < .25:   # both pens burst into ink
            for tip, _, _ in self.pens(self.t_bones - 1e-3):
                for k in range(12):
                    a = k / 12 * math.tau
                    r = 4 + 80 * out_cubic(tb / .25)
                    c.circle("top", tip[0] + math.cos(a) * r, tip[1] - 40 + math.sin(a) * r, 7,
                             fade_gray(230, 1 - tb / .25))
        if t < self.t_shatter:
            jx = 2 if int(t * 60) % 2 else 0
            for b in self.bones:
                tau = t - b["t"]
                ext = 0.0 if tau < 0 else min(1.0, out_cubic(tau / .05)) + .1 * math.sin(clamp(tau / .12) * math.pi)
                L = b["l0"] + (b["l1"] - b["l0"]) * ext
                j = jx if tau >= 0 else 0
                put_bone(c, b["x"] + j, b["y"] - 6, b["y"] + L)
            return
        # ---------------- shatter: the pieces fly apart as before and fade; on
        # the way out each one loses its bone shape (it softens and frays into ink)
        ts = t - self.t_shatter
        g = 1 - smooth(ts / .7)
        melt = smooth((ts - .04) / .38)
        for bi, b in enumerate(self.bones):
            mid = np.array([b["x"], b["y"] + b["l1"] * .5])
            out = mid - self.center
            out = out / (np.linalg.norm(out) + 1e-9)
            for k in range(2):
                f = b["frag"]
                v = out * (240 + 120 * f[k]) + np.array([f[k + 2] * 90, -80])
                p = mid + v * ts + np.array([0, 520 * ts * ts]) + np.array([0, (k - .5) * 24])
                ang = math.degrees(f[k + 4] * 12 * ts)
                spr = melted_bone(18, ang, melt, bi * 2 + k)
                c.sprites.append((spr, int(p[0] - spr.w / 2), int(p[1] - spr.h / 2), g, None))


# ======================================================================
class SoakedEdge(Attack):
    """The arena's top border slowly soaks up ink: grey stains creep through the
    white line and short ink stalactites hang beneath it. Harmless."""

    def __init__(self, t0, t1, seed=0):
        rng = np.random.default_rng(seed)
        self.ts, self.te = t0, t1
        self.stains = rng.uniform(0, 1, (40, 3))      # x position, size, delay
        self.hang = rng.uniform(0, 1, (22, 3))        # x position, length, phase
        self.t0, self.t1 = t0, t1 + .4

    def draw(self, c):
        t = c.t
        x, y, w, h = c.box
        g = smooth((t - self.ts) / .5) * (1 - smooth((t - self.te) / .4))
        if g <= 0:
            return
        for sx, sz, dl in self.stains:
            k = smooth((t - self.ts - dl * .6) / .5) * g
            if k <= 0:
                continue
            px = x + 6 + sx * (w - 12)
            rw = (6 + 16 * sz) * k
            c.poly("top", [(px - rw, y), (px - rw * .6, y - 2), (px + rw * .7, y - 1), (px + rw, y + 2),
                           (px + rw * .8, y + 6), (px - rw * .7, y + 7)], 120 + int(60 * sz))
            c.poly("top", [(px - rw * .5, y + 1), (px + rw * .4, y + 1), (px + rw * .3, y + 4),
                           (px - rw * .4, y + 4)], 70)
        for hx, hl, ph in self.hang:
            k = smooth((t - self.ts - ph * .8) / .6) * g
            if k <= 0:
                continue
            px = x + 8 + hx * (w - 16)
            L = (4 + 10 * hl) * k * (1 + .25 * math.sin(t * 3 + ph * 9))
            c.line("art", [(px, y + 5), (px, y + 5 + L)], 4, 150)
            c.circle("art", px, y + 5 + L, 6, 175)


class RainDrop(Attack):
    """A drop swells under the soaked top border, falls, and splashes on the
    floor of the arena exactly on the beat: a crown of droplets, a short jet,
    then a dark stain."""

    def __init__(self, x, t_land, y_top, y_bot, fall=.40, seed=0, big=False):
        self.x, self.tl, self.yt, self.yb = x, t_land, y_top, y_bot
        self.fall = fall
        self.tr = t_land - fall
        self.r = 8 if big else 6
        rng = np.random.default_rng(seed)
        n = 9 if big else 7
        ang = np.linspace(-2.75, -.39, n) + rng.uniform(-.12, .12, n)
        spd = rng.uniform(130, 210, n) * (1.2 if big else 1)
        self.crown = [(math.cos(a) * s, math.sin(a) * s) for a, s in zip(ang, spd)]
        self.blot = rng.uniform(0, 1, 14)
        self.t0, self.t1 = self.tr - .30, t_land + 1.6
        self.impact = t_land
        self.shake = 1.5 if big else 0.0

    def draw(self, c):
        t, x, r = c.t, self.x, self.r
        if t < self.tr:                              # swelling under the border
            u = (t - self.tr + .30) / .30
            rr = r * out_cubic(u)
            c.line("art", [(x, self.yt), (x, self.yt + rr * 1.6)], max(2, rr), 150)
            c.circle("art", x, self.yt + rr * 1.6, rr * 2, int(150 + 80 * u))
            return
        if t < self.tl:                              # falling
            u = in_quad((t - self.tr) / self.fall)
            y = self.yt + 6 + (self.yb - self.yt - r - 6) * u
            tail = 4 + 22 * u
            c.poly("art", [(x - r * .8, y - r * .3), (x, y - r - tail), (x + r * .8, y - r * .3)], 200)
            c.circle("art", x, y, r * 2, 240)
            c.circle("haz", x, y, r * 2, 255)
            c.poly("haz", [(x - r * .5, y), (x, y - r - tail * .5), (x + r * .5, y)], 255)
            return
        tau = t - self.tl
        g = 1 - smooth((tau - .3) / 1.2)
        if not getattr(self, "pooled", False):       # on the bare floor a stain is left
            c.poly("art", [(px, self.yb - 2 + (py - self.yb) * .3) for px, py in
                           jagged_circle(x, self.yb, r * 2.6, self.blot)], fade_gray(30, g))
        elif tau < .5:                               # on the pool: rings on the surface
            rr = 8 + 60 * out_cubic(tau / .5)
            c.line("art", [(x - rr, self.yb + 1), (x - rr * .6, self.yb - 1)], 2, fade_gray(240, 1 - tau / .5))
            c.line("art", [(x + rr * .6, self.yb - 1), (x + rr, self.yb + 1)], 2, fade_gray(240, 1 - tau / .5))
        if tau < .07:                                # impact flash, flat on the floor
            k = tau / .07
            c.poly("art", [(x - r * 3.5 * (1 - k * .4), self.yb), (x, self.yb - r * 1.2),
                           (x + r * 3.5 * (1 - k * .4), self.yb)], 255)
        if tau < .30:                                # rebound jet
            jh = 26 * math.sin(clamp(tau / .30) * math.pi)
            if jh > 3:
                c.line("art", [(x, self.yb), (x, self.yb - jh)], 4, 225)
                c.circle("art", x, self.yb - jh, 7, 240)
                c.line("haz", [(x, self.yb), (x, self.yb - jh)], 4, 255)
                c.circle("haz", x, self.yb - jh, 7, 255)
        for vx, vy in self.crown:                    # crown of droplets
            px = x + vx * tau
            py = self.yb - 3 + vy * tau + 560 * tau * tau
            if py > self.yb or tau > .7:
                continue
            c.circle("art", px - vx * .012, py - vy * .012, 4, 150)
            c.circle("art", px, py, 6, 235)
            c.circle("haz", px, py, 6, 255)


class FloorWave(Attack):
    """A slide becomes a splash crest racing along the arena floor."""

    def __init__(self, x0, x1, ts, te, y_bot, seed=0, warn=.25, surf=None, height=26):
        self.x0, self.x1, self.ts, self.te, self.yb = x0, x1, ts, max(te, ts + .12), y_bot
        self.warn = warn
        self.surf = surf                 # optional (x, t) -> floor y (a sloshing pool surface)
        self.height = height
        rng = np.random.default_rng(seed)
        self.spray = rng.uniform(0, 1, (24, 3))
        self.t0, self.t1 = ts - warn, self.te + .6

    def pos(self, t):
        u = smooth((t - self.ts) / (self.te - self.ts))
        return self.x0 + (self.x1 - self.x0) * u

    def floor(self, x, t):
        return self.surf(x, t) if self.surf is not None else self.yb

    def draw(self, c):
        t = c.t
        d = 1 if self.x1 >= self.x0 else -1
        if t < self.ts:                              # warning: a ripple line on the floor
            u = (t - self.t0) / self.warn
            on = int(t * 24) % 2 == 0
            for k in range(0, int(abs(self.x1 - self.x0)) + 1, 10):
                px = self.x0 + d * k
                c.circle("art", px, self.floor(px, t) - 3, 4, 90 if on else 60)
            c.circle("art", self.x0, self.floor(self.x0, t) - 6, 6 + 8 * u, 120)
            return
        if t > self.te + .25:
            return
        x = self.pos(min(t, self.te))
        yb = self.floor(x, t)
        fade = 1 - smooth((t - self.te) / .25)
        hgt = self.height * fade
        # crest: a curled hump leaning in the travel direction
        pts = [(x - d * 34, yb), (x - d * 18, yb - hgt * .45), (x - d * 4, yb - hgt),
               (x + d * 8, yb - hgt * .9), (x + d * 14, yb - hgt * .5), (x + d * 12, yb)]
        c.poly("art", pts, 230)
        c.poly("art", [(x - d * 14, yb), (x - d * 6, yb - hgt * .5), (x + d * 4, yb - hgt * .6),
                       (x + d * 6, yb)], 255)
        if hgt > 4:
            c.poly("haz", pts, 255)
        for sx, sy, sz in self.spray:                # spray thrown ahead and up
            a = (t * 4 + sx) % 1
            px = x + d * (6 + 30 * a) + (sx - .5) * 10
            py = yb - hgt - 4 - 30 * a * sy + 60 * a * a
            if py < yb - 2:
                c.circle("art", px, py, 4 + 3 * sz, int(200 * (1 - a) * fade))


class InkFlick(Attack):
    """Ink Sans flicks his brush: a fan of ink droplets sprays off the soaked top
    edge and arcs down across the canvas (Pollock-style drip painting)."""

    def __init__(self, origin, direction, t_hit, n=4, seed=0, tele=.22, gravity=430):
        self.O = np.array(origin, float)
        self.d = 1 if direction >= 0 else -1
        self.th, self.tele, self.g = t_hit, tele, gravity
        rng = np.random.default_rng(seed)
        ang = np.radians(np.linspace(18, 64, n) + rng.uniform(-4, 4, n))
        spd = rng.uniform(170, 270, n)
        self.v = [(self.d * math.cos(a) * s, math.sin(a) * s) for a, s in zip(ang, spd)]
        self.t0, self.t1 = t_hit - tele, t_hit + 1.0
        self.impact = None

    def at(self, k, tau):
        vx, vy = self.v[k]
        return self.O[0] + vx * tau, self.O[1] + vy * tau + .5 * self.g * tau * tau

    def draw(self, c):
        t = c.t
        tau = t - self.th
        ox, oy = self.O
        if tau < 0:                                   # bead trembling on the edge + arc preview
            u = (tau + self.tele) / self.tele
            j = 2 if int(t * 60) % 2 else -2
            c.circle("art", ox + j * (u > .5), oy + 2, 6 + 6 * u, int(120 + 100 * u))
            mid = len(self.v) // 2
            for s in np.linspace(.04, .36, 8):
                px, py = self.at(mid, s)
                c.circle("art", px, py, 4, int(50 + 40 * u))
            return
        if tau < .1:                                  # the flick mark on the edge
            c.line("art", [(ox - self.d * 6, oy), (ox + self.d * 26 * (1 - tau / .1), oy + 6)], 4, 230)
        surf = getattr(self, "surf", None)
        for k in range(len(self.v)):
            px, py = self.at(k, tau)
            if surf is not None and py > surf(px, t):
                continue                              # swallowed by the rising pool
            qx, qy = self.at(k, max(0.0, tau - .035))
            c.line("art", [(qx, qy), (px, py)], 5, 210)
            c.circle("art", px, py, 8, 245)
            c.line("haz", [(qx, qy), (px, py)], 5, 255)
            c.circle("haz", px, py, 8, 255)


class PoolLevel:
    """How high the fallen ink stands in the arena over time: every drop that
    lands raises it a little; at the end it surges up to the brim."""

    def __init__(self, landings, step, cap, surge_t, surge_dur, full, cx=320.0, half=175.0):
        self.land = sorted(landings)
        self.step, self.cap = step, cap
        self.surge_t, self.surge_dur, self.full = surge_t, surge_dur, full
        self.cx, self.half = cx, half
        self.tilts = []                  # (time, target): the pool sloshes from side to side
        self.rise = []                   # (time, level): ink poured in on the beat
        self.freeze_t = self.engulf_t = 1e9   # hazard stops rising / the soul is swallowed
        self.floor_y = None              # world y of the arena floor (set by the builder)

    def __call__(self, t, hazard=False):
        if hazard and t >= self.engulf_t:
            return 0.0                   # engulfed: the soul is inside the ink now
        if hazard and t >= self.freeze_t:
            t = self.freeze_t
        lv = 0.0
        for tl in self.land:
            if tl > t:
                break
            lv += self.step * smooth((t - tl) / .22)
        lv = min(lv, self.cap)
        for tk, target in self.rise:
            if tk > t:
                break
            lv = lv + (max(lv, target) - lv) * out_cubic(clamp((t - tk) / .08))
        if not hazard and t > self.surge_t:
            lv = lv + (self.full - lv) * in_quad((t - self.surge_t) / self.surge_dur)
        return lv

    def tilt(self, t):
        """Height difference at the walls (+: the right side stands higher). Each
        key swings the surface over with a sloshing overshoot."""
        a = 0.0
        for tk, target in self.tilts:
            if tk > t:
                break
            u = clamp((t - tk) / .26)
            e = 1 + 2.4 * (u - 1) ** 3 + 1.4 * (u - 1) ** 2
            a = a + (target - a) * e
        return a

    def height(self, x, t, hazard=False):
        """Ink depth at world x (the tilted surface; never below the floor)."""
        lv = self(t, hazard)
        if lv <= 0:
            return 0.0
        return max(0.0, lv + self.tilt(t) * (x - self.cx) / self.half)

    def surface(self, x, t, hazard=True):
        """World y of the ink surface at x."""
        return self.floor_y - self.height(x, t, hazard)


class InkPool(Attack):
    """The ink that rained down collects on the arena floor and rises."""

    def __init__(self, level: PoolLevel, t0, t1, seed=0):
        self.level = level
        self.t0, self.t1 = t0, t1
        rng = np.random.default_rng(seed)
        self.foam = rng.uniform(0, 1, (30, 2))

    def draw(self, c):
        t = c.t
        x, y, w, h = c.box
        ix0, ix1, floor = x + 5, x + w - 5, y + h - 5
        L = self.level
        lv = L(t)
        if lv < 1:
            return
        surf, hsurf = [], []
        for px in range(ix0 - 4, ix1 + 6, 4):
            dy = 2 * math.sin(px * .07 + t * 5) + 1.5 * math.sin(px * .19 - t * 7)
            surf.append((px, floor - L.height(px, t) + dy))
            hsurf.append((px, floor - L.height(px, t, hazard=True) + dy))
        body = surf + [(ix1 + 6, floor + 4), (ix0 - 4, floor + 4)]
        c.poly("art", body, 108)
        c.line("art", surf, 4, 232)
        for fx_, fy in self.foam:                          # foam riding the surface
            px = ix0 + ((fx_ + t * .05 * (1 if fy > .5 else -1)) % 1) * (ix1 - ix0)
            k = int((px - (ix0 - 4)) / 4)
            k = min(max(k, 0), len(surf) - 1)
            c.circle("art", px, surf[k][1] + 2 + 3 * fy, 4, 190)
        # sloshing: every swing throws a sheet of spray off the wall it climbs
        for j, (tk, target) in enumerate(L.tilts):
            age = t - tk - .16
            if not (0 <= age < .5) or target == 0:
                continue
            wall = ix1 - 6 if target > 0 else ix0 + 6
            inward = -1 if target > 0 else 1
            y0 = floor - L.height(wall, tk + .16)
            for k in range(6):
                vx = inward * (60 + 34 * k)
                vy = -(250 - 22 * k)
                px, py = wall + vx * age, y0 + vy * age + 700 * age * age
                if py > floor - L.height(px, t):
                    continue
                c.circle("art", px, py, 7 if k % 2 else 5, fade_gray(235, 1 - age / .5))
                if getattr(L, "spray_hazard", lambda tt: False)(tk):
                    c.circle("haz", px, py, 6, 255)
        if L(t, hazard=True) > 4:
            c.poly("haz", hsurf + [(ix1 + 6, floor + 4), (ix0 - 4, floor + 4)], 255)


# ======================================================================
class InkStroke(Attack):
    """A brush stroke on the canvas, readable as a game attack:
      1. warning - the exact same stroke is first swept in as diluted, pale ink
         (`lead` s ahead) and flickers darker just before the hit; harmless
      2. live    - on the beat it is painted in solid black: the only black on
         the canvas is ink that hurts
      3. gone    - it thins and fades out within .2 s (or shatters into pale,
         harmless flakes at `shatter`), so no old marks clutter the canvas"""

    def __init__(self, path, t_hit, width=12, lead=.55, sweep=.06, linger=.12, seed=0,
                 shatter=None, dry=.2):
        pts = np.array(path, float)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.concatenate([[0], np.cumsum(seg)])
        n = max(3, int(cum[-1] / 4))
        s = np.linspace(0, cum[-1], n)
        self.p = np.stack([np.interp(s, cum, pts[:, 0]), np.interp(s, cum, pts[:, 1])], 1)
        self.u = s / max(cum[-1], 1e-6)
        rng = np.random.default_rng(seed)
        prof = np.where(self.u < .12, .55 + .45 * (self.u / .12), 1 - .75 * np.clip((self.u - .45) / .55, 0, 1))
        self.w = width * prof * (0.9 + 0.2 * rng.random(n))
        self.gap = (rng.random(n) < .18) & (self.u > .55)          # dry-brush breaks in the tail
        self.th, self.lead, self.sweep, self.linger = t_hit, lead, sweep, linger
        self.shatter, self.dry = shatter, dry
        self.flakes = rng.uniform(-1, 1, (len(self.p) // 3 + 1, 3))
        self.splat = rng.uniform(0, 1, (6, 3))
        end = (shatter if shatter is not None else t_hit + sweep + linger + dry)
        self.t0, self.t1 = t_hit - lead, end + (.6 if shatter is not None else 0)
        self.impact = None

    def draw(self, c):
        t = c.t
        n = len(self.p)
        if t < self.th:                                          # warning: a clear wash of ink
            k = clamp((t - self.th + self.lead) / .1)
            m = max(2, int(k * n))
            late = t > self.th - .16
            val = (150 if int(t * 60) % 4 < 2 else 96) if late else 100
            for i in range(m - 1):
                w = self.w[i] * 1.15
                c.line("art", [self.p[i], self.p[i + 1]], w, val)
                c.circle("art", self.p[i][0], self.p[i][1], w, val)
            if k < 1:                                            # the bead laying the wash down
                q = self.p[m - 1]
                c.circle("art", q[0], q[1], self.w[m - 1] * 1.6, 170)
            return
        head = clamp((t - self.th) / self.sweep)
        m = max(2, int(head * n))
        live_until = self.th + self.sweep + self.linger
        if self.shatter is not None and t >= self.shatter:      # flakes
            a = t - self.shatter
            g = 1 - smooth(a / .5)
            for j, (fx_, fy, fr) in enumerate(self.flakes):
                i = min(n - 1, j * 3)
                q = self.p[i] + np.array([fx_, fy]) * 160 * a + np.array([0, 300 * a * a])
                c.circle("art", q[0], q[1], 4 + 4 * abs(fr), fade_gray(90, g))     # pale: harmless
            return
        if self.shatter is not None:
            val, live = 255, True
        elif t < live_until:
            val, live = 255, True
        else:
            u = (t - live_until) / self.dry
            if u >= 1:
                return
            val = int(200 * (1 - smooth(u)))
            live = False
        thin = 1.0 if live else 1 - .6 * smooth((t - live_until) / self.dry)
        for i in range(m - 1):
            if self.gap[i]:
                continue
            w = self.w[i] * thin
            c.line("art", [self.p[i], self.p[i + 1]], w, val)
            c.circle("art", self.p[i][0], self.p[i][1], w, val)
            if live and w > 3:
                c.line("haz", [self.p[i], self.p[i + 1]], w, 255)
        if t - self.th < .12:                                    # pale splatter where the brush lands
            a = (t - self.th) / .12
            for sx, sy, sr in self.splat:
                q = self.p[0] + (np.array([sx, sy]) - .5) * 40 * a
                c.circle("art", q[0], q[1], 3 + 4 * sr, fade_gray(100, 1 - a))


class Enso(Attack):
    """A single unbroken brush circle (禅 圆相) painted slowly around the arena;
    harmless - the closing gesture before a rest."""

    def __init__(self, center, radius, a0, sweep_deg, ts, te, width=16, hold=1.2, seed=0):
        self.c = np.array(center, float)
        self.r, self.a0, self.sw = radius, math.radians(a0), math.radians(sweep_deg)
        self.ts, self.te, self.hold = ts, te, hold
        rng = np.random.default_rng(seed)
        self.jit = rng.uniform(.85, 1.15, 400)
        self.t0, self.t1 = ts, te + hold + 1.0

    def draw(self, c):
        t = c.t
        u = smooth((t - self.ts) / (self.te - self.ts))
        g = 1 - smooth((t - self.te - self.hold) / 1.0)
        n = int(220 * u)
        prev = None
        for i in range(n):
            s = i / 220
            a = self.a0 + self.sw * s
            r = self.r * (1 + .04 * math.sin(s * 9))
            p = (self.c[0] + math.cos(a) * r * 1.25, self.c[1] + math.sin(a) * r)
            w = 16 * (.5 + .5 * math.sin(min(1, s * 3) * math.pi / 2)) * (1 - .6 * s) * self.jit[i]
            if prev is not None and not (s > .7 and self.jit[i] > 1.1):
                c.line("top", [prev, p], max(2, w), fade_gray(125, g))     # 淡墨: pale, harmless
            prev = p


# ======================================================================
class InkThread(Attack):
    """A thread of ink drops from the soaked top border and hangs there for a
    beat; then it hardens into a bone and falls into the pool."""

    def __init__(self, x, t_drop, t_fall, y_top, length, surface, seed=0, swell=.22):
        self.x, self.td, self.tf = x, t_drop, t_fall
        self.yt, self.L = y_top, length
        self.surface = surface             # (x, t) -> world y the bone sinks into
        self.swell = swell
        rng = np.random.default_rng(seed)
        self.ph = rng.uniform(0, math.tau)
        self.t0, self.t1 = t_drop - swell, t_fall + .9
        self.impact = None

    def _fall(self, tau):
        return 40 * tau + 1100 * tau * tau

    def draw(self, c):
        t, x = c.t, self.x
        if t < self.td:                                  # a bead swells under the border
            u = clamp((t - self.t0) / self.swell)
            rr = 6 * out_cubic(u)
            c.line("art", [(x, self.yt), (x, self.yt + rr * 1.4)], max(2, rr), 150)
            c.circle("art", x, self.yt + rr * 1.4, max(2, rr * 2), int(150 + 80 * u))
            return
        if t < self.tf:                                  # the thread hangs
            ext = out_cubic(clamp((t - self.td) / .07))
            L = self.L * ext * (1 + .04 * math.sin((t - self.td) * 14))
            sway = 2 * math.sin(t * 9 + self.ph) * ext
            bone = t >= self.tf - .04 or (t >= self.tf - .12 and int(t * 60) % 2 == 0)
            if bone:                                     # hardening: ink -> bone
                put_bone(c, x, self.yt + 1, self.yt + L + 6)
                return
            c.line("art", [(x, self.yt), (x + sway * .5, self.yt + L * .6), (x + sway, self.yt + L)], 6, 230)
            c.circle("art", x + sway, self.yt + L, 12, 245)
            c.circle("art", x + sway - 2, self.yt + L - 2, 4, 255)
            c.line("haz", [(x, self.yt), (x + sway, self.yt + L)], 6, 255)
            c.circle("haz", x + sway, self.yt + L, 12, 255)
            return
        tau = t - self.tf                                # the bone drops into the pool
        top = self.yt + 1 + self._fall(tau)
        bot = top + self.L + 5
        sy = self.surface(x, t)
        if bot < sy:
            put_bone(c, x, top, bot)
        elif sy - top > 16:
            put_bone(c, x, top, sy)
        # where it pierces the surface: a splash and rings
        a, b = 1100.0, 40.0
        cdist = sy - (self.yt + 1 + self.L + 5)
        if cdist > 0:
            te = (-b + math.sqrt(b * b + 4 * a * cdist)) / (2 * a)
            age = tau - te
            if 0 <= age < .45:
                rr = 6 + 44 * out_cubic(age / .45)
                g = 1 - age / .45
                c.line("art", [(x - rr, sy + 1), (x - rr * .55, sy - 2)], 2, fade_gray(240, g))
                c.line("art", [(x + rr * .55, sy - 2), (x + rr, sy + 1)], 2, fade_gray(240, g))
                if age < .18:
                    jh = 20 * math.sin(age / .18 * math.pi)
                    c.line("art", [(x, sy), (x, sy - jh)], 4, 225)
                    c.circle("art", x, sy - jh, 6, 240)


class InkSlam(Attack):
    """Ink Sans lands and drives his brush straight down through the arena: a
    pencil guide first, then a full-height column of ink with dry-brush edges,
    a splat on the floor and droplets thrown up from it."""

    def __init__(self, x, t_hit, y_top, y_floor, width=26, tele=.3, linger=.2, seed=0):
        self.x, self.th, self.yt, self.yf = x, t_hit, y_top, y_floor
        self.w, self.tele, self.linger = width, tele, linger
        rng = np.random.default_rng(seed)
        self.edge = rng.uniform(-1, 1, (2, 48))
        self.blot = rng.uniform(0, 1, 16)
        ang = np.radians(np.linspace(-165, -15, 8) + rng.uniform(-8, 8, 8))
        spd = rng.uniform(170, 260, 8)
        self.drops = [(math.cos(a) * s, math.sin(a) * s) for a, s in zip(ang, spd)]
        self.t0, self.t1 = t_hit - tele, t_hit + linger + 1.5
        self.impact = None

    def column(self, ybot):
        n = 48
        ys = np.linspace(self.yt, ybot, n)
        half = self.w / 2
        left, right = [], []
        for i, y in enumerate(ys):
            k = (y - self.yt) / max(1.0, self.yf - self.yt)
            hw = half * (1.05 - .25 * k)                 # a brush: heavy at the top
            left.append((self.x - hw + 3 * self.edge[0][i], y))
            right.append((self.x + hw + 3 * self.edge[1][i], y))
        return left + right[::-1]

    def draw(self, c):
        t, x = c.t, self.x
        half = self.w / 2
        if t < self.th:                                  # pencil guide, drawn top -> down
            u = clamp((t - self.t0) / (self.tele * .6))
            on = .75 + .25 * (int(t * 30) % 2)
            yb = self.yt + (self.yf - self.yt) * u
            for y in np.arange(self.yt, yb, 8):
                for ex in (x - half, x + half):
                    c.line("art", [(ex, y), (ex, y + 4)], 2, int(80 * on))
            c.line("art", [(x, self.yt), (x, yb)], 2, int(46 * on))
            return
        tau = t - self.th
        head = clamp(tau / .05)
        ybot = self.yt + (self.yf - self.yt) * head
        live_end = self.linger + .05
        if tau < live_end:
            val, live = 255, True
        else:
            d = tau - live_end
            val = int(255 - 205 * smooth(d / .12)) if d < .12 else int(50 * (1 - smooth((d - .12) / 1.1)))
            live = False
        poly = self.column(ybot)
        if val > 2:
            c.poly("art", poly, val)
            for k in range(4):                           # dry-brush streaks down the column
                sx = x - half * .6 + k * half * .4
                c.line("art", [(sx, self.yt + 6 + 9 * k), (sx, ybot - 10 - 7 * k)], 2,
                       max(0, int(val * .55)))
        if live:
            c.poly("haz", poly, 255)
        if head >= 1:                                    # floor splat
            g = 1 - smooth((tau - .3) / 1.4)
            c.poly("art", [(px, self.yf - 2 + (py - self.yf) * .35) for px, py in
                           jagged_circle(x, self.yf, 30, self.blot)], fade_gray(40 if not live else 200, g))
        if tau < .07:                                    # impact flash, wider than the column
            c.poly("art", [(x - half * 2.2, self.yt), (x + half * 2.2, self.yt),
                           (x + half * 1.2, self.yt + 14), (x - half * 1.2, self.yt + 14)], 255)
        if tau < .6:                                     # droplets thrown up from the floor
            for vx, vy in self.drops:
                a = tau - .04
                if a < 0:
                    continue
                px, py = x + vx * a, self.yf - 4 + vy * a + 760 * a * a
                if py > self.yf - 2:
                    continue
                c.circle("art", px - vx * .012, py - vy * .012, 4, 150)
                c.circle("art", px, py, 7, 240)
                c.circle("haz", px, py, 6, 255)


# ======================================================================
# Attacks that come from OUTSIDE the canvas. Outside they are white (they are
# seen coming - that is their warning); the moment they cross into the paper
# they turn to ink (scene.compose inverts sprites inside the canvas), a pale
# splash marks the crossing, and at the far side the paper soaks them up.
def put_bone_line(c, p0, p1, gain=1.0, hazard=True, thick=8):
    """A white bone from p0 to p1 at any angle (sprite) + its hazard."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = float(np.hypot(*d))
    if L < 2 * CAP + 2:
        return
    ang = math.degrees(math.atan2(d[0], d[1])) % 180
    spr = bone_sprite(L, "white", ang)
    m = (p0 + p1) / 2
    c.sprites.append((spr, int(round(m[0] - spr.w / 2)), int(round(m[1] - spr.h / 2)), gain, None))
    if hazard:
        u = d / L
        c.line("haz", [p0 + u * 3, p1 - u * 3], thick, 255)


def _clip_param(p, d, g):
    """Entry and exit distance of the ray p + s*d through the arena interior."""
    lo, hi = -1e9, 1e9
    for o, v, a, b in ((p[0], d[0], g[0], g[2]), (p[1], d[1], g[1], g[3])):
        if abs(v) < 1e-9:
            if not (a <= o <= b):
                return None
            continue
        s0, s1 = (a - o) / v, (b - o) / v
        lo, hi = max(lo, min(s0, s1)), min(hi, max(s0, s1))
    return (lo, hi) if lo < hi else None


class FlyingBone:
    """One bone travelling in a straight line from `start` along `d`; enters the
    canvas, then is soaked up when its head reaches the far side."""

    def __init__(self, start, d, speed, t_go, length, inner, seed=0, trail=False):
        self.trail = trail
        self.S, self.d = np.asarray(start, float), np.asarray(d, float) / np.linalg.norm(d)
        self.v, self.tg, self.L = speed, t_go, length
        self.inner = inner                                  # ix0, iy0, ix1, iy1
        cp = _clip_param(self.S, self.d, inner)
        self.s_in, self.s_out = cp if cp else (0.0, 0.0)
        self.t_in = t_go + (self.s_in - length / 2) / speed      # head crosses into the paper
        self.t_out = t_go + (self.s_out - length / 2) / speed    # head reaches the far side
        self.t_end = self.t_out + length / speed + .6
        self.captured_at = None                             # set when a spiral binds it
        rng = np.random.default_rng(seed)
        self.blot = rng.uniform(0, 1, 14)

    def centre(self, t):
        """Centre of the bone at t, or None when it is not flying (yet / any more)."""
        if t < self.tg:
            return None
        s = (t - self.tg) * self.v
        if s + self.L / 2 > self.s_out:
            return None
        return self.S + self.d * s

    def draw(self, c, t):
        if t < self.tg or t > self.t_end:
            return
        if self.captured_at is not None and t >= self.captured_at:
            return
        s = (t - self.tg) * self.v                          # centre distance along the ray
        head = s + self.L / 2
        tail = s - self.L / 2
        if head > self.s_out:                               # soaked up at the far side
            head = self.s_out
        if head - tail >= 2 * CAP + 2:
            if self.trail:                                  # afterimage: the speed of the run
                speed_trail(c, self.S + self.d * max(0.0, tail), self.d, min(80.0, max(0.0, tail) + 10))
            put_bone_line(c, self.S + self.d * tail, self.S + self.d * head)
        # pale splash where it crossed into the paper (harmless; carried off by the paper)
        a = t - self.t_in
        if 0 <= a < .3:
            q = self.S + self.d * self.s_in + np.array(scroll_d(t, self.t_in))
            c.poly("art", jagged_circle(q[0], q[1], 4 + 12 * out_cubic(a / .3), self.blot, n=10),
                   fade_gray(105, 1 - a / .3))
        # a pale stain blooms where the paper drank it - at its edge, which is where
        # the paper drinks (it does not ride off with the moving sheet: it would leave
        # the canvas at once and the bone would seem simply to vanish)
        a = t - (self.t_out - self.L / self.v)               # from the moment its head touches the edge
        if a >= 0:
            q = self.S + self.d * self.s_out
            big = 1.5 if self.trail else 1.0
            c.poly("art", jagged_circle(q[0], q[1], (4 + 8 * out_cubic(min(1, a / .2))) * big, self.blot, n=12),
                   fade_gray(95, 1 - smooth((a - .15) / .35)))
            if self.trail and a < .4:                       # it splashes out past the edge as it is drunk
                for j in range(3):
                    r = self.blot[j]
                    v = self.d * (230 + 260 * r) + np.array([0.0, (self.blot[j + 5] - .5) * 300])
                    px, py = q + v * a + np.array([0.0, 520 * a * a])
                    c.circle("top", px, py, (3 + 3 * r) * (1 - .5 * a / .4), fade_gray(215, 1 - a / .4))


class BoneShots(Attack):
    """Single bones thrown in from outside the canvas (no orb), each entering the
    paper on its beat: [(t_enter, start (x, y), direction (dx, dy))]."""

    def __init__(self, inner, shots, speed=620.0, length=30, seed=0, trail=False):
        rng = np.random.default_rng(seed)
        self.bones = []
        for t_enter, start, d in shots:
            probe = FlyingBone(start, d, speed, 0.0, length, inner)
            self.bones.append(FlyingBone(start, d, speed, t_enter - probe.t_in, length, inner,
                                         seed=int(rng.integers(1_000_000)), trail=trail))
        self.t0 = min(b.tg for b in self.bones)
        self.t1 = max(b.t_end for b in self.bones)
        self.impact = None

    def draw(self, c):
        for b in self.bones:
            b.draw(c, c.t)


class PenVolleys(Attack):
    """M17-M18. On a strong beat a bar of ink writes itself beside the arena (a
    bright bead runs down it, no brush in sight); every following sixteenth pops
    one piece of that bar into a bone; on the next strong beat the bones fly into
    the canvas as a wall with a gap, turn to ink, and are soaked up at the far side."""

    def __init__(self, g, bars, t_in, t_out, speed=380.0, seed=0, trail=False):
        self.g = g
        self.inner = (g.ix0, g.iy0, g.ix1, g.iy1)
        self.bar_x = {"L": g.x - 62, "R": g.x + g.w + 62}
        self.y0, self.y1 = g.iy0 + 2, g.iy1 - 2
        self.t_in, self.t_out = t_in, t_out
        rng = np.random.default_rng(seed)
        self.jit = rng.uniform(.7, 1.3, 400)
        self.bars = []
        for b in bars:                                   # dict(side, paint, pops=[(t, y)], fire)
            x = self.bar_x[b["side"]]
            d = np.array([1.0, 0.0]) if b["side"] == "L" else np.array([-1.0, 0.0])
            bones = [FlyingBone((x, y), d, speed, b["fire"], 34, self.inner, seed=int(rng.integers(1_000_000)),
                                trail=trail)
                     for _, y in b["pops"]]
            self.bars.append(dict(b, x=x, bones=bones))
        self.t0 = t_in
        self.t1 = max(max(bb.t_end for bb in b["bones"]) for b in self.bars) if self.bars else t_out
        self.t1 = max(self.t1, t_out + .3)
        self.impact = None

    def draw(self, c):
        t = c.t
        for b in self.bars:
            x = b["x"]
            a = t - b["paint"]
            if a < 0:
                continue
            if t < b["fire"]:
                # the bar, painted top -> bottom, minus the pieces already turned to bone
                yb = self.y0 + (self.y1 - self.y0) * smooth(min(1.0, a / .12))
                popped = [y for tp, y in b["pops"] if tp <= t]
                n = max(1, int((yb - self.y0) / 4))
                for j in range(n):
                    ya = self.y0 + j * 4
                    if any(abs(ya - y) < 9 for y in popped):
                        continue
                    w = 8 * self.jit[(j + int(x)) % 400]
                    c.line("top", [(x + (self.jit[(j * 7) % 400] - 1) * 2, ya), (x, ya + 4)], w, 190)
                if a < .16:                                    # the bead of ink writing the bar
                    c.circle("top", x, yb, 14, 245)
                    c.circle("top", x, yb - 7, 8, 200)
                    for k in range(3):
                        c.line("top", [(x - 5 + 5 * k, yb - 10 - 14 * self.jit[k]), (x - 5 + 5 * k, yb - 4)],
                               2, 150)
                for tp, y in b["pops"]:
                    if tp <= t:                                # popped bone waiting in the bar
                        put_bone_line(c, (x - 17, y), (x + 17, y), hazard=False)
                        ap = t - tp
                        if ap < .1:
                            c.circle("top", x, y, 10 + 30 * ap / .1, fade_gray(200, 1 - ap / .1))
            else:
                af = t - b["fire"]
                if af < .3:                                    # what is left of the bar breaks up
                    for j in range(0, int((self.y1 - self.y0) / 8)):
                        ya = self.y0 + j * 8
                        q = (x + (self.jit[j] - 1) * 60 * af, ya + 120 * af * af)
                        c.circle("top", q[0], q[1], 4, fade_gray(150, 1 - af / .3))
                for bone in b["bones"]:
                    bone.draw(c, t)
                    a = t - bone.t_out
                    if 0 <= a < .45 and bone.d[0] < 0:          # splashed out past the left edge
                        q = bone.S + bone.d * bone.s_out
                        for j in range(3):
                            r = self.jit[(int(q[1]) * 3 + j * 37) % 400]
                            r2 = self.jit[(int(q[1]) * 7 + j * 53 + 11) % 400]
                            vx = -330 - 640 * (r - .7)                # flung out hard...
                            vy = (r2 - 1) * 430 + (j - 1) * 70        # ...and fanned up and down
                            px = self.g.x - 14 + vx * a
                            py = q[1] + vy * a + 650 * a * a
                            c.circle("top", px, py, (3 + 3 * r) * (1 - .4 * a / .45), fade_gray(225, 1 - a / .45))


class PaperRun(Attack):
    """The paper moving under the canvas like a treadmill (run: left, climb: down):
    old pale stains and dry-brush marks ride on it - they come in at the leading
    edge and are carried across. Pale and harmless; drawn first, under everything;
    they fade out as the paper stops (the tunnel)."""

    z = -1

    def __init__(self, inner, offset, t_in, t_stop, seed=0):
        self.inner, self.offset = inner, offset
        self.t0, self.t1 = t_in, t_stop + .05
        self.t_stop = t_stop
        self.impact = None
        rng = np.random.default_rng(seed)
        ix0, iy0, ix1, iy1 = inner
        # where the canvas window sits on the sheet over time (sheet coords = screen - offset)
        wins = []
        for t in np.arange(t_in, t_stop, .02):
            dx, dy = offset(t)
            wins.append((ix0 - dx, iy0 - dy, ix1 - dx, iy1 - dy))
        wins = np.array(wins)
        x_lo, y_lo = wins[:, 0].min(), wins[:, 1].min()
        x_hi, y_hi = wins[:, 2].max(), wins[:, 3].max()
        first = wins[0]
        self.marks = []                                      # (X, Y on the sheet, kind, size, value, blot, angle)
        step = 30.0
        for X in np.arange(x_lo, x_hi, step):
            for Y in np.arange(y_lo, y_hi, step):
                x, y = X + rng.uniform(0, step), Y + rng.uniform(0, step)
                inside = ((wins[:, 0] <= x) & (x <= wins[:, 2]) & (wins[:, 1] <= y) & (y <= wins[:, 3])).any()
                if not inside or (first[0] - 6 <= x <= first[2] + 6 and first[1] - 6 <= y <= first[3] + 6):
                    continue                                 # (none sitting there from the start: all ride in)
                if rng.random() < .45:
                    continue
                kind = "dash" if rng.random() < .35 else "stain"
                self.marks.append((x, y, kind, rng.uniform(2.5, 8.0) if kind == "stain" else rng.uniform(8, 22),
                                   rng.uniform(22, 50), rng.uniform(0, 1, 12), rng.uniform(0, math.pi)))

    def draw(self, c):
        t = c.t
        dx, dy = self.offset(t)
        ix0, iy0, ix1, iy1 = self.inner
        fade = 1 - smooth((t - (self.t_stop - .4)) / .4)
        if fade <= 0:
            return
        for X, Y, kind, size, val, blot, ang in self.marks:
            x, y = X + dx, Y + dy
            if not (ix0 - 30 <= x <= ix1 + 30 and iy0 - 30 <= y <= iy1 + 30):
                continue
            v = fade_gray(val, fade)
            if kind == "stain":
                c.poly("art", jagged_circle(x, y, size, blot, n=12), v)
            else:
                c.line("art", [(x, y), (x + math.cos(ang) * size, y + math.sin(ang) * size)], 2, v)


class InkGather(Attack):
    """The warning before the run's bones (M17-18): drops of ink swirl in from the
    dark beside the canvas, speeding up, and gather where a bone is about to form -
    into a wall's bar just as the bright bead writing it passes their height, or
    into the spot a single bone launches from. White over the dark, never on the
    paper (they only come in from the right). Harmless; decoration only.
    targets: dict(kind="bar", x, y0, y1, paint, write) | dict(kind="pt", x, y, t)."""

    def __init__(self, targets, t_min, seed=0):
        rng = np.random.default_rng(seed)
        self.drops = []                                     # (ts, ta, tx, ty, R, th0, swirl, size)
        for tg in targets:
            if tg["kind"] == "bar":
                n = 14
                for i in range(n):
                    f = (i + rng.uniform(.2, .8)) / n
                    ty = tg["y0"] + (tg["y1"] - tg["y0"]) * f
                    ta = tg["paint"] + tg["write"] * smooth(f)   # meets the bead as it passes
                    self.drops.append(self._drop(rng, ta, tg["x"], ty, t_min, (70, 150), .34))
            else:
                for i in range(6):
                    tx = tg["x"] + rng.uniform(-14, 14)
                    ta = tg["t"] - rng.uniform(0, .02)
                    self.drops.append(self._drop(rng, ta, tx, tg["y"], t_min, (34, 86), .30))
        self.pops = [(tg["t"], tg["x"], tg["y"]) for tg in targets if tg["kind"] == "pt"]
        self.t0 = min(d[0] for d in self.drops) if self.drops else 0.0
        self.t1 = max(d[1] for d in self.drops) + .15 if self.drops else 0.0
        self.impact = None

    @staticmethod
    def _drop(rng, ta, tx, ty, t_min, radii, lead):
        ts = max(t_min, ta - lead - rng.uniform(0, .08))
        th0 = math.radians(rng.uniform(-60, 60))            # from the right only: never over the paper
        swirl = rng.uniform(.25, .5) * rng.choice([-1, 1])
        return (ts, ta, tx, ty, rng.uniform(*radii), th0, swirl, rng.uniform(4.5, 8.0))

    @staticmethod
    def _pos(d, u):
        ts, ta, tx, ty, R, th0, swirl, size = d
        r = R * (1 - u ** 2.2)                               # accelerating in
        th = th0 + swirl * u
        return tx + r * math.cos(th), ty + r * math.sin(th)

    def draw(self, c):
        t = c.t
        for d in self.drops:
            ts, ta = d[0], d[1]
            if not (ts <= t < ta) or ta - ts < .03:
                continue
            u = (t - ts) / (ta - ts)
            g = clamp(u / .25)
            x, y = self._pos(d, u)
            px, py = self._pos(d, max(0.0, u - .12))
            c.line("top", [(px, py), (x, y)], 3, fade_gray(150, g))          # a short wet trail
            c.circle("top", x, y, d[7] * (1 - .35 * u), fade_gray(235, g))
        for tp, x, y in self.pops:                           # the gathered ink sets into the bone
            a = t - tp
            if 0 <= a < .12:
                c.ring("top", x, y, 6 + 22 * out_cubic(a / .12), 2, fade_gray(200, 1 - a / .12))


def draw_orb(c, x, y, t, fires, r_max=15.0, charge=.45):
    """A white orb outside the canvas: pops in `charge` s before its first shot,
    a ring shrinks onto it before every shot, it flashes and recoils as it fires,
    and shrinks away after the last one. fires = [(t_fire, direction)]."""
    ta = fires[0][0] - charge
    grow = out_cubic(clamp((t - ta) / .12))
    shrink = 1 - smooth((t - fires[-1][0] - .08) / .17)
    r = r_max * grow * shrink
    for tf, d in fires:
        a = tf - t
        if 0 < a < charge:
            rr = r_max + (r_max * 1.7) * (a / charge)
            c.line("top", [(x + math.cos(k / 12 * math.tau) * rr, y + math.sin(k / 12 * math.tau) * rr)
                           for k in range(13)], 2, fade_gray(150, 1 - a / charge))
    for tf, d in fires:
        a = t - tf
        if 0 <= a < .12:
            kick = 4 * (1 - a / .12)
            c.circle("top", x - d[0] * kick, y - d[1] * kick, (r + 6 * (1 - a / .12)) * 2, 255)
    if r > .5:
        c.circle("top", x, y, r * 2, 255)
        c.circle("top", x + r * .5, y - r * .9, max(4, r * .4), 255)       # small companion spark


class BoneOrbs(Attack):
    """M19-M20. White orbs appear around the outside of the canvas (a shrinking
    ring charges them - the warning) and on the beat each fires a bone at the
    soul; the bone turns to ink inside the canvas and is soaked up at the far side."""

    def __init__(self, g, shots, speed=340.0, seed=0):
        # shots: [(t_enter, key, (orb x, orb y), aim (x, y))]
        self.g = g
        self.inner = (g.ix0, g.iy0, g.ix1, g.iy1)
        rng = np.random.default_rng(seed)
        self.orbs = {}                                   # orb position -> list of shot times
        self.bones = []
        for t_enter, pos, aim in shots:
            pos = np.asarray(pos, float)
            d = np.asarray(aim, float) - pos
            d /= np.linalg.norm(d)
            start = pos + d * 16
            probe = FlyingBone(start, d, speed, 0.0, 26, self.inner)
            t_go = t_enter - probe.t_in                  # the head enters the paper on the beat
            self.bones.append(FlyingBone(start, d, speed, t_go, 26, self.inner, seed=int(rng.integers(1_000_000))))
            self.orbs.setdefault(tuple(pos), []).append((t_go, d))
        # an orb stays while it keeps firing (shots closer than .7 s share one orb)
        self.spans = []
        for pos, fires in self.orbs.items():
            fires.sort(key=lambda f: f[0])
            cur = [fires[0]]
            for f in fires[1:]:
                if f[0] - cur[-1][0] < .7:
                    cur.append(f)
                else:
                    self.spans.append((pos, cur))
                    cur = [f]
            self.spans.append((pos, cur))
        self.t0 = min(f[0][0] for _, f in self.spans) - .5
        self.t1 = max(b.t_end for b in self.bones)
        self.impact = None

    def draw(self, c):
        t = c.t
        for pos, fires in self.spans:
            if fires[0][0] - .45 <= t < fires[-1][0] + .25:
                draw_orb(c, pos[0], pos[1], t, fires)
        for b in self.bones:
            b.draw(c, t)


class HelixBind(Attack):
    """A pseudo-3D spiral drawn ring by ring from the bottom of the canvas up
    around a column - pale, the warning (back of each ring thin and faint, front
    thicker). At the top it pulls tight in steps and binds whatever is inside:
    the column is deadly while it tightens and every bone caught inside is pulled
    onto the rope with it; then the rope bursts and the bound bones shatter - one
    shard flies straight into the soul (a scripted hit, see choreo.FORCED_HITS)."""

    def __init__(self, cx, y_bot, y_top, R, keys, tighten, t_release, seed=0):
        # keys: [(time, rings drawn)] - e.g. ring completions on the beats
        self.cx, self.yb, self.yt, self.R = cx, y_bot, y_top, R
        self.keys = keys
        self.n = keys[-1][1]
        self.tight = tighten                          # [(time, radius fraction)]
        self.tr = t_release                           # the rope bursts
        self.phi = -math.pi / 2
        self.caught = []                              # (bone, x, y, angle) at the bind
        self.burst = []                               # (unused: the bones shatter now)
        self.hit = None                               # (time, target xy): the shard that hits
        self.t0, self.t1 = keys[0][0], t_release + 1.2
        self.impact = t_release
        self.shake = 4.0
        self.z = 1                                    # drawn over the 3x3 grid and the tunnel

    def bind_prey(self, bones, inner, extra=6, speed=320.0):
        """Catch every bone that is inside the column when it starts to tighten,
        and prepare what the burst throws out (deterministic, done at build time)."""
        tb = self.tight[0][0]
        for b in bones:
            c = b.centre(tb)
            if c is None or abs(c[0] - self.cx) > self.R + 6 or not (self.yt - 8 < c[1] < self.yb + 8):
                continue
            b.captured_at = tb
            self.caught.append((b, float(c[0]), float(c[1]), math.degrees(math.atan2(b.d[0], b.d[1]))))
        # what shatters: the bound bones, and the rope itself along its length
        self.pieces = [y for _, _, y, _ in self.caught] + list(np.linspace(self.yt + 10, self.yb - 10, extra))
        rng = np.random.default_rng(len(self.pieces))
        self.pv = rng.uniform(-1, 1, (len(self.pieces), 3, 4))

    def drawn(self, t):
        ks = self.keys
        if t <= ks[0][0]:
            return 0.0
        for (ta, ua), (tb, ub) in zip(ks, ks[1:]):
            if t < tb:
                return ua + (ub - ua) * (t - ta) / (tb - ta)
        return self.n

    def radius(self, t):
        r = 1.0
        for tk, frac in self.tight:
            if t < tk:
                break
            u = clamp((t - tk) / .05)
            e = 1 + 2.2 * (u - 1) ** 3 + 1.2 * (u - 1) ** 2
            r = r + (frac - r) * e
        return self.R * r

    def draw(self, c):
        t = c.t
        U = self.drawn(t) * math.tau
        if U <= 0:
            return
        rr = self.radius(t)
        ry = max(2.0, rr * .3)
        pitch = (self.yb - self.yt) / self.n
        if t >= self.tr:
            a = t - self.tr
            if a < .2:                                          # the burst itself (pale, harmless)
                for k in range(10):
                    q = k / 10 * math.tau
                    r = 6 + 70 * out_cubic(a / .2)
                    c.circle("art", self.cx + math.cos(q) * r, (self.yt + self.yb) / 2 + math.sin(q) * r * 1.4,
                             6, fade_gray(110, 1 - a / .2))
            # the rope crushes the bones inside into shards of ink - no bone shape left -
            # that fly out with the burst; most of them are live
            # (on the top layer: ink on the paper, white past its edge - they fly far)
            g = 1 - smooth((a - .5) / .35)
            if g > 0:
                K = 1.8
                travel = (1 - math.exp(-K * a)) / K
                for i, y in enumerate(self.pieces):
                    for k in range(3):
                        f = self.pv[i, k]
                        side = 1 if (i + k) % 2 else -1
                        v = np.array([side * (400 + 230 * abs(f[0])), -200 * f[1] - 70])
                        q = np.array([self.cx, y]) + v * travel + np.array([0, 170 * a * a])
                        rot = f[2] * 3 + f[3] * 14 * a
                        sz = 6 + 3 * abs(f[3])
                        tri = [(q[0] + math.cos(rot + o) * sz * (1.4 if o == 0 else .8),
                                q[1] + math.sin(rot + o) * sz * (1.4 if o == 0 else .8)) for o in (0, 2.3, 4.0)]
                        c.poly("top", tri, fade_gray(255, g))
                        if (i * 3 + k) % 5 and g > .5:                 # four in five of them cut
                            c.circle("haz", q[0], q[1], sz * 1.4, 255)
            # ...and one shard flies straight into the soul
            if self.hit is not None:
                th, target = self.hit
                target = np.asarray(target, float)
                start = np.array([self.cx, min(max(target[1], self.yt), self.yb)])
                dur = max(.02, th - self.tr)
                if a <= dur:
                    u = a / dur
                    p = start + (target - start) * u
                    d = target - start
                    d = d / (np.linalg.norm(d) + 1e-9)
                    put_bone_line(c, p - d * 9, p + d * 9, hazard=False)
                elif a < dur + .3:
                    b = (a - dur) / .3
                    for k in range(8):
                        q = k / 8 * math.tau
                        r = 4 + 26 * out_cubic(b)
                        c.circle("top", target[0] + math.cos(q) * r, target[1] + math.sin(q) * r, 5,
                                 fade_gray(230, 1 - b))
            return
        tb, te = self.tight[0][0], self.tight[-1][0] + .05
        for b, x, y, ang in self.caught:                        # bound onto the rope
            if t < tb:
                continue
            u = smooth((t - tb) / (te - tb))
            bx = x + (self.cx - x) * u
            a = math.radians(ang + (0 - ang) * u)
            hx, hy = math.sin(a) * 13, math.cos(a) * 13
            put_bone_line(c, (bx - hx, y - hy), (bx + hx, y + hy))
        solid = t >= self.tight[0][0]
        gone = 0.0
        if solid:
            vf, vb = int(255 * (1 - gone)), int(170 * (1 - gone))
        else:
            blink = t > self.tight[0][0] - .1 and int(t * 60) % 4 < 2
            vf, vb = (112, 60) if blink else (78, 40)
        if vf < 3:
            return
        us = np.arange(0.0, U, .1)
        if len(us) < 2:
            return
        us = np.append(us, U)
        x = self.cx + rr * np.cos(us + self.phi)
        y = self.yb - pitch * us / math.tau + ry * np.sin(us + self.phi)
        front = np.sin(us + self.phi) > 0
        for want, w, v in ((False, 3, vb), (True, 6, vf)):    # back first, then front
            for i in range(len(us) - 1):
                if front[i] == want:
                    c.line("art", [(x[i], y[i]), (x[i + 1], y[i + 1])], w, v)
        if not solid and U < self.n * math.tau:                 # the bead drawing it
            c.circle("art", x[-1], y[-1], 10, 120)
        # hazard: the whole column while it pulls tight, then just the rope
        if solid and t < self.tr:
            tightening = t < self.tight[-1][0] + .06
            half = (rr + 3) if tightening else max(rr, 5) + 2
            c.rect("haz", self.cx - half, self.yt - 6, self.cx + half, self.yb + 6, 255)


class LodgedSpokes(Attack):
    """M24: on each eighth two orbs on opposite sides throw long bones at the
    centre of the canvas, where they stick - eight spokes of ink build up - until
    the break shatters them; the pieces lose their shape and fade (harmless)."""

    def __init__(self, g, pairs, t_shatter, speed=460.0, length=72, seed=0):
        # pairs: [(t_lodge, orb a, orb b)] - the heads meet the centre at t_lodge
        self.C = np.array([g.cx, g.cy])
        self.inner = (g.ix0, g.iy0, g.ix1, g.iy1)
        self.ts, self.v, self.L = t_shatter, speed, length
        self.bones = []
        rng = np.random.default_rng(seed)
        for tl_, *orbs in pairs:
            for pos in orbs:
                pos = np.asarray(pos, float)
                d = self.C - pos
                dist = np.linalg.norm(d)
                d /= dist
                start = pos + d * 16
                stop = dist - 16 - 5                         # head distance from start at rest
                self.bones.append(dict(t=tl_, pos=pos, d=d, start=start, stop=stop,
                                       t_go=tl_ - stop / speed, frag=rng.uniform(-1, 1, 6)))
        self.t0 = min(b["t_go"] for b in self.bones) - .45
        self.t1 = t_shatter + .9
        self.impact = None

    def draw(self, c):
        t = c.t
        for b in self.bones:
            if t < b["t_go"] + .25:
                draw_orb(c, b["pos"][0], b["pos"][1], t, [(b["t_go"], b["d"])])
            if t < b["t_go"]:
                continue
            if t < self.ts:
                head = min(b["stop"], (t - b["t_go"]) * self.v)
                tail = head - self.L
                p0, p1 = b["start"] + b["d"] * tail, b["start"] + b["d"] * head
                put_bone_line(c, p0, p1)
                if (t - b["t"]) >= 0 and t - b["t"] < .1:      # it bites into the paper
                    q = b["start"] + b["d"] * b["stop"]
                    c.circle("art", q[0], q[1], 10 + 20 * (t - b["t"]) / .1, fade_gray(110, 1 - (t - b["t"]) / .1))
                continue
            a = t - self.ts                                    # shattered: harmless melting pieces
            g = 1 - smooth(a / .7)
            mid = b["start"] + b["d"] * (b["stop"] - self.L / 2)
            out = mid - self.C
            out = out / (np.linalg.norm(out) + 1e-9)
            f = b["frag"]
            for k in range(2):
                v = out * (200 + 90 * f[k]) + np.array([f[k + 2] * 70, -60])
                q = mid + v * a + np.array([0, 480 * a * a]) + b["d"] * (k - .5) * 30
                spr = melted_bone(24, math.degrees(f[k + 4] * 10 * a), smooth((a - .02) / .35), k)
                # dim: inside the canvas they read pale (harmless), never black
                c.sprites.append((spr, int(q[0] - spr.w / 2), int(q[1] - spr.h / 2), g * .35, None))


# ======================================================================
# Geometric attacks (M21-M23): big shapes that read at a glance.
class GridFill(Attack):
    """留白 - pale lines split the canvas into a grid, every cell but one is washed
    pale (warning: go to the white cell), then on the beat they flood with ink."""

    def __init__(self, inner, t_hit, safe, cols=3, rows=3, lead=.45, hold=.15, seed=0):
        self.inner, self.th, self.safe = inner, t_hit, safe
        self.cols, self.rows, self.lead, self.hold = cols, rows, lead, hold
        self.t0, self.t1 = t_hit - lead, t_hit + hold + .25
        self.impact = None

    def cells(self):
        ix0, iy0, ix1, iy1 = self.inner
        cw, ch = (ix1 - ix0) / self.cols, (iy1 - iy0) / self.rows
        for i in range(self.cols):
            for j in range(self.rows):
                if (i, j) != tuple(self.safe):
                    yield ix0 + i * cw, iy0 + j * ch, ix0 + (i + 1) * cw, iy0 + (j + 1) * ch

    def draw(self, c):
        t = c.t
        ix0, iy0, ix1, iy1 = self.inner
        a = t - self.th
        k = clamp((a + self.lead) / .1)
        lines = int(70 * (1 - smooth((a - self.hold) / .2))) if a > 0 else 70
        for i in range(1, self.cols):                         # grid lines drawn across
            x = ix0 + (ix1 - ix0) * i / self.cols
            c.line("art", [(x, iy0), (x, iy0 + (iy1 - iy0) * k)], 2, lines)
        for j in range(1, self.rows):
            y = iy0 + (iy1 - iy0) * j / self.rows
            c.line("art", [(ix0, y), (ix0 + (ix1 - ix0) * k, y)], 2, lines)
        if a < 0:
            if a < -self.lead + .1:
                return
            late = a > -.1
            val = (96 if int(t * 60) % 4 < 2 else 60) if late else 52
            for x0, y0, x1, y1 in self.cells():
                c.rect("art", x0 + 3, y0 + 3, x1 - 3, y1 - 3, val)
            return
        if a < self.hold:                                     # flood
            g = out_cubic(clamp(a / .05))
            for x0, y0, x1, y1 in self.cells():
                mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                hw, hh = (x1 - x0) / 2 * (.55 + .45 * g), (y1 - y0) / 2 * (.55 + .45 * g)
                c.rect("art", mx - hw + 1, my - hh + 1, mx + hw - 1, my + hh - 1, 255)
                c.rect("haz", mx - hw + 2, my - hh + 2, mx + hw - 2, my + hh - 2, 255)
            return
        f = 1 - smooth((a - self.hold) / .22)                 # the ink drains away
        for x0, y0, x1, y1 in self.cells():
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            hw, hh = (x1 - x0) / 2 * f, (y1 - y0) / 2 * f
            if hw > 1:
                c.rect("art", mx - hw, my - hh, mx + hw, my + hh, int(96 * f))


class GridCollapse(Attack):
    """切纸塌陷 - the 留白 grid as cut paper. Four blades fly in from off screen
    (white streaks over the dark, dark cuts over the paper) and slice the canvas
    3x3: the two upright cuts four sixteenths before the hit, the two level cuts
    three before. Every cell but the soul's is washed pale as it is cut, sags and
    trembles from two sixteenths before (the gap round it darkens), flickers at
    the end - then on the hit the eight pieces drop away, tumbling in perspective
    into the black below: the holes are the hazard, only the soul's piece is left.
    After `hold` ink floods back into the holes (grey, harmless) and the paper
    heals from the rims inwards. Same hazard cells and timing as GridFill."""

    FALL = .32          # s for a piece to tumble away
    HEAL = .2           # s for the holes to close again

    def __init__(self, inner, t_hit, safe, step, vp, cols=3, rows=3, hold=.15, seed=0):
        self.inner, self.th, self.safe = inner, t_hit, tuple(safe)
        self.step, self.vp = step, np.asarray(vp, float)
        self.cols, self.rows, self.hold = cols, rows, hold
        self.t_v, self.t_h, self.t_sag = t_hit - 4 * step, t_hit - 3 * step, t_hit - 2 * step
        self.t0, self.t1 = self.t_v - .02, t_hit + hold + self.HEAL + .02
        self.impact = None
        rng = np.random.default_rng(seed)
        self.tumble = {}                                       # per piece: axis angle, turn, spin sign
        for ij in self.doomed():
            self.tumble[ij] = (rng.uniform(0, math.pi), rng.uniform(.9, 1.5) * rng.choice([-1, 1]),
                               rng.uniform(.5, 1.0))
        small = rng.random((6, 10))                            # ragged heal front
        self.noise = np.asarray(Image.fromarray((small * 255).astype(np.uint8)).resize(
            (int(inner[2] - inner[0]) + 2, int(inner[3] - inner[1]) + 2), Image.Resampling.BICUBIC), float) / 255

    def doomed(self):
        return [(i, j) for i in range(self.cols) for j in range(self.rows) if (i, j) != self.safe]

    def cell(self, ij):
        ix0, iy0, ix1, iy1 = self.inner
        cw, ch = (ix1 - ix0) / self.cols, (iy1 - iy0) / self.rows
        return ix0 + ij[0] * cw, iy0 + ij[1] * ch, ix0 + (ij[0] + 1) * cw, iy0 + (ij[1] + 1) * ch

    # ---------------------------------------------------------------- blades
    def cuts(self):
        """(time, kind, position, direction) of the four blades."""
        ix0, iy0, ix1, iy1 = self.inner
        xs = [ix0 + (ix1 - ix0) * i / self.cols for i in range(1, self.cols)]
        ys = [iy0 + (iy1 - iy0) * j / self.rows for j in range(1, self.rows)]
        out = [(self.t_v, "v", x, 1 if n % 2 == 0 else -1) for n, x in enumerate(xs)]
        out += [(self.t_h, "h", y, 1 if n % 2 == 0 else -1) for n, y in enumerate(ys)]
        return out

    SWEEP = .07         # s for a blade to cross the screen
    SPAN = (-80.0, 720.0)

    def blade_seg(self, kind, pos, d, u):
        """The cut so far (u: 0..1 of the sweep) as a world segment, and the head."""
        a, b = self.SPAN
        if d < 0:
            a, b = b, a
        head = a + (b - a) * u
        if kind == "v":
            return (pos, a), (pos, head)
        return (a, pos), (head, pos)

    def draw(self, c):
        t, k = c.t, c.k
        if t < self.t0 or t >= self.t1:
            return
        a = t - self.th
        Wl, Hl = c.size
        mine = Image.new("L", (Wl, Hl))
        dr = ImageDraw.Draw(mine)
        P = lambda pts: [(x / k, y / k) for x, y in pts]

        # cut lines left behind by the blades (pale, harmless), fading after the hold
        fade = 1 - smooth((a - self.hold) / .2) if a > self.hold else 1.0
        heads, segs = [], []
        for tc, kind, pos, d in self.cuts():
            u = (t - tc) / self.SWEEP
            if u < 0:
                continue
            segs.append(self.blade_seg(kind, pos, d, min(1.0, u)))
            if u < 1.25:
                heads.append((kind, pos, d, u))

        def cut_lines():
            if fade > 0 and segs:
                ln = Image.new("L", (Wl, Hl))
                ld = ImageDraw.Draw(ln)
                for p0, p1 in segs:
                    ld.line(P([p0, p1]), fill=int(84 * fade), width=max(1, round(2 / k)))
                mine.paste(Image.fromarray(np.maximum(np.asarray(mine), np.asarray(ln))))

        doomed = self.doomed()
        if a < 0:                                              # cut, washed, sagging
            if t >= self.t_v:
                wash = int(52 * clamp((t - self.t_v) / .05))
                sag = in_quad(clamp((t - self.t_sag) / (2 * self.step)))
                late = a > -.1
                f = int(t * 60)
                for n, ij in enumerate(doomed):
                    x0, y0, x1, y1 = self.cell(ij)
                    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                    if sag > 0:                                # the gap round a sinking piece
                        dr.rectangle((x0 / k, y0 / k, x1 / k - .01, y1 / k - .01), fill=int(60 + 70 * sag))
                    s = 1 - .06 * sag
                    jx = ((f * 7 + n * 13) % 3 - 1) * sag * 1.5
                    jy = ((f * 11 + n * 5) % 3 - 1) * sag * 1.5
                    hw, hh = (x1 - x0) / 2 * s - 1.5, (y1 - y0) / 2 * s - 1.5
                    val = (96 if f % 4 < 2 else 60) if late else int(wash + 18 * sag)
                    dr.rectangle(((mx - hw + jx) / k, (my - hh + jy) / k,
                                  (mx + hw + jx) / k - .01, (my + hh + jy) / k - .01), fill=val)
            cut_lines()
            self._blades(c, heads)
            self._merge(c, mine, None)
            return

        cut_lines()                                            # (the void swallows them)
        # the holes: every doomed cell, as one void
        hole = Image.new("L", (Wl, Hl))
        hd = ImageDraw.Draw(hole)
        for ij in doomed:
            x0, y0, x1, y1 = self.cell(ij)
            hd.rectangle((x0 / k, y0 / k, x1 / k - .01, y1 / k - .01), fill=255)
        hm = np.asarray(hole) > 0
        m = np.asarray(mine).copy()
        hz = None
        if a < self.hold:                                      # open: black void, pieces tumbling away
            m[hm] = 255
            pieces = Image.new("L", (Wl, Hl))
            pd = ImageDraw.Draw(pieces)
            u = clamp(a / self.FALL)
            depth = 1 - (1 - u) ** 2                           # pulled away at once, then sinking on
            f = 420.0                                          # focal length (px)
            z = 3.0 * f * depth
            sc = f / (f + z)
            for ij in doomed:
                x0, y0, x1, y1 = self.cell(ij)
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                ax, turn, spin = self.tumble[ij]
                cx += (cx - self.vp[0]) * .7 * depth           # the pieces part as they drop
                cy += (cy - self.vp[1]) * .7 * depth
                ang = turn * depth * 1.25                      # tumbling about an axis in the paper
                axis = np.array([math.cos(ax), math.sin(ax), 0.0])
                hw, hh = (x1 - x0) / 2 - 1.5, (y1 - y0) / 2 - 1.5
                corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
                pts = []
                for dx, dy in corners:
                    v = np.array([dx, dy, 0.0])
                    v = (v * math.cos(ang) + np.cross(axis, v) * math.sin(ang)
                         + axis * axis.dot(v) * (1 - math.cos(ang)))
                    s = f / (f + z + v[2])
                    px = self.vp[0] + (cx + v[0] - self.vp[0]) * s
                    py = self.vp[1] + (cy + v[1] - self.vp[1]) * s
                    pts.append((px / k, py / k))
                facing = .5 + .5 * abs(math.cos(ang))
                shade = 236 * sc ** .45 * facing               # paper going dark as it falls away
                pd.polygon(pts, fill=int(max(1, min(254, 255 - shade))),
                           outline=int(max(1, min(254, 255 - .55 * shade))))   # its cut edge
            pa = np.asarray(pieces)
            on = hm & (pa > 0)
            m[on] = pa[on]
            from scipy import ndimage as ndi
            hz = ndi.binary_erosion(hm & ~on, iterations=max(1, round(2 / k)))
        else:                                                  # ink floods back; the paper heals inwards
            p = smooth((a - self.hold) / self.HEAL)
            from scipy import ndimage as ndi
            dist = ndi.distance_transform_edt(hm)
            dmax = max(1.0, dist.max())
            ix0, iy0 = int(self.inner[0] / k), int(self.inner[1] / k)
            nz = np.zeros((Hl, Wl))
            nk = self.noise[::k, ::k]
            ys, xs = slice(iy0, min(Hl, iy0 + nk.shape[0])), slice(ix0, min(Wl, ix0 + nk.shape[1]))
            nz[ys, xs] = nk[:ys.stop - ys.start, :xs.stop - xs.start]
            wet = hm & (dist / dmax + .35 * nz > p * 1.4)
            m[wet] = int(150 - 80 * p)
        self._blades(c, heads)
        self._merge(c, Image.fromarray(m), hz)

    def _merge(self, c, mine, hz):
        """Max-blend into the art layer, so ink drawn earlier (the tunnel) stays on top."""
        art = np.asarray(c.art).copy()
        np.maximum(art, np.asarray(mine), out=art)
        c.art.paste(Image.fromarray(art))
        if hz is not None and hz.any():
            h = np.asarray(c.haz).copy()
            h[hz] = 255
            c.haz.paste(Image.fromarray(h))

    def _blades(self, c, heads):
        """The flying blades on the free layer: a white streak over the dark; over
        the paper (where that layer reads inverted) a dark-grey cut, never black."""
        if not heads:
            return
        import scene as S
        k = c.k
        pm = S.paper_mask(c.t, c.box)[0][::k, ::k]
        Wl, Hl = c.size
        core, glow = Image.new("L", (Wl, Hl)), Image.new("L", (Wl, Hl))
        dc, dg = ImageDraw.Draw(core), ImageDraw.Draw(glow)
        for kind, pos, d, u in heads:
            if u > 1:
                continue
            _, head = self.blade_seg(kind, pos, d, u)
            tail = 150.0 * d
            hx, hy = head
            tx, ty = (hx, hy - tail) if kind == "v" else (hx - tail, hy)
            dg.line([(tx / k, ty / k), (hx / k, hy / k)], fill=255, width=max(1, round(8 / k)))
            dc.line([(tx / k, ty / k), (hx / k, hy / k)], fill=255, width=max(1, round(3 / k)))
        cm, gm = np.asarray(core) > 0, np.asarray(glow) > 0
        top = np.asarray(c.top).copy()
        g_only = gm & ~cm
        top[g_only & ~pm] = 110                                # dim glow over the dark
        top[g_only & pm] = 255 - 190                           # light grey over the paper
        top[cm & ~pm] = 255                                    # white streak over the dark
        top[cm & pm] = 255 - 95                                # dark grey cut over the paper
        c.top.paste(Image.fromarray(top))


class SquareTunnel(Attack):
    """Square frames rush out of a vanishing point in the middle of the canvas
    towards the camera - a tunnel we fly through. Each frame has a wedge cut out;
    the soul must sit in the cut when the frame sweeps past it. Inside the canvas
    the frames are ink, past its edge they fly on as white squares."""

    def __init__(self, center, frames, r_pass=60.0, T=.32, gap=110.0, t_show=0.0, seed=0):
        # frames: [(t_pass, gap_angle_deg, spin_deg[, gate])] - half-size r_pass at t_pass;
        # a gate (strong beat) is heavier with a narrower cut. Drawn on the paper only.
        self.C = np.asarray(center, float)
        self.frames, self.rp, self.T, self.gap = frames, r_pass, T, math.radians(gap / 2)
        self.t0 = max(t_show, min(f[0] for f in frames) - 3.2 * T)
        self.t1 = max(f[0] for f in frames) + .9 * T
        self.impact = None

    def draw(self, c):
        t = c.t
        if t < self.t0:
            return
        for fr in self.frames:
            tp, ga, spin = fr[:3]
            gate = len(fr) > 3 and fr[3]
            Cf = np.asarray(fr[4], float) if len(fr) > 4 else self.C   # its own vanishing point
            z = 1 + (tp - t) / self.T
            if z > 4.2 or z < .14:
                continue
            s = self.rp / z                                   # half size (perspective)
            w = float(np.clip((11 if gate else 7) / z, 2, 20))
            half_gap = self.gap * (.72 if gate else 1.0)
            ang = math.radians(spin) * (1 - z) * .5            # turns a little as it nears
            g0 = math.radians(ga)
            ca, sa = math.cos(ang), math.sin(ang)
            pts = []
            n = 64
            for i in range(n + 1):
                u = i / n * 4                                  # walk the square's perimeter
                side, f = int(u) % 4, u - int(u)
                q = [(-1 + 2 * f, -1), (1, -1 + 2 * f), (1 - 2 * f, 1), (-1, 1 - 2 * f)][side]
                x, y = q[0] * s, q[1] * s
                x, y = x * ca - y * sa, x * sa + y * ca
                a = math.atan2(y, x)
                cut = abs((a - g0 + math.pi) % math.tau - math.pi) < half_gap
                pts.append(None if cut else (Cf[0] + x, Cf[1] + y))
            # far frames are a pale tunnel (harmless); they turn to solid ink as
            # they come close (half-size ~40 px) - only black hurts
            near = z <= 1.5
            val = 255 if near else int(min(1.0, (4.2 - z) / 1.0) * (110 if z < 2.2 else 80))
            if s > 110:                                       # past the arena it thins away
                val = int(val * clamp(1 - (s - 110) / 150))
                w = min(w, 10.0)
                if val < 4:
                    continue
            for p, q in zip(pts, pts[1:]):
                if p is not None and q is not None:
                    c.line("art", [p, q], w, val)
                    if near:
                        c.line("haz", [p, q], w, 255)


class WireCube(Attack):
    """A wireframe cube in perspective in the middle of the canvas: sketched pale
    first, then inked, and on every step it snaps round - its edges sweep the
    canvas. Back edges thin, front edges heavy."""

    def __init__(self, center, size, t_sketch, t_ink, steps, t_end, D=320.0, seed=0):
        # steps: [(time, yaw_deg, pitch_deg)] - it snaps to each pose on the beat
        self.C, self.h, self.D = np.asarray(center, float), size / 2, D
        self.ts, self.ti, self.steps, self.te = t_sketch, t_ink, steps, t_end
        self.t0, self.t1 = t_sketch, t_end + .3
        v = np.array([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
        self.V = v
        self.E = [(i, j) for i in range(8) for j in range(i + 1, 8) if np.abs(v[i] - v[j]).sum() == 2]
        self.impact = None

    def pose(self, t):
        yaw, pitch = self.steps[0][1], self.steps[0][2]
        for tk, yk, pk in self.steps:
            if t < tk:
                break
            u = clamp((t - tk) / .06)
            e = 1 + 2.2 * (u - 1) ** 3 + 1.2 * (u - 1) ** 2
            yaw, pitch = yaw + (yk - yaw) * e, pitch + (pk - pitch) * e
        return math.radians(yaw), math.radians(pitch)

    def draw(self, c):
        t = c.t
        if t > self.te + .3:
            return
        yaw, pitch = self.pose(t)
        cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
        P = []
        for x, y, z in self.V * self.h:
            x, z = x * cy + z * sy, -x * sy + z * cy
            y, z = y * cp - z * sp, y * sp + z * cp
            k = self.D / (self.D + z)
            P.append((self.C[0] + x * k, self.C[1] + y * k, z))
        inked = t >= self.ti
        gone = smooth((t - self.te) / .25)
        draw_k = clamp((t - self.ts) / .25)
        for n_e, (i, j) in enumerate(sorted(self.E, key=lambda e: -(P[e[0]][2] + P[e[1]][2]))):
            if n_e >= int(draw_k * 12 + .999):
                continue
            zc = (P[i][2] + P[j][2]) / 2
            front = zc < 0
            if inked:
                val = int((255 if front else 175) * (1 - gone))
                w = 7 if front else 4
            else:
                late = t > self.ti - .1 and int(t * 60) % 4 < 2
                val = (110 if late else 74) if front else (64 if late else 42)
                w = 5 if front else 3
            if val < 3:
                continue
            a, b = (P[i][0], P[i][1]), (P[j][0], P[j][1])
            c.line("art", [a, b], w, val)
            if inked and gone < .5:
                c.line("haz", [a, b], w, 255)


def _homography(src, dst):
    """3x3 matrix taking the four src points onto the four dst points."""
    A, b = [], []
    for (u, v), (x, y) in zip(src, dst):
        A.append([u, v, 1, 0, 0, 0, -u * x, -v * x])
        A.append([0, 0, 0, u, v, 1, -u * y, -v * y])
        b += [x, y]
    h = np.linalg.solve(np.array(A, float), np.array(b, float))
    return np.append(h, 1.0).reshape(3, 3)


def warp_sprite(value, alpha, quad):
    """A small image mapped (nearest neighbour) onto a screen quad - the image's
    corners TL, TR, BR, BL land on quad[0..3]. Returns (Sprite, x0, y0) or None."""
    from common import Sprite
    h, w = alpha.shape
    quad = np.asarray(quad, float)
    x0, y0 = np.floor(quad.min(0)).astype(int)
    x1, y1 = np.ceil(quad.max(0)).astype(int)
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    try:
        Hinv = np.linalg.inv(_homography([(0, 0), (w, 0), (w, h), (0, h)], quad))
    except np.linalg.LinAlgError:
        return None                                       # seen exactly edge-on
    yy, xx = np.mgrid[y0:y1, x0:x1] + .5
    q = Hinv @ np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size)])
    with np.errstate(divide="ignore", invalid="ignore"):
        u, v = q[0] / q[2], q[1] / q[2]
    ok = np.isfinite(u) & np.isfinite(v) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    ui, vi = np.where(ok, u, 0).astype(int), np.where(ok, v, 0).astype(int)
    a = (ok & alpha[vi, ui]).reshape(xx.shape)
    if not a.any():
        return None
    val = value[vi, ui].reshape(xx.shape).astype(np.uint8)
    return Sprite(val, a), int(x0), int(y0)


class CanvasCube(Attack):
    """The soul has been inside a rectangle all along: it is inked in (the reveal),
    then the rectangle grows depth and becomes a box seen from a fixed 3D angle,
    and the soul goes 3D with it - it rides the front face. Bones spear through
    the box, aimed at the soul: from the left/right across the front face at its
    height, from top/bottom at its column, from the front/back straight through
    it (a pale line or ring warns first). A spear is black while it moves and
    just after it bites, then pale (harmless) and it melts away. Then the box
    turns, a slash cutting it at every turn, and bursts.

    Hazards are drawn in the face's own (unturned) coordinates - the same ones the
    soul is planned in - while everything you see is turned and projected."""

    EDGES = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]

    def __init__(self, inner, t_reveal, t_extrude, steps, bones, t_burst, D=420.0, seed=0, slashes=()):
        x0, y0, x1, y1 = inner
        self.C = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        self.w, self.h = x1 - x0, y1 - y0
        self.dmax = min(self.w, self.h)
        self.tr, self.tx, self.steps, self.tb, self.D = t_reveal, t_extrude, steps, t_burst, D
        # bones: [(t_hit, face, aim (x, y) in screen/plan coords)] face in +x -x +y -y +z -z
        self.bones = []
        for th, face, aim in bones:
            ax = float(np.clip(aim[0] - self.C[0], -self.w / 2 + 12, self.w / 2 - 12))
            ay = float(np.clip(aim[1] - self.C[1], -self.h / 2 + 12, self.h / 2 - 12))
            self.bones.append((th, face, ax, ay))
        self.slashes = list(slashes)                     # [(t_hit, face, diagonal 0/1)]
        rng = np.random.default_rng(seed)
        self.frag = rng.uniform(-1, 1, (40, 4))
        self.t0, self.t1 = t_reveal, t_burst + .8
        self.impact = t_burst
        self.shake = 6.0

    # ---------------------------------------------------------------- geometry
    def pose(self, t):
        yaw, pitch = self.steps[0][1], self.steps[0][2]
        for tk, yk, pk in self.steps:
            if t < tk:
                break
            u = clamp((t - tk) / .07)
            e = 1 + 2.2 * (u - 1) ** 3 + 1.2 * (u - 1) ** 2
            yaw, pitch = yaw + (yk - yaw) * e, pitch + (pk - pitch) * e
        return math.radians(yaw), math.radians(pitch)

    def depth(self, t):
        return self.dmax * smooth((t - self.tx) / .1)

    def project(self, p, t):
        """Box-local point (x, y about the centre; z = 0 on the front face, +z back)."""
        yaw, pitch = self.pose(t)
        d = self.depth(t)
        x, y, z = p[0], p[1], p[2] - d / 2
        cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
        x, z = x * cy + z * sy, -x * sy + z * cy
        y, z = y * cp - z * sp, y * sp + z * cp
        z += d / 2
        k = self.D / max(60.0, self.D + z)
        return np.array([self.C[0] + x * k, self.C[1] + y * k]), z, k

    def soul_view(self, xy, t):
        """Where the (planned, face-local) soul appears, and how large (k)."""
        if not (self.tx <= t < self.tb):
            return np.asarray(xy, float), 1.0
        p, _, k = self.project((xy[0] - self.C[0], xy[1] - self.C[1], 0.0), t)
        return p, k

    SOUL_T = 10.0       # the 3D soul's thickness (a heart-shaped token on the front face)

    def soul_layers(self, xy, t):
        """The soul as a solid heart lying on the front face, seen in the box's own
        perspective: one slice per pixel of thickness, each the heart mapped onto
        its plane, dark sides and a lit cap. [(Sprite, x, y)] far to near."""
        from common import Sprite
        src = sprite("player.png")
        mask = src.a
        cap = src.v.copy()
        cap[3:6, 3:5] = 200                                   # a glint on the left lobe
        cap[2, 4], cap[6, 3], cap[3, 5] = 150, 150, 150
        sx, sy = xy[0] - self.C[0], xy[1] - self.C[1]
        sq = [(-8, -8), (8, -8), (8, 8), (-8, 8)]
        n = int(self.SOUL_T)
        layers = []
        for i in range(n + 1):
            z = self.SOUL_T / 2 - i                           # back (into the face) to front
            P = [self.project((sx + u, sy + v, z), t) for u, v in sq]
            quad = np.array([p for p, _, _ in P])
            zr = float(np.mean([zz for _, zz, _ in P]))
            front = i == n
            val = cap if front else np.full_like(src.v, 18 + 2 * i)
            spr = warp_sprite(val, mask, quad)
            if spr is not None:
                layers.append((zr, i, spr))
        layers.sort(key=lambda e: (-e[0], e[1]))              # far first (turned away: the cap is behind)
        return [(s, x0, y0) for _, _, (s, x0, y0) in layers]

    def corners(self, t):
        d = self.depth(t)
        hw, hh = self.w / 2, self.h / 2
        return [(sx * hw, sy * hh, sz * d) for sx in (-1, 1) for sy in (-1, 1) for sz in (0, 1)]

    FLY = .2            # a bone's flight (it bites at the end of it)
    LIVE = .12          # ...and is still black this long after the bite
    STUCK = .42         # once it has bitten in it turns pale (harmless) and stays stuck in the box

    def bone_line(self, face, ax, ay, a, d):
        """The spear's (tail, head) in box coords, `a` s after it set off. `face` is
        the side it comes in from: -x left, +x right, -y top, +y bottom, -z front."""
        u = clamp(a / self.FLY)
        hw, hh = self.w / 2, self.h / 2
        if face[1] in "xy":                                  # flies right across the face, sticks in the far side
            ax_i = 0 if face[1] == "x" else 1
            half = hw if ax_i == 0 else hh
            sg = 1 if face[0] == "-" else -1                 # direction of travel
            L = 72.0
            head_s = -sg * (half + 6) + sg * (2 * half + 28) * u
            head = np.array([ax, ay, 0.0])
            head[ax_i] = head_s
            tail = head.copy()
            tail[ax_i] -= sg * L
        elif face == "-z":                                   # from the front, straight through the face
            head = np.array([ax, ay, -190 + 216 * u])
            tail = head - np.array([0, 0, 76.0])
        else:                                                # from the back
            head = np.array([ax, ay, d + 190 - (d + 216) * u])
            tail = head + np.array([0, 0, 76.0])
        return tail, head

    def bone_hazard(self, c, face, ax, ay, tail, head):
        """The spear where it crosses the front face (z = 0), in face-local coords."""
        hw, hh = self.w / 2, self.h / 2
        if face[1] == "x":
            x0, x1 = sorted((tail[0], head[0]))
            x0, x1 = max(x0, -hw), min(x1, hw)
            if x1 > x0:
                c.line("haz", [self.C + (x0, ay), self.C + (x1, ay)], 12, 255)
        elif face[1] == "y":
            y0, y1 = sorted((tail[1], head[1]))
            y0, y1 = max(y0, -hh), min(y1, hh)
            if y1 > y0:
                c.line("haz", [self.C + (ax, y0), self.C + (ax, y1)], 12, 255)
        else:
            z0, z1 = sorted((tail[2], head[2]))
            if z0 <= 0 <= z1:
                q = self.C + np.array([ax, ay])
                c.circle("haz", q[0], q[1], 26, 255)

    def face_diag(self, face, diag, d):
        """A slash across one face: (start, end) in box coords, inset from the corners."""
        hw, hh, i = self.w / 2 * .78, self.h / 2 * .78, .78
        sgn = 1 if diag else -1
        if face == "-z":
            return np.array([-hw, -hh * sgn, 0.0]), np.array([hw, hh * sgn, 0.0])
        if face == "+z":
            return np.array([-hw, hh * sgn, d]), np.array([hw, -hh * sgn, d])
        zc, zr = d / 2, d / 2 * i
        if face in ("+x", "-x"):
            x = self.w / 2 * (1 if face == "+x" else -1)
            return np.array([x, -hh * sgn, zc - zr]), np.array([x, hh * sgn, zc + zr])
        y = self.h / 2 * (1 if face == "+y" else -1)
        return np.array([-hw, y, zc - zr * sgn]), np.array([hw, y, zc + zr * sgn])

    # ---------------------------------------------------------------- drawing
    def draw(self, c):
        t = c.t
        if t < self.tr:
            return
        if t >= self.tb:
            return self.draw_burst(c, t)
        P = [self.project(p, t) for p in self.corners(t)]
        if t < self.tx:
            # the rectangle is simply there - it shows up only from the moment the
            # camera is deep enough inside it that no side can be in the frame, so
            # the pull-back finds it as if it had always been there
            quad = [P[0][0], P[4][0], P[6][0], P[2][0]]
            c.line("art", quad + [quad[0]], 6, 255)
            self.draw_bones(c, t, 0.0)
            return
        d = self.depth(t)
        # the soul's plane is the front face: shade it very lightly so it reads as a surface
        front = [P[0][0], P[4][0], P[6][0], P[2][0]]
        c.poly("art", front, 22)
        self.draw_slashes(c, t)                               # (over the shading of the front face)
        order = sorted(self.EDGES, key=lambda e: -(P[e[0]][1] + P[e[1]][1]))
        for i, j in order:
            fr = (P[i][1] + P[j][1]) / 2 < d / 2
            c.line("art", [P[i][0], P[j][0]], 7 if fr else 4, 255 if fr else 175)
        self.draw_bones(c, t, d)

    def draw_bones(self, c, t, d):
        for th, face, ax, ay in self.bones:
            a = t - (th - self.FLY)
            if a < 0:
                if a > -.36:                                  # the warning: where it will run / strike
                    blink = a > -.14 and int(t * 60) % 4 < 2
                    val = 120 if blink else 70
                    if face[1] in "xy":
                        hw, hh = self.w / 2, self.h / 2
                        if face[1] == "x":
                            q0, q1 = (-hw, ay, 0.0), (hw, ay, 0.0)
                        else:
                            q0, q1 = (ax, -hh, 0.0), (ax, hh, 0.0)
                        n = 12
                        pts = [self.project(np.array(q0) + (np.array(q1) - np.array(q0)) * i / n, t) for i in range(n + 1)]
                        for (p0, _, k0), (p1, _, _) in zip(pts, pts[1:]):
                            c.line("art", [p0, p1], 9 * k0, val)
                        # an arrowhead at the side it comes in from
                        e, _, k = pts[0] if face[0] == "-" else pts[-1]
                        f, _, _ = pts[1] if face[0] == "-" else pts[-2]
                        dv = (f - e) / (np.linalg.norm(f - e) + 1e-9)
                        nv = np.array([-dv[1], dv[0]])
                        c.poly("art", [e + dv * 16 * k, e - nv * 9 * k, e + nv * 9 * k], val)
                    else:
                        pq, _, k = self.project((ax, ay, 0.0), t)
                        c.circle("art", pq[0], pq[1], 30 * k, val)
                        c.circle("art", pq[0], pq[1], 16 * k, 0)
                continue
            tail, head = self.bone_line(face, ax, ay, a, d)
            bite = a - self.FLY                               # s since it bit in
            live = bite < self.LIVE
            # black while it is live, then pale - but it stays where it went in,
            # a solid bone standing in the box (turning with it) until the burst
            gain = 1.0 if live else 1.0 + (self.STUCK - 1.0) * smooth((bite - self.LIVE) / .15)
            self.bone_3d(c, tail, head, face, t, gain)
            if 0 <= bite < .12:                               # it bites in
                hq = head if face[1] in "xy" else np.array([ax, ay, 0.0])
                pq, _, k = self.project(hq if face[1] == "z" else (head[0], head[1], 0.0), t)
                c.circle("art", pq[0], pq[1], (10 + 30 * bite / .12) * k, fade_gray(110, 1 - bite / .12))
            if live:                                          # hazard in the face's own plane
                self.bone_hazard(c, face, ax, ay, tail, head)

    BONE_R = 3.0        # shaft radius
    KNOB_R = 3.2        # each of the two knobs at either end (they stay inside the hazard's width)
    KNOB_OFF = 2.4

    def bone_3d(self, c, tail, head, face, t, gain):
        """A bone as a solid in the box's perspective: a shaft that thins with
        distance between two double knobs, lit from the upper left. The far end
        is drawn first, the near end over it."""
        tail, head = np.asarray(tail, float), np.asarray(head, float)
        L = float(np.linalg.norm(head - tail))
        if L < 1e-6:
            return
        u = (head - tail) / L
        n = np.array([0.0, 1.0, 0.0]) if face[1] in "xz" else np.array([1.0, 0.0, 0.0])
        ends = []
        for e, inward in ((tail, u), (head, -u)):
            cen = e + inward * 2.5
            pe, ze, ke = self.project(e, t)
            lobes = [self.project(cen + n * s * self.KNOB_OFF, t) for s in (1, -1)]
            ends.append((ze, pe, ke, lobes))
        body, lit = int(255 * gain), int(150 * gain)

        def knob(e):
            for q, _, k in e[3]:
                c.circle("art", q[0], q[1], 2 * self.KNOB_R * k, body)
            for q, _, k in e[3]:
                r = self.KNOB_R * k
                c.circle("art", q[0] - r * .35, q[1] - r * .35, r * .7, lit)

        far, near = sorted(ends, key=lambda e: -e[0])
        knob(far)
        (_, pt, kt, _), (_, ph, kh, _) = ends
        dv = ph - pt
        ln = float(np.linalg.norm(dv))
        if ln > 1:
            nv = np.array([-dv[1], dv[0]]) / ln
            wt, wh = self.BONE_R * kt, self.BONE_R * kh
            c.poly("art", [pt + nv * wt, ph + nv * wh, ph - nv * wh, pt - nv * wt], body)
            s = 1 if nv @ np.array([-1.0, -1.0]) > 0 else -1     # the side facing the light
            a_, b_ = pt + dv * .12, ph - dv * .12
            c.line("art", [a_ + nv * s * wt * .45, b_ + nv * s * wh * .45], max(1.0, (wt + wh) * .3), lit)
        knob(near)

    def draw_slashes(self, c, t):
        d = self.depth(t)
        for th, face, diag in self.slashes:
            a = t - th
            if a < -.38:
                continue
            p0, p1 = self.face_diag(face, diag, d)
            n = 16
            pts = [self.project(p0 + (p1 - p0) * i / n, t) for i in range(n + 1)]
            if a < 0:                                         # a clear wash of ink first
                late = a > -.14
                val = (150 if int(t * 60) % 4 < 2 else 96) if late else 100
                for (q0, _, k0), (q1, _, _) in zip(pts, pts[1:]):
                    c.line("art", [q0, q1], 12 * k0, val)
                continue
            m = max(1, int(n * clamp(a / .05)))
            for i in range(m):
                (q0, _, k0), (q1, _, _) = pts[i], pts[i + 1]
                w = 11 * k0 * (1.15 - .5 * i / n)
                c.line("art", [q0, q1], w, 255)
                c.circle("art", q0[0], q0[1], w, 255)
            if face == "-z":                                  # on the soul's own face: it cuts
                c.line("haz", [self.C + p0[:2], self.C + p1[:2]], 10, 255)

    def draw_burst(self, c, t):
        a = t - self.tb
        g = 1 - smooth(a / .6)
        if g <= 0:
            return
        tf = self.tb - 1e-3
        P = [self.project(p, tf) for p in self.corners(tf)]
        for k, (i, j) in enumerate(self.EDGES):
            m = (P[i][0] + P[j][0]) / 2
            out = m - self.C
            out = out / (np.linalg.norm(out) + 1e-9)
            f = self.frag[k]
            off = out * (260 + 120 * f[0]) * a + np.array([f[1] * 60, 380 * a * a])
            rot = f[2] * 6 * a
            cr, sr = math.cos(rot), math.sin(rot)
            pts = []
            for q in (P[i][0], P[j][0]):
                r = q - m
                pts.append(m + off + np.array([r[0] * cr - r[1] * sr, r[0] * sr + r[1] * cr]) * (1 - .5 * a))
            c.line("art", pts, 4, fade_gray(110, g))               # pale: harmless
        for n, (th, face, ax, ay) in enumerate(self.bones):          # spears still in it shatter too
            tail, head = self.bone_line(face, ax, ay, 1.0, self.depth(tf))
            (pt, _, _), (ph, _, _) = self.project(tail, tf), self.project(head, tf)
            m = (pt + ph) / 2
            out = m - self.C
            out = out / (np.linalg.norm(out) + 1e-9)
            f = self.frag[12 + n]
            q = m + out * (300 + 100 * f[0]) * a + np.array([0, 400 * a * a])
            spr = melted_bone(22, math.degrees(f[2] * 8 * a), smooth(a / .35), n)
            c.sprites.append((spr, int(q[0] - spr.w / 2), int(q[1] - spr.h / 2), g * .45, None))
        if a < .12:
            c.circle("art", self.C[0], self.C[1], 30 + 300 * a, fade_gray(140, 1 - a / .12))


class OrigamiBox(Attack):
    """折纸立方体 (M22.75-M24.75). The soul was in a rectangle all along (the reveal).
    On the M22.75 sixteenths the paper round it is scored into a box net - four
    flaps of three panels hinged on the rectangle's edges - and the top flap
    creases up (30/60/90 deg): the back wall. On the twelfth triplets the bottom,
    left and right flaps rise while the strip of floor each will hit is washed
    pale, then slap down onto the floor panel by panel (the panel's back is black
    ink: the hazard) and stand up as walls. On the last triplet the lid comes down
    from the camera, a dithered sheet of paper, and the warning goes up once:
    small rectangles of the box - floor cells, wall panels, lid cells - are marked
    with a dashed cut line. The lid shuts on the M24 downbeat; the closed box lifts
    off the sheet and tumbles like a die, 120 deg a beat (a full turn in three), and
    as it turns the marked pieces are cut to bits in a random order. A floor cell
    cut out leaves a hole of ink (hazard) the soul must keep off. It drops back,
    an X is cut across its floor, and it bursts.
    The floor is the rectangle itself and stays square to the screen whenever a
    hazard matters, so hazards are drawn in plain screen / plan coordinates."""

    SIDES = ("top", "bottom", "left", "right")
    SOUL_T = 10.0
    LID_HIGH = 22.0     # how far above the walls the lid hovers when it appears

    def __init__(self, inner, t_reveal, t_score, fold_top, slams, warn_times, t_shut, tumbles, slash_t, t_burst,
                 soul_at_warn=None, n_floor_marks=3, n_wall_marks=4, n_lid_marks=3,
                 height=70.0, D=300.0, vp=(-20.0, 30.0), roll_axis=(1.0, .55, .7), lift=22.0, seed=0):
        x0, y0, x1, y1 = inner
        self.C = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        self.W, self.H, self.h = x1 - x0, y1 - y0, height
        self.D, self.VP = D, self.C + np.asarray(vp, float)
        self.tr, self.ts, self.tb = t_reveal, t_score, t_burst
        self.tx = t_score                                   # (scene: the soul is solid from here on)
        self.fold_top = list(fold_top)
        self.slams = slams                                  # side -> (t_lift, [t_slam of panel 0, 1, 2])
        self.t_shut, self.tumbles, self.slash_t = t_shut, list(tumbles), slash_t
        a = np.asarray(roll_axis, float)
        self.axis = a / np.linalg.norm(a)
        self.LIFT = lift
        self.pivot = np.array([0.0, 0.0, -height / 2])
        rng = np.random.default_rng(seed)
        self.frag = rng.uniform(-1, 1, (40, 4))
        # the warning: which pieces get cut (each cut also cuts the strip of floor
        # under it - its footprint), when each is marked, and when it is cut. The
        # soul's own cell is among them and one edge-middle cell is left clear (the
        # X cut later misses those), so the soul has to get out of the way in time
        cells = [(i, j) for i in range(3) for j in range(3)]
        c0 = self.cell_of(soul_at_warn) if soul_at_warn is not None else (1, 1)
        mids = [(1, 0), (0, 1), (2, 1), (1, 2)]
        far = [m for m in mids if m != c0]
        self.safe = min(far, key=lambda m: (abs(m[0] - c0[0]) + abs(m[1] - c0[1]), rng.random()))
        free = [cl for cl in cells if cl not in (self.safe, c0, (1, 1))]
        fl = [c0] if c0 != self.safe else []
        fl += [free[k] for k in rng.choice(len(free), n_floor_marks - len(fl), replace=False)]
        picked = [("floor", i, j) for i, j in fl]
        walls = [("wall", sd, i) for sd in ("top", "bottom", "left", "right") for i in range(3)]
        walls = [w for w in walls if not self._touches(self.footprint(w), self.safe)]
        picked += [walls[k] for k in sorted(rng.choice(len(walls), n_wall_marks, replace=False))]
        lids = [("lid", i, j) for i, j in cells if (i, j) != self.safe and ("floor", i, j) not in picked]
        picked += [lids[k] for k in sorted(rng.choice(len(lids), n_lid_marks, replace=False))]
        order = rng.permutation(len(picked))                 # one warning, in three bursts...
        self.marks = {}
        for n, k in enumerate(order):
            self.marks[picked[k]] = warn_times[n % len(warn_times)]
        # ...and cut in a random order as it turns - but whatever cuts the soul's own
        # cell waits for the second turn at least, so it has time to get clear
        late = [k for k in picked if self._touches(self.footprint(k), c0)]
        rest = [k for k in picked if k not in late]
        rest = [rest[k] for k in rng.permutation(len(rest))]
        late = [late[k] for k in rng.permutation(len(late))]
        per = math.ceil(len(picked) / len(self.tumbles))
        seq = rest[:per] + [x for pair in zip(late, rest[per:] + [None] * len(late)) for x in pair if x is not None]
        seq += [x for x in rest[per:] if x not in seq]
        self.cuts = {}
        for n, key in enumerate(seq):
            tk = self.tumbles[min(n // per, len(self.tumbles) - 1)]
            self.cuts[key] = tk + .015 + .022 * (n % per)
        self.spin = {key: rng.uniform(-1, 1, 3) for key in picked}
        self.t0, self.t1 = t_reveal, t_burst + .8
        self.impact = t_burst
        self.shake = 6.0

    # ------------------------------------------------------------ the folds
    @staticmethod
    def _back(u, s=1.6):
        u = clamp(u)
        return 1 + (s + 1) * (u - 1) ** 3 + s * (u - 1) ** 2

    def theta(self, side, i, t):
        """Fold angle (deg) of panel i of a flap: 0 flat on the sheet, 90 a wall,
        180 down on the floor."""
        if side == "top":
            th = 0.0
            for k, tk in enumerate(self.fold_top):
                if t < tk:
                    break
                th += (30.0 * (k + 1) - th) * self._back((t - tk) / .06)
            return th
        t_lift, slam = self.slams[side]
        if t < t_lift:
            return 0.0
        ts = slam[i]
        first = min(slam)
        th = 100.0 * smooth((t - t_lift) / max(.05, first - .05 - t_lift))   # rises, poised past upright
        if t < ts - .05:
            if t > ts - .2:                                 # it shivers just before it strikes
                th += 3.0 * math.sin(t * 90 + i)
            return th
        if t < ts:                                          # comes down hard
            u = (t - (ts - .05)) / .05
            return th + (180.0 - th) * u * u
        if t < ts + .06:
            return 180.0
        return 180.0 + (90.0 - 180.0) * self._back((t - ts - .06) / .14, 1.2)   # springs back up: a wall

    def panel_range(self, side, i):
        if side in ("top", "bottom"):
            return -self.W / 2 + self.W * i / 3, -self.W / 2 + self.W * (i + 1) / 3
        return -self.H / 2 + self.H * i / 3, -self.H / 2 + self.H * (i + 1) / 3

    def flap_point(self, side, a, s, th):
        c, sn = math.cos(math.radians(th)), math.sin(math.radians(th))
        W2, H2 = self.W / 2, self.H / 2
        if side == "top":
            return np.array([a, -H2 - s * c, -s * sn])
        if side == "bottom":
            return np.array([a, H2 + s * c, -s * sn])
        if side == "left":
            return np.array([-W2 - s * c, a, -s * sn])
        return np.array([W2 + s * c, a, -s * sn])

    def strip(self, side, i):
        a0, a1 = self.panel_range(side, i)
        W2, H2, h = self.W / 2, self.H / 2, self.h
        cx, cy = self.C
        if side == "top":
            return cx + a0, cy - H2, cx + a1, cy - H2 + h
        if side == "bottom":
            return cx + a0, cy + H2 - h, cx + a1, cy + H2
        if side == "left":
            return cx - W2, cy + a0, cx - W2 + h, cy + a1
        return cx + W2 - h, cy + a0, cx + W2, cy + a1

    def cell_of(self, xy):
        x, y = xy[0] - self.C[0] + self.W / 2, xy[1] - self.C[1] + self.H / 2
        return (int(clamp(x / (self.W / 3), 0, 2.999)), int(clamp(y / (self.H / 3), 0, 2.999)))

    WALL_CUT = 24.0     # how far into the floor a wall panel's cut reaches

    def footprint(self, key):
        """The strip of floor (box coords x0, y0, x1, y1) a piece's cut runs through:
        a floor cell itself; the cell under a lid cell; the edge under a wall panel."""
        if key[0] in ("floor", "lid"):
            return self.cell_xy(key[1], key[2])
        side, i = key[1], key[2]
        a0, a1 = self.panel_range(side, i)
        W2, H2, d = self.W / 2, self.H / 2, self.WALL_CUT
        return {"top": (a0, -H2, a1, -H2 + d), "bottom": (a0, H2 - d, a1, H2),
                "left": (-W2, a0, -W2 + d, a1), "right": (W2 - d, a0, W2, a1)}[side]

    def _touches(self, fp, cell):
        x0, y0, x1, y1 = fp
        c0, c1, c2, c3 = self.cell_xy(*cell)
        return x0 < c2 - 1 and c0 + 1 < x1 and y0 < c3 - 1 and c1 + 1 < y1

    def cell_xy(self, i, j):
        W3, H3 = self.W / 3, self.H / 3
        return -self.W / 2 + i * W3, -self.H / 2 + j * H3, -self.W / 2 + (i + 1) * W3, -self.H / 2 + (j + 1) * H3

    def lid_z(self, t):
        a = t - (min(self.marks.values()) - .02)
        if a < 0:
            return None
        u = clamp(a / max(.05, self.t_shut - (min(self.marks.values()) - .02)))
        return -self.h - self.LID_HIGH * (1 - u * u)          # comes down, faster and faster, and shuts

    def piece(self, key, t):
        """A small rectangle of the box: its four corners (box coords), outward normal."""
        if key[0] == "floor":
            x0, y0, x1, y1 = self.cell_xy(key[1], key[2])
            return [np.array(v) for v in ((x0, y0, 0.0), (x1, y0, 0.0), (x1, y1, 0.0), (x0, y1, 0.0))], np.array([0, 0, 1.0])
        if key[0] == "lid":
            z = self.lid_z(t)
            z = -self.h if z is None else z
            x0, y0, x1, y1 = self.cell_xy(key[1], key[2])
            return [np.array(v) for v in ((x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z))], np.array([0, 0, -1.0])
        side, i = key[1], key[2]
        a0, a1 = self.panel_range(side, i)
        th = self.theta(side, i, t)
        pts = [self.flap_point(side, a, s, th) for a, s in ((a0, 0), (a1, 0), (a1, self.h), (a0, self.h))]
        nrm = {"top": (0, -1, 0), "bottom": (0, 1, 0), "left": (-1, 0, 0), "right": (1, 0, 0)}[side]
        return pts, np.array(nrm, float)

    # ------------------------------------------------------------ the tumble
    def pose(self, t):
        """(rotation, lift, scale): the shut box floats up a little and tumbles
        about a slanted axis, 120 deg a beat; square on the sheet otherwise."""
        if not self.tumbles or t < self.t_shut + .04 or t >= self.tb:
            return np.eye(3), 0.0, 1.0
        ang = 0.0
        for tk in self.tumbles:
            if t >= tk:
                ang += 120.0 * self._back((t - tk) / .1, 1.3)
        a = math.radians(ang)
        x, y, z = self.axis
        K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        R = np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * (K @ K)
        last = self.tumbles[-1] + .1
        up = smooth((t - (self.t_shut + .04)) / .08) * (1 - smooth((t - last) / max(.02, self.slash_t - last)))
        return R, -self.LIFT * up, 1 - .22 * up

    def project(self, p, t, pose=None):
        R, lift, sc = pose if pose is not None else self.pose(t)
        q = (R @ (np.asarray(p, float) - self.pivot)) * sc + self.pivot
        q[2] += lift
        k = self.D / max(40.0, self.D + q[2])
        return self.VP + (self.C + q[:2] - self.VP) * k, q[2], k

    def soul_view(self, xy, t):
        if not (self.tx <= t < self.tb):
            return np.asarray(xy, float), 1.0
        p, _, k = self.project((xy[0] - self.C[0], xy[1] - self.C[1], 0.0), t)
        return p, k

    def soul_layers(self, xy, t):
        from common import Sprite
        src = sprite("player.png")
        mask = src.a
        cap = src.v.copy()
        cap[3:6, 3:5] = 200
        cap[2, 4], cap[6, 3], cap[3, 5] = 150, 150, 150
        sx, sy = xy[0] - self.C[0], xy[1] - self.C[1]
        sq = [(-8, -8), (8, -8), (8, 8), (-8, 8)]
        n = int(self.SOUL_T)
        pose = self.pose(t)
        W2, H2 = self.W / 2, self.H / 2
        fl = [self.project((x, y, 0.0), t, pose)[0] for x, y in ((-W2, -H2), (W2, -H2), (W2, H2), (-W2, H2))]
        if sum(fl[k][0] * fl[(k + 1) % 4][1] - fl[(k + 1) % 4][0] * fl[k][1] for k in range(4)) < .45 * 2 * self.W * self.H:
            return []                                        # the floor is turned away / edge on: the box hides it
        layers = []
        for i in range(n + 1):
            P = [self.project((sx + u, sy + v, -float(i)), t, pose) for u, v in sq]
            quad = np.array([p for p, _, _ in P])
            zr = float(np.mean([zz for _, zz, _ in P]))
            val = cap if i == n else np.full_like(src.v, 18 + 2 * i)
            spr = warp_sprite(val, mask, quad)
            if spr is not None:
                layers.append((zr, i, spr))
        layers.sort(key=lambda e: (-e[0], e[1]))
        return [(s, x0, y0) for _, _, (s, x0, y0) in layers]

    # ------------------------------------------------------------ drawing
    def _net_quads(self):
        out = []
        for side in self.SIDES:
            for i in range(3):
                a0, a1 = self.panel_range(side, i)
                pts = [self.flap_point(side, a, s, 0.0) for a, s in ((a0, 0), (a1, 0), (a1, self.h), (a0, self.h))]
                out.append((side, i, [self.C + p[:2] for p in pts]))
        return out

    @staticmethod
    def _area(pts):
        return sum(pts[k][0] * pts[(k + 1) % 4][1] - pts[(k + 1) % 4][0] * pts[k][1] for k in range(len(pts)))

    def draw(self, c):
        t = c.t
        if t < self.tr:
            return
        if t >= self.tb:
            return self.draw_burst(c, t)
        W2, H2 = self.W / 2, self.H / 2
        base = [self.C + np.array(v) for v in ((-W2, -H2), (W2, -H2), (W2, H2), (-W2, H2))]
        if t < self.ts:                                      # the rectangle is simply there (the reveal)
            c.line("art", base + [base[0]], 6, 255)
            return
        pose = self.pose(t)
        lifted = pose[1] < -.5
        # --- the sheet: the net scored into it; holes where the paper has gone up
        sc = clamp((t - self.ts) / .06)
        for side, i, q in self._net_quads():
            if self.theta(side, i, t) > 3 or lifted:
                c.poly("art", q, 135)                        # the shadowed underneath of the sheet
            else:
                seg = [q[1], q[2], q[3], q[0]]                # (the hinge, q0-q1, is the floor's crease)
                for j in range(3):
                    p0, p1 = seg[j], seg[j + 1]
                    c.line("art", [p0, p0 + (p1 - p0) * sc], 3, 150)
        if lifted:
            c.poly("art", base, 135)
        # --- the box, far to near: floor cells, wall panels, lid cells, flying bits
        items = []

        def add(pts3, kind, key, extra=None):
            P = [self.project(p, t, pose) for p in pts3]
            items.append((float(np.mean([z for _, z, _ in P])), kind, key, [p for p, _, _ in P], extra))

        for i in range(3):
            for j in range(3):
                key = ("floor", i, j)
                pts, _ = self.piece(key, t)
                add(pts, "floor", key)
        for side in self.SIDES:
            for i in range(3):
                th = self.theta(side, i, t)
                if th <= 3 and not lifted:
                    continue
                key = ("wall", side, i)
                if key in self.cuts and t >= self.cuts[key]:
                    continue
                pts, _ = self.piece(key, t)
                add(pts, "wall", key, th)
        if self.lid_z(t) is not None:
            for i in range(3):
                for j in range(3):
                    key = ("lid", i, j)
                    if key in self.cuts and t >= self.cuts[key]:
                        continue
                    pts, _ = self.piece(key, t)
                    add(pts, "lid", key)
        for key, tc in self.cuts.items():                     # the cut pieces fly apart in two halves
            a = t - tc
            if not (0 <= a < .45):
                continue
            pts, nrm = self.piece(key, min(t, tc))
            u = a / .45
            for half, tri in enumerate(((pts[0], pts[1], pts[2]), (pts[0], pts[2], pts[3]))):
                cen = np.mean(tri, axis=0)
                d_in = (tri[1] - cen) if half == 0 else (tri[2] - cen)
                d_in = d_in / (np.linalg.norm(d_in) + 1e-9)
                f = self.spin[key]
                off = nrm * 70 * u + d_in * 26 * u + np.array([0, 0, 0.0])
                ang = (2.4 + f[0]) * u * (1 if half else -1)
                ax_ = tri[2] - tri[0] if half == 0 else tri[1] - tri[0]
                ax_ = ax_ / (np.linalg.norm(ax_) + 1e-9)
                Kx = np.array([[0, -ax_[2], ax_[1]], [ax_[2], 0, -ax_[0]], [-ax_[1], ax_[0], 0]])
                Rr = np.eye(3) + math.sin(ang) * Kx + (1 - math.cos(ang)) * (Kx @ Kx)
                tri2 = [cen + off + Rr @ (p - cen) * (1 - .3 * u) for p in tri]
                add(tri2, "shard", key, 1 - u)
        items.sort(key=lambda e: -e[0])
        n_floor = 0
        for z, kind, key, pts, extra in items:
            if kind == "floor":
                up = self._area(pts) > 0
                n_floor += 1
                if n_floor == 9:                               # (the floor is down: what cuts through it, next)
                    after_floor = True
                else:
                    after_floor = False
                if key in self.cuts and t >= self.cuts[key]:
                    c.poly("art", pts, 255)                  # cut out: a hole of ink (hazard, in the floor's own coords)
                    x0, y0, x1, y1 = self.cell_xy(key[1], key[2])
                    c.rect("haz", self.C[0] + x0 + 2, self.C[1] + y0 + 2, self.C[0] + x1 - 2, self.C[1] + y1 - 2, 255)
                else:
                    c.poly("art", pts, 10 if up else 92)
                    c.line("art", pts + [pts[0]], 2, 70)       # the floor's own grid, pale
                    self.draw_mark(c, key, pts, t, up)
                if after_floor:
                    self.draw_footprints(c, t, pose, up)
            elif kind == "wall":
                th = extra
                inside = self._area(pts) * self._flat_sign(key[1]) > 0
                hot = th >= 165 and not lifted
                if hot:
                    fill, edge = 255, 255
                elif inside:
                    fill, edge = int(14 + 34 * abs(math.sin(math.radians(th)))), 175
                else:
                    fill, edge = 92, 175
                c.poly("art", pts, fill)
                c.line("art", pts + [pts[0]], 3, edge)
                if hot:
                    x0, y0, x1, y1 = self.strip(key[1], key[2])
                    c.rect("haz", x0 + 2, y0 + 2, x1 - 2, y1 - 2, 255)
                elif th >= 60:
                    self.draw_mark(c, key, pts, t, True)
            elif kind == "lid":
                self._dither(c, pts, int(34 * clamp((t - min(self.marks.values()) + .02) / .15)) or 1)
                c.line("art", pts + [pts[0]], 2, 150)
                self.draw_mark(c, key, pts, t, True)
            else:                                              # a flying half of a cut piece (pale)
                c.poly("art", pts, fade_gray(60, extra))
                c.line("art", pts + [pts[0]], 2, fade_gray(140, extra))
        # the floor's outline (its creases) and the slashes over everything on it
        fl = [self.project((x - self.C[0], y - self.C[1], 0.0), t, pose)[0] for x, y in base]
        if self._area(fl) > 0:
            c.line("art", fl + [fl[0]], 5, 200)
        for key, tc in self.cuts.items():                     # the blade that cuts a piece: a quick black stroke
            a = t - tc
            if -.03 <= a < .05:
                pts, _ = self.piece(key, min(t, tc))
                P = [self.project(p, t, pose)[0] for p in (pts[0], pts[2])]
                g = clamp((a + .03) / .03)
                c.line("art", [P[0], P[0] + (P[1] - P[0]) * g], 4, 255 if key[0] == "floor" else 165)
        self.draw_slash(c, t, pose)

    FLASH = .1          # a blade cutting a lid / wall piece cuts through the floor under it this long

    def draw_footprints(self, c, t, pose, up):
        """The floor under every marked lid / wall piece: washed pale from its warning
        (where not to stand), then, as the piece is cut, the blade goes right down
        through the box - a flash of ink across that strip (the hazard)."""
        for key, tw in self.marks.items():
            if key[0] == "floor":
                continue
            tc = self.cuts[key]
            if not (tw <= t < tc + self.FLASH):
                continue
            x0, y0, x1, y1 = self.footprint(key)
            pts = [self.project((x, y, 0.0), t, pose)[0] for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
            if t < tc:
                if up:
                    late = t > tc - .12
                    c.poly("art", [p + (np.mean(pts, axis=0) - p) * .06 for p in pts],
                           (96 if int(t * 60) % 4 < 2 else 56) if late else 48)
                continue
            c.poly("art", pts, 255)
            c.rect("haz", self.C[0] + x0 + 2, self.C[1] + y0 + 2, self.C[0] + x1 - 2, self.C[1] + y1 - 2, 255)

    def _flat_sign(self, side):
        if not hasattr(self, "_fs"):
            self._fs = {}
            for s, i, q in self._net_quads():
                if i == 0:
                    self._fs[s] = 1 if self._area(q) > 0 else -1
        return self._fs[side]

    def _dither(self, c, pts, val):
        """Fill a quad on every other pixel (a see-through sheet of paper)."""
        k = c.k
        m = Image.new("L", c.size)
        ImageDraw.Draw(m).polygon([(p[0] / k, p[1] / k) for p in pts], fill=255)
        mm = np.asarray(m) > 0
        yy, xx = np.indices(mm.shape)
        mm &= (xx + yy) % 2 == 0
        art = np.asarray(c.art).copy()
        art[mm] = np.maximum(art[mm], val)
        c.art.paste(Image.fromarray(art))

    def draw_mark(self, c, key, pts, t, visible):
        """The one warning: a marked piece gets a pale wash and a dashed cut line
        (popping in with a flash), and blinks just before it is cut."""
        if key not in self.marks or not visible:
            return
        tm_, tc = self.marks[key], self.cuts[key]
        if not (tm_ <= t < tc):
            return
        late = t > tc - .12
        wash = (110 if int(t * 60) % 4 < 2 else 70) if late else 62
        inner = [p + (np.mean(pts, axis=0) - p) * .1 for p in pts]
        c.poly("art", inner, wash)
        p0, p1 = inner[0], inner[2]
        n = 7
        for j in range(n):                                   # the dashed line it will be cut along
            if j % 2 == 0:
                c.line("art", [p0 + (p1 - p0) * j / n, p0 + (p1 - p0) * (j + 1) / n], 3, 165)
        a = t - tm_
        if a < .15:
            c.line("art", pts + [pts[0]], 4, fade_gray(170, 1 - a / .15))

    def draw_slash(self, c, t, pose):
        """The X across the floor on the M24.5 slides: a wash of ink first."""
        a = t - self.slash_t
        if a < -.38:
            return
        hw, hh = self.W / 2 * .78, self.H / 2 * .78
        for sgn in (1, -1):
            p0, p1 = np.array([-hw, -hh * sgn, 0.0]), np.array([hw, hh * sgn, 0.0])
            n = 16
            pts = [self.project(p0 + (p1 - p0) * j / n, t, pose) for j in range(n + 1)]
            if a < 0:
                late = a > -.14
                val = (150 if int(t * 60) % 4 < 2 else 96) if late else 100
                for (q0, _, k0), (q1, _, _) in zip(pts, pts[1:]):
                    c.line("art", [q0, q1], 12 * k0, val)
                continue
            m = max(1, int(n * clamp(a / .05)))
            for j in range(m):
                (q0, _, k0), (q1, _, _) = pts[j], pts[j + 1]
                w = 11 * k0 * (1.15 - .5 * j / n)
                c.line("art", [q0, q1], w, 255)
                c.circle("art", q0[0], q0[1], w, 255)
            c.line("haz", [self.C + p0[:2], self.C + p1[:2]], 10, 255)

    def draw_burst(self, c, t):
        """The box bursts: what is left of it flies apart, turning and fading (pale,
        harmless); the hole in the sheet closes over at once."""
        a = t - self.tb
        g = 1 - smooth(a / .6)
        if g <= 0:
            return
        tf = self.tb - 1e-3
        pieces = []
        for i in range(3):
            for j in range(3):
                for kind in ("floor", "lid"):
                    key = (kind, i, j)
                    if key in self.cuts:
                        continue
                    pts, _ = self.piece(key, tf)
                    pieces.append([self.project(p, tf)[0] for p in pts])
        for side in self.SIDES:
            for i in range(3):
                key = ("wall", side, i)
                if key in self.cuts:
                    continue
                pts, _ = self.piece(key, tf)
                pieces.append([self.project(p, tf)[0] for p in pts])
        for k, pts in enumerate(pieces):
            m = np.mean(pts, axis=0)
            out = m - self.C
            out = out / (np.linalg.norm(out) + 1e-9)
            f = self.frag[k % len(self.frag)]
            off = out * (300 + 140 * f[0]) * a + np.array([f[1] * 60, 420 * a * a])
            rot = f[2] * 7 * a
            cr, sr = math.cos(rot), math.sin(rot)
            s = 1 - .55 * a
            q = [m + off + np.array([(p - m)[0] * cr - (p - m)[1] * sr, (p - m)[0] * sr + (p - m)[1] * cr]) * s
                 for p in pts]
            c.poly("art", q, fade_gray(40, g))
            c.line("art", q + [q[0]], 3, fade_gray(120, g))
        hole = 1 - smooth(a / .12)
        if hole > 0:
            W2, H2 = self.W / 2, self.H / 2
            for side, i, q in self._net_quads():
                c.poly("art", q, fade_gray(135, hole))
            c.poly("art", [self.C + np.array(v) for v in ((-W2, -H2), (W2, -H2), (W2, H2), (-W2, H2))],
                   fade_gray(135, hole))
        if a < .12:
            c.circle("art", self.C[0], self.C[1], 30 + 300 * a, fade_gray(140, 1 - a / .12))


# ======================================================================
# The rest after the cube (M25-M28.5): the canvas now wraps the buttons too;
# the soul goes to FIGHT and strikes Ink Sans again and again - every slash is
# caught by a shield of ink. Nothing here can hurt the soul.
class FightTarget(Attack):
    """UT's attack meter, painted in ink on the canvas: a wide target eye; one bar
    after another slides in and stops on the centre line on each strike."""

    def __init__(self, center, half_w, half_h, t_paint, strikes, t_end, slide=.42, seed=0):
        self.C = np.asarray(center, float)
        self.hw, self.hh = half_w, half_h
        self.tp, self.strikes, self.te, self.slide = t_paint, strikes, t_end, slide
        self.t0, self.t1 = t_paint, t_end + .5
        self.impact = None

    def draw(self, c):
        t = c.t
        g = 1 - smooth((t - self.te) / .45)
        if g <= 0:
            return
        cx, cy = self.C
        u = clamp((t - self.tp) / .3)
        # the eye: nested ellipses, heaviest outside, painted round in one sweep
        for k, (fr, w, v) in enumerate(((1.0, 7, 235), (.72, 5, 170), (.44, 4, 130), (.18, 3, 110))):
            n = int(64 * clamp(u * 1.4 - k * .12))
            pts = [(cx + math.cos(a) * self.hw * fr, cy + math.sin(a) * self.hh * fr)
                   for a in np.linspace(-math.pi / 2, -math.pi / 2 + math.tau * n / 64, max(2, n))]
            if n > 1:
                c.line("art", pts, w, int(v * g))
        if u >= 1:
            c.line("art", [(cx, cy - self.hh - 8), (cx, cy + self.hh + 8)], 3, int(200 * g))
        # the bars: each slides in and stops on the centre on its strike
        for i, ts in enumerate(self.strikes):
            a = t - (ts - self.slide)
            if a < 0 or t > ts + .22:
                continue
            side = -1 if i % 2 == 0 else 1
            x = cx + side * self.hw * (1 - out_cubic(clamp(a / self.slide)))
            hit = t >= ts
            val = 255 if not hit else int(255 * (1 - (t - ts) / .22))
            w = 12 if not hit else 12 + 16 * (t - ts) / .22
            c.rect("art", x - w / 2, cy - self.hh - 4, x + w / 2, cy + self.hh + 4, val)


_PEN_ROT = {}


def pen_turned(angle: float, mirrored: bool = False, white: bool = False, k: float = 1.0):
    """The 2x brush (pen0) turned by `angle` degrees (counter-clockwise). `white`
    gives its flash silhouette, `k` shrinks it about its centre (the blink)."""
    aq = int(round(angle / 4)) * 4
    kq = round(k * 10) / 10 if white else 1.0
    key = (aq, mirrored, white, kq)
    if key not in _PEN_ROT:
        from common import Sprite
        base = pen_sprite(0, mirrored)
        iv = Image.fromarray(base.v).rotate(aq, resample=Image.Resampling.NEAREST, expand=True)
        ia = Image.fromarray(base.a.astype(np.uint8) * 255).rotate(aq, resample=Image.Resampling.NEAREST, expand=True)
        if kq < 1:
            size = (max(2, int(iv.width * kq)), max(2, int(iv.height * kq)))
            iv, ia = iv.resize(size, Image.Resampling.NEAREST), ia.resize(size, Image.Resampling.NEAREST)
        v, a = np.asarray(iv).copy(), np.asarray(ia) > 127
        if white:
            v[:] = 255
        _PEN_ROT[key] = Sprite(v, a)
    return _PEN_ROT[key]


class InkShieldStrikes(Attack):
    """Each strike: UT's slash (sprite/fight/1-6) cuts down over Ink Sans - who does
    not move at all. His brush blinks off his back and into the slash's path (a
    glint, a white flash, then the brush), takes the hit - it dips and flings a
    crescent of ink, the slash breaks on it in sparks - and blinks across to meet
    the next one. After the last it blinks back onto his back (BRUSH_AWAY)."""

    APPEAR = .08        # the brush is there this long before the strike lands
    VANISH = .115       # ... and leaves this long before the next one lands
    BACK = .36          # after the last strike it holds this long, then goes home

    def __init__(self, center, strikes, seed=0):
        self.C = np.asarray(center, float)
        self.strikes = strikes
        rng = np.random.default_rng(seed)
        self.jit = rng.uniform(-1, 1, (len(strikes), 6))
        self.drops = rng.uniform(0, 1, (len(strikes), 8, 3))
        self.motes = rng.uniform(0, 1, (len(strikes), 7, 3))
        self.t0, self.t1 = strikes[0] - .2, strikes[-1] + 1.0
        self.impact = None
        self._slash = None

    @property
    def brush_away(self):
        """When the brush is off his back: (leaves, returns)."""
        return (self.strikes[0] - self.APPEAR, self.strikes[-1] + self.BACK + .05)

    def slash(self, k):
        if self._slash is None:
            from common import Sprite
            out = []
            for i in range(1, 7):
                s = sprite(f"fight/{i}.png")
                v = np.clip(s.v.astype(int) * 1.5, 0, 255).astype(np.uint8)       # lift the red to a bright grey
                out.append(Sprite(v.repeat(2, 0).repeat(2, 1), s.a.repeat(2, 0).repeat(2, 1)))
            self._slash = out
        return self._slash[k]

    def arc(self, i):
        """The crescent of ink the brush flings as it takes the hit (alternating sides)."""
        j = self.jit[i]
        cx, cy = self.C[0] + j[0] * 6, self.C[1] + 100 + j[1] * 6
        a0, a1 = math.radians(228 + 6 * j[2]), math.radians(312 + 6 * j[3])
        if i % 2:
            a0, a1 = a1, a0
        s = np.linspace(a0, a1, 28)
        return np.stack([cx + np.cos(s) * 104, cy + np.sin(s) * 104], 1)

    def window(self, i):
        ts = self.strikes[i]
        t_in = ts - self.APPEAR
        t_out = self.strikes[i + 1] - self.VANISH if i + 1 < len(self.strikes) else ts + self.BACK
        return t_in, t_out

    def brush(self, c, i):
        """Returns the brush sprite entry for strike i at c.t (or None); draws glints/motes."""
        t = c.t
        ts = self.strikes[i]
        t_in, t_out = self.window(i)
        j = self.jit[i]
        side = 1 if i % 2 == 0 else -1
        a = t - ts
        dip = 0.0
        if a >= 0:                                     # takes the hit: the tip dips, then settles
            dip = 16 * out_cubic(clamp(a / .03)) if a < .03 else 16 - 11 * smooth((a - .03) / .12)
        ang = side * (54 - dip)
        bx = self.C[0] + j[0] * 8
        by = self.C[1] - 2 + j[1] * 6 + (6 * dip / 16)
        # motes left where it blinked out
        if t_out <= t < t_out + .24:
            u = (t - t_out) / .24
            for r1, r2, r3 in self.motes[i]:
                ang_m = r1 * math.tau
                d = 8 + 46 * out_cubic(u) * (.5 + r2)
                c.circle("top", bx + math.cos(ang_m) * d, by + math.sin(ang_m) * d - 20 * u,
                         2 + 2 * r3, fade_gray(230, 1 - u))
        for tb in (t_in, t_out):                       # a thin ring bursts where it blinks in / out
            u = (t - tb) / .12
            if 0 <= u < 1:
                c.ring("top", bx, by, 10 + 70 * out_cubic(u), 3 * (1 - u) + 1, fade_gray(255, 1 - u))
        if t < t_in or t >= t_out + .035:
            return None
        if t < t_in + .015 or t >= t_out + .025:      # the glint: a line and a ring
            g = clamp((t - t_in) / .015) if t < t_in + .015 else 1 - clamp((t - t_out - .025) / .01)
            c.line("top", [(bx - 80 * g - 10, by), (bx + 80 * g + 10, by)], 2, 255)
            c.circle("top", bx, by, 4 + 10 * g, 255)
            return None
        if t < t_in + .045:                            # the flash: a white brush growing out of the glint
            pen = pen_turned(ang, side < 0, white=True, k=.3 + .7 * (t - t_in - .015) / .03)
        elif t >= t_out:                               # flash and shrink away
            pen = pen_turned(ang, side < 0, white=True, k=1 - .75 * (t - t_out) / .025)
        else:
            pen = pen_turned(ang, side < 0)
        return (pen, int(bx - pen.w / 2), int(by - pen.h / 2), 1.0, None)

    def draw(self, c):
        t = c.t
        pens = []
        for i, ts in enumerate(self.strikes):
            a = t - ts
            if a < -.2 or a > .9:
                continue
            p = self.brush(c, i)
            if p is not None:
                pens.append(p)
            if a < 0:
                continue
            pts = self.arc(i)
            n = len(pts)
            if a < .3:                                        # the crescent, flung in one stroke
                m = max(2, int(n * clamp(a / .06)))
                for k in range(m - 1):
                    w = 16 * (1 - .75 * k / n)
                    c.line("top", [pts[k], pts[k + 1]], w, 165)
                    c.circle("top", pts[k][0], pts[k][1], w, 165)
                if a < .2:                                    # drops thrown off the end of the cut
                    end, d = pts[-1], pts[-1] - pts[-2]
                    d = d / (np.linalg.norm(d) + 1e-9)
                    for q, (r1, r2, r3) in enumerate(self.drops[i]):
                        sp = 120 + 160 * r1
                        p = end + d * sp * a + np.array([(r2 - .5) * 60 * a, 300 * a * a])
                        c.circle("top", p[0], p[1], 4 + 4 * r3, fade_gray(165, 1 - a / .2))
            else:                                             # it runs away as drips
                b = (a - .3) / .6
                for k in range(0, n, 3):
                    x, y = pts[k]
                    c.line("top", [(x, y), (x, y + 46 * b * (.6 + .4 * math.sin(k)))], 4, fade_gray(150, 1 - b))
            # the slash, cut straight down over him - stopped by the brush and the ink
            if a < .3:
                spr = self.slash(min(5, int(a / .05)))
                c.sprites.append((spr, int(self.C[0] - spr.w / 2 + self.jit[i][4] * 6), int(self.C[1] - spr.h / 2),
                                  1.0, None))
            if a < .25:                                       # sparks where it breaks on the brush
                hit = (self.C[0] + self.jit[i][0] * 8, self.C[1] - 2 + self.jit[i][1] * 6)
                for q in range(9):
                    ang = q / 9 * math.tau + self.jit[i][5]
                    d = 10 + 60 * out_cubic(a / .25)
                    c.circle("top", hit[0] + math.cos(ang) * d, hit[1] + math.sin(ang) * d, 5,
                             fade_gray(240, 1 - a / .25))
        c.sprites.extend(pens)                                # the brush in front of the slash


class HuntingInk(Attack):
    """The ring of ink the cube bursts into does not dry. The drops are thrown far
    out, stop, tremble - and then, as if they had come alive, fall in around the
    soul in a wide ring and close in on it while it heads for FIGHT: slowly and
    steadily, never in jerks (close_progress), from a tilted ring to a round one,
    spinning faster, so that when the soul reaches the button (t_cut) the drops are
    about to touch it.
    The one press of FIGHT then cuts the whole ring in a single sweep: every drop
    takes UT's slash and stops dead where it was cut. Drawn as ink (black on the
    paper); it never touches the soul, and nothing here hurts."""

    WAKE = (.3, .52)           # thrown out until WAKE[0]; trembling; alive from WAKE[1]
    RUSH = .4                  # ...then this long to fall in around the soul
    THROW = 900.0              # how hard the burst throws the drops out
    R0, R_LAST, R1 = 120.0, 36.0, 25.0   # ring radius: wide at first / after the last lunge / at the cut
    LUNGE = .18                # each lunge inwards takes this long
    FLAT0, FLAT1 = .5, .92     # seen at an angle (ry = FLAT * rx) - it rises to face us as it tightens
    SPIN0, SPIN1 = 1.0, 2.6    # rad/s: faster as it tightens
    SWEEP = .1                 # the one slash runs round the whole ring in this long

    def __init__(self, center, t_burst, target, t_cut, t_end, lunges=(), n=8, tele=.42, seed=0):
        self.C = np.asarray(center, float)
        self.tb, self.tc, self.te, self.n, self.tele = t_burst, t_cut, t_end, n, tele
        self.lunges = sorted(lunges)
        # the radius after each lunge: the steps get bigger towards the end
        m = len(self.lunges)
        self.radii = [self.R0 + (self.R_LAST - self.R0) * (.45 * u + .55 * u * u)
                      for u in (k / max(1, m) for k in range(m + 1))]
        rng = np.random.default_rng(seed)
        self.spl = rng.uniform(0, 1, (n, 5, 3))
        self.slash_dx = rng.uniform(-3, 3, n)
        self.target = target
        self.t_alive = t_burst + self.WAKE[1]
        self.t_close = self.t_alive + self.RUSH
        # integrate the paths once (the soul's path is known in advance)
        dt = 1 / 120
        self.ts = np.arange(t_burst, t_end + dt, dt)
        P = np.zeros((len(self.ts), n, 2))
        ang = np.linspace(0, math.tau, n, endpoint=False) + rng.uniform(-.1, .1, n)
        p = np.repeat(self.C[None], n, 0).astype(float)
        v = np.stack([np.cos(ang), np.sin(ang)], 1) * self.THROW
        slot, o_prev, phase = None, None, 0.0
        self.cut_t = np.full(n, np.inf)
        self.cut_c, self.cut_a0 = None, 0.0
        for i, t in enumerate(self.ts):
            P[i] = p
            tau = t - t_burst
            if tau < self.WAKE[0]:                        # thrown out, slowing down
                v *= math.exp(-7.0 * dt)
            elif tau < self.WAKE[1]:                      # stops, trembles
                v *= math.exp(-14.0 * dt)
            else:                                         # alive: rush in, then circle and close in
                sp = np.asarray(target(t), float)
                if slot is None:                          # each drop takes the slot on its own side
                    rel = p - sp
                    a0 = np.arctan2(rel[:, 1] / self.FLAT0, rel[:, 0])
                    srt = np.argsort(a0)
                    slot = np.zeros(n)
                    base = a0[srt[0]]
                    for j, k in enumerate(srt):
                        slot[k] = base + j * math.tau / n
                phase += self.spin(t) * dt
                o = self.orbit(sp, t, slot + phase)
                vo = np.zeros_like(o) if o_prev is None else (o - o_prev) / dt
                o_prev = o
                # a spring onto its place on the ring; once there it holds on tight
                # and moves with it (so the ring stays centred on the walking soul)
                K = 95.0 + 305.0 * smooth((t - self.t_close) / .3)
                acc = (o - p) * K + (vo * smooth((t - self.t_alive) / .3) - v) * 2 * math.sqrt(K)
                v = v + acc * dt * smooth((t - self.t_alive) / .2)
                spd = np.linalg.norm(v, axis=1, keepdims=True)
                v = np.where(spd > 520, v / (spd + 1e-9) * 520, v)
                if t >= t_cut and self.cut_c is None:     # the slash sweeps round from the top
                    self.cut_c = sp.copy()
                    rel = p - sp
                    a = np.arctan2(rel[:, 1], rel[:, 0])
                    self.cut_a0 = -math.pi / 2
                    self.cut_t = t_cut + self.SWEEP * ((a - self.cut_a0) % math.tau) / math.tau
            v[t >= self.cut_t] = 0                        # a cut drop stops dead
            p = p + v * dt
        self.P = P
        self.t0, self.t1 = t_burst - tele, t_end + .2
        self.impact = None
        self.overscan = True                              # also drawn past the screen (scene.overscan)

    def radius(self, t):
        """Wide once it has rung the soul, then closing in slowly and smoothly all
        the way to the cut (no steps, no overshoot)."""
        return self.R0 + (self.R1 - self.R0) * close_progress(t, self.t_close, self.tc)

    def closing(self, t):
        return clamp((self.R0 - self.radius(t)) / (self.R0 - self.R1), 0.0, 1.1)

    def spin(self, t):
        return self.SPIN0 + (self.SPIN1 - self.SPIN0) * min(1.0, self.closing(t))

    def ring(self, t):
        r = self.radius(t)
        e = min(1.0, self.closing(t))
        return r, r * (self.FLAT0 + (self.FLAT1 - self.FLAT0) * e)

    def orbit(self, sp, t, th):
        rx, ry = self.ring(t)
        cx = sp[0]                                         # (the sheet runs on past the screen)
        cy = sp[1] + max(0.0, 236 + ry - sp[1])
        return np.stack([cx + np.cos(th) * rx, cy + np.sin(th) * ry], 1)

    def slash(self, k):
        if not hasattr(self, "_slash"):
            from common import Sprite
            out = []
            for i in range(1, 7):
                s_ = sprite(f"fight/{i}.png")
                v = s_.v.astype(float)
                v = np.clip(v / max(1.0, v[s_.a].max() if s_.a.any() else 1) * 255, 0, 255).astype(np.uint8)
                # half size: one to each drop of the tight ring, without piling up
                ys = (np.arange(s_.h // 2) * 2).astype(int)
                xs = (np.arange(s_.w // 2) * 2).astype(int)
                a_ = s_.a[ys][:, xs] | s_.a[ys + 1][:, xs] | s_.a[ys][:, xs + 1]
                out.append(Sprite(np.where(a_, v.max(), 0).astype(np.uint8), a_))
            self._slash = out
        return self._slash[k]

    def draw(self, c):
        t = c.t
        if t < self.tb:                                   # the ink gathers into the middle
            u = (t - self.tb + self.tele) / self.tele
            for i in range(self.n):
                a = i / self.n * math.tau + .6 * (1 - u)
                r = 50 * (1 - out_cubic(u)) + 4
                c.circle("top", self.C[0] + math.cos(a) * r, self.C[1] + math.sin(a) * r, 6, fade_gray(90 + 90 * u, 1))
            return
        i = min(len(self.ts) - 1, int((t - self.tb) * 120))
        j = max(0, i - 5)
        tau = t - self.tb
        sp = np.asarray(self.target(t), float)
        e = self.closing(t)
        rx, ry = self.ring(t)
        flat = 1 - (ry / rx - self.FLAT0) / (self.FLAT1 - self.FLAT0)     # 1 tilted .. 0 facing us
        # the one slash: a cut that runs round the whole ring
        if self.cut_c is not None and self.tc <= t < self.tc + self.SWEEP + .2:
            a = t - self.tc
            prog = clamp(a / self.SWEEP)
            g = 1 - smooth((a - self.SWEEP) / .2)
            m = max(2, int(48 * prog))
            R1x, R1y = self.R1 + 7, (self.R1 + 7) * self.FLAT1
            pts = [(self.cut_c[0] + math.cos(self.cut_a0 + math.tau * prog * q / m) * R1x,
                    self.cut_c[1] + math.sin(self.cut_a0 + math.tau * prog * q / m) * R1y) for q in range(m + 1)]
            c.line("top", pts, 3, fade_gray(140, g))
        order = np.argsort(self.P[i, :, 1])                # far (upper) drops first
        for k in order:
            x, y = self.P[i, k]
            px, py = self.P[j, k]
            size = 16.0
            if tau > self.WAKE[1]:                         # on the ring: nearer us (lower) is larger
                dy = (y - sp[1]) / max(1.0, ry)
                near = clamp((math.hypot(x - sp[0], y - sp[1]) - 70) / 60)
                size = 17.0 - 2.0 * e + 4.0 * float(np.clip(dy, -1, 1)) * (1 - near) * flat
            cut = t >= self.cut_t[k]
            if self.WAKE[0] < tau < self.WAKE[1] + .1:      # trembling as it wakes up
                x += 1.5 * math.sin(t * 61 + k)
                y += 1.5 * math.cos(t * 53 + k * 2)
            c.circle("top", x, y, size + 6, 1)             # a rim of paper: it reads over the HUD
            if not cut:
                c.line("top", [(px, py), (x, y)], max(3.0, size * .42), 200)
            c.circle("top", x, y, size, 245)
            if not cut:
                c.circle("top", x - size * .15, y - size * .15, max(2.0, size * .28), 255)
                continue
            a = t - self.cut_t[k]
            # cut through: a clean line across it, a few drops thrown off
            c.line("top", [(x + 2 + self.slash_dx[k], y - 11), (x - 2 + self.slash_dx[k], y + 11)], 3, 1)
            if a < .3:
                for r1, r2, r3 in self.spl[k]:
                    q = a / .3
                    ang = (r1 - .5) * 2.2 + (math.pi * .25 if r2 > .5 else -math.pi * .75)
                    d = 6 + 34 * out_cubic(q) * (.5 + r3)
                    c.circle("top", x + math.cos(ang) * d, y + math.sin(ang) * d + 30 * q * q, 3 + 2 * r3,
                             fade_gray(230, 1 - q))
                spr = self.slash(min(5, int(a / .05)))      # UT's slash (half size), cut down through the drop
                c.sprites.append((spr, int(x - spr.w / 2 + self.slash_dx[k]), int(y - spr.h / 2), 1.0, None))
