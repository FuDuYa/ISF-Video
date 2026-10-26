"""The ink curtain (M28-M33.5).

After the FIGHT cut the black over the paper - where Ink Sans used to stand - is
not empty: it is a curtain of black ink, and he is hiding behind it. It looks
just like the dark until the player cuts it: on every note of M31-M32.9375 one
piece of it takes UT's slash and comes away, falling past the paper with its wet
edges catching the light, and behind it there is a different, dark-grey dark.
On the M33 downbeat the rest falls at once and Ink Sans is there.

The pieces are the cells of a jittered Voronoi grid, so every cut edge is
straight. Drawn straight into the frame (full resolution, world coordinates)
between Ink Sans and the free layer (scene.compose_canvas)."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

import choreo as C
from chart import parse_chart
from common import ROOT, W, H, clamp, smooth, sprite

BACK = 26            # the real dark behind the curtain
SHEEN = 70           # a wet edge of the curtain where a piece has come away
FALL_G = 2400.0      # px/s^2


class Curtain:
    def __init__(self, seed=31):
        self.t0, self.t1, self.reveal = C.CURTAIN_START, C.SECTION_END, C.REVEAL
        notes, _ = parse_chart(ROOT / "maidata.txt", C.OFFSET)
        self.cuts = sorted({round(n.t, 4) for n in notes if C.tm(31) - 1e-3 <= n.t < C.REVEAL - 1e-3})
        rng = np.random.default_rng(seed)
        cols, rows = 7, 4
        pts = []
        for i in range(cols):
            for j in range(rows):
                pts.append(((i + rng.uniform(.2, .8)) * W / cols, -40 + (j + rng.uniform(.2, .8)) * 290 / rows))
        self.seeds = np.array(pts)
        yy, xx = np.mgrid[0:H, 0:W]
        d = (xx[..., None] - self.seeds[:, 0]) ** 2 + (yy[..., None] - self.seeds[:, 1]) ** 2
        self.label = np.argmin(d, axis=2).astype(np.int16)
        self.label[yy > 300] = -1                            # (only the dark above the paper)
        n = len(self.seeds)
        # the order the pieces come away: the ones in front of Ink Sans mostly
        # later, so that he shows bit by bit
        over = np.array([1.0 if 220 < p[0] < 420 and p[1] < 240 else 0.0 for p in self.seeds])
        key = rng.random(n) + .55 * over
        order = list(np.argsort(key))
        self.cut_of = {}
        for k, tc in zip(order, self.cuts):
            self.cut_of[int(k)] = tc
        rest = order[len(self.cuts):]
        for i, k in enumerate(rest):                          # the rest falls on the reveal, all at once
            self.cut_of[int(k)] = self.reveal + .012 * i
        self.slashed = {int(k) for k in order[:len(self.cuts)]}
        self.boxes = {}
        for k in range(n):
            ys, xs = np.nonzero(self.label == k)
            if len(ys):
                y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                self.boxes[k] = (x0, y0, self.label[y0:y1, x0:x1] == k)
        self.drift = rng.uniform(-60, 60, n)
        self.spin = rng.uniform(-90, 90, n)
        self._slash = None

    def active(self, t):
        return self.t0 <= t < self.t1

    def slash_frames(self):
        """UT's slash, three times its size (a piece of the curtain is big)."""
        if self._slash is None:
            from common import Sprite
            out = []
            for i in range(1, 7):
                s_ = sprite(f"fight/{i}.png")
                v = np.where(s_.a, 255, 0).astype(np.uint8)
                out.append(Sprite(np.kron(v, np.ones((3, 3), np.uint8)), np.kron(s_.a, np.ones((3, 3), bool))))
            self._slash = out
        return self._slash

    def draw(self, frame, t, pm):
        if not self.active(t):
            return
        region = ~pm
        region[300:] = False
        gone_ids = [k for k, tc in self.cut_of.items() if t >= tc + .04]
        gone = region & np.isin(self.label, gone_ids) if gone_ids else np.zeros_like(region)
        stay = region & ~gone & (self.label >= 0)
        frame[gone] = np.maximum(frame[gone], BACK)           # the real dark behind it (and him)
        frame[stay] = 0                                       # the curtain: pure black ink
        if gone.any():                                        # its cut edges catch the light
            edge = np.zeros_like(stay)
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1), (2, 0), (0, 2), (-2, 0), (0, -2)):
                edge |= np.roll(gone, (dy, dx), axis=(0, 1))
            frame[stay & edge] = SHEEN
        for k, tc in self.cut_of.items():
            a = t - tc
            if 0 <= a < .05 and k in self.slashed:            # the piece flashes as it is cut
                x0, y0, m = self.boxes[k]
                sub = frame[y0:y0 + m.shape[0], x0:x0 + m.shape[1]]
                rim = m & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1))
                sub[rim] = 255
            if .04 <= a < .7:                                 # it comes away and falls
                self.draw_piece(frame, k, a - .04, pm)
            if 0 <= a < .3 and k in self.slashed:             # UT's slash, cut through it
                x0, y0, m = self.boxes[k]
                cx, cy = x0 + m.shape[1] / 2, y0 + m.shape[0] / 2
                spr = self.slash_frames()[min(5, int(a / .05))]
                from common import paste
                paste(frame, spr, int(cx - spr.w / 2), int(cy - spr.h / 2))

    def draw_piece(self, frame, k, a, pm):
        x0, y0, m = self.boxes[k]
        h, w = m.shape
        ang = self.spin[k] * a
        img = Image.fromarray((m * 255).astype(np.uint8)).rotate(ang, resample=Image.Resampling.NEAREST, expand=True)
        mm = np.asarray(img) > 127
        hh, ww = mm.shape
        cx = x0 + w / 2 + self.drift[k] * a
        cy = y0 + h / 2 + .5 * FALL_G * a * a
        X0, Y0 = int(round(cx - ww / 2)), int(round(cy - hh / 2))
        xa, ya, xb, yb = max(0, X0), max(0, Y0), min(W, X0 + ww), min(H, Y0 + hh)
        if xa >= xb or ya >= yb:
            return
        sub = mm[ya - Y0:yb - Y0, xa - X0:xb - X0]
        g = 1 - smooth((a - .3) / .3)                         # it soaks away as it goes
        rim = sub & ~(np.roll(sub, 1, 0) & np.roll(sub, -1, 0) & np.roll(sub, 1, 1) & np.roll(sub, -1, 1))
        tgt = frame[ya:yb, xa:xb]
        on_paper = pm[ya:yb, xa:xb]
        body = sub & ~rim
        # over the dark it is black ink (its wet edge catches the light); over the
        # paper it is already soaking in - a grey stain, never black, gone at once
        dark = body & ~on_paper
        tgt[dark] = 0
        wet = body & on_paper
        tgt[wet] = np.minimum(tgt[wet], int(255 - 120 * g * (1 - smooth(clamp((cy - 240) / 120)) * .5)))
        if g > 0:
            tgt[rim] = np.where(on_paper[rim], int(255 - 150 * g), int(150 * g)).astype(np.uint8)
