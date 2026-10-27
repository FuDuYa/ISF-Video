"""The ink over Ink Sans (M28-M34).

After the FIGHT cut he is gone - stepped behind a patch of black ink just the
size of him, black on the black above the paper, so there is simply nothing
there. On every note of M31-M32.9375 the aim meter is confirmed once more and one
piece of the patch takes the cut: it does not break like glass, it glitches out
as ink - torn into bands that slip sideways, scanlines, blocks dropping out,
wet edges, ink dragged down - and is gone, showing a bit more of him behind it.
The last piece is gone before the M33 downbeat, and nothing is left of it.

The pieces are the cells of a jittered Voronoi grid inside the patch. Drawn
straight into the frame (full resolution, world coordinates) between Ink Sans
and the free layer (scene.compose_canvas)."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

import choreo as C
from common import W, H, smooth, sprite

BOX = (226, 56, 400, 242)      # Ink Sans with a margin (he breathes a little)
SHEEN = 58                     # the wet edge left where a piece has gone
GLITCH = .22                   # s for a piece to glitch away (shorter for the last ones: all gone by M33)


def _noise(rng, h, w, cell):
    small = rng.random((h // cell + 2, w // cell + 2))
    return np.asarray(Image.fromarray((small * 255).astype(np.uint8)).resize(
        (w + 2 * cell, h + 2 * cell), Image.Resampling.BICUBIC), float)[cell:cell + h, cell:cell + w] / 255


class Curtain:
    def __init__(self, seed=31):
        self.t0, self.t1, self.reveal = C.CURTAIN_START, C.SECTION_END, C.REVEAL
        self.cuts = list(C.CURTAIN_CUTS)
        rng = np.random.default_rng(seed)
        x0, y0, x1, y1 = BOX
        yy, xx = np.mgrid[0:H, 0:W]
        # the patch: Ink Sans's box with a ragged, inky edge
        cx, cy, rx, ry = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
        nz = np.zeros((H, W))
        nz[y0 - 12:y1 + 12, x0 - 12:x1 + 12] = _noise(rng, y1 - y0 + 24, x1 - x0 + 24, 9)
        d = np.maximum(np.abs(xx - cx) - rx, np.abs(yy - cy) - ry)
        self.patch = d < 6 * (nz - .35)
        # its pieces: one per cut
        n = len(self.cuts)
        cols, rows = 6, max(1, math.ceil(n / 6))
        pts = []
        for i in range(cols):
            for j in range(rows):
                pts.append((x0 + (i + rng.uniform(.2, .8)) * (x1 - x0) / cols,
                            y0 + (j + rng.uniform(.2, .8)) * (y1 - y0) / rows))
        pts = np.array(pts[:n])
        wx = np.zeros((H, W)); wy = np.zeros((H, W))
        wx[y0 - 12:y1 + 12, x0 - 12:x1 + 12] = _noise(rng, y1 - y0 + 24, x1 - x0 + 24, 7) - .5
        wy[y0 - 12:y1 + 12, x0 - 12:x1 + 12] = _noise(rng, y1 - y0 + 24, x1 - x0 + 24, 7) - .5
        qx, qy = xx + 26 * wx, yy + 26 * wy                    # (warped: the borders wander like ink)
        dd = (qx[..., None] - pts[:, 0]) ** 2 + (qy[..., None] - pts[:, 1]) ** 2
        self.label = np.where(self.patch, np.argmin(dd, axis=2), -1).astype(np.int16)
        order = rng.permutation(n)
        self.cut_of = {}
        for k, tc in zip(order, self.cuts):
            self.cut_of[int(k)] = tc
        # how long each one glitches: never past the reveal
        self.dur = {k: max(.05, min(GLITCH, self.reveal - .02 - tc)) for k, tc in self.cut_of.items()}
        self.boxes = {}
        for k in range(n):
            ys, xs = np.nonzero(self.label == k)
            if len(ys):
                self.boxes[k] = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
        self._slash = None

    def active(self, t):
        return self.t0 <= t < self.t1

    def slash_frames(self):
        """UT's slash, at its own size."""
        if self._slash is None:
            from common import Sprite
            out = []
            for i in range(1, 7):
                s_ = sprite(f"fight/{i}.png")
                v = np.where(s_.a, 255, 0).astype(np.uint8)
                out.append(Sprite(np.where(s_.a, 210, 0).astype(np.uint8), s_.a.copy()))
            self._slash = out
        return self._slash

    def draw(self, frame, t, pm):
        if not self.active(t) or t >= self.reveal:
            return                                            # (from the reveal on: not a trace of it)
        behind = frame.copy()                                 # him, on the dark, as he really is
        gone, live = [], []
        for k, tc in self.cut_of.items():
            if t >= tc + self.dur[k]:
                gone.append(k)
            elif t >= tc:
                live.append(k)
        stay = self.patch & ~np.isin(self.label, gone + live) & ~pm
        frame[stay] = 0                                       # the patch: pure black, like the dark round it
        for k in live:
            self.glitch(frame, behind, k, t, pm)

    def glitch(self, frame, behind, k, t, pm):
        """One piece going: torn into bands of ink that slip sideways (less and
        less ink as it goes), a shifted copy of what is behind in others, bright
        scanlines, blocks dropping out, drips dragged down - then nothing."""
        tc, dur = self.cut_of[k], self.dur[k]
        a = (t - tc) / dur
        x0, y0, x1, y1 = self.boxes[k]
        m = (self.label[y0:y1, x0:x1] == k) & ~pm[y0:y1, x0:x1]
        h, w = m.shape
        rng = np.random.default_rng(k * 1000 + int(t * 60))  # (it jitters on every frame)
        out = behind[y0:y1, x0:x1].copy()
        ink = 1 - smooth(a)                                    # how much of it is still ink
        y = 0
        while y < h:
            bh = int(rng.integers(2, 7))
            sh = int(rng.integers(-18, 19) * (.3 + a))
            band = slice(y, min(h, y + bh))
            r = rng.random()
            if r < ink:                                        # still ink: black, slipped sideways, ragged ends
                row = np.zeros((min(h, y + bh) - y, w), bool)
                row[:] = m[band]
                row = np.roll(row, sh, axis=1)
                cut = int(rng.integers(0, max(1, w // 4)))
                if sh > 0:
                    row[:, :cut] = False
                else:
                    row[:, w - cut:] = False
                out[band][row] = 0
            elif r < ink + .25:                                # a slipped copy of him behind it
                out[band] = np.roll(behind[y0:y1, x0:x1][band], sh // 2, axis=1)
            y += bh
        # scanlines and dropped-out blocks (bright: they read on the black)
        for _ in range(int(3 + 5 * (1 - a))):
            ry = int(rng.integers(0, h))
            out[ry, :][m[ry, :]] = int(rng.integers(120, 230))
        for _ in range(int(6 * (1 - a))):
            bx, by = int(rng.integers(0, max(1, w - 6))), int(rng.integers(0, max(1, h - 4)))
            blk = m[by:by + 4, bx:bx + 6]
            out[by:by + 4, bx:bx + 6][blk] = int(rng.integers(40, 110))
        sub = frame[y0:y1, x0:x1]
        sub[m] = out[m]
        # ink dragged down out of it: a few drips running over him, shortening as it goes
        rd = np.random.default_rng(k)
        for _ in range(3):
            dx = int(rd.integers(0, w))
            col = np.nonzero(m[:, dx])[0]
            if not len(col):
                continue
            top = y0 + col.max()
            ln = int((10 + 26 * rd.random()) * math.sin(math.pi * min(1.0, a * 1.2)))
            ys = slice(top, min(H, top + ln))
            xs = slice(x0 + dx, min(W, x0 + dx + 2))
            region = frame[ys, xs]
            region[~pm[ys, xs]] = np.minimum(region[~pm[ys, xs]], 20)
        # UT's slash across it as it is cut
        if a < .6:
            spr = self.slash_frames()[min(5, int(a / .6 * 6))]
            from common import paste
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            paste(frame, spr, int(cx - spr.w / 2), int(cy - spr.h / 2))
