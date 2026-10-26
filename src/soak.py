"""The arena soaks through and becomes a sheet of paper (end of M15).

Once the pool has engulfed the arena (choreo.ENGULF), on three triplet notes
the white frame drinks the ink: grey stains creep through it, ink bleeds out
of its edges and runs down the outside. On the next three notes the inside
turns to paper along a ragged wet front - darker where it has only just
soaked, a dark tide line at the front, faint tide marks where each pulse
stopped. On the M16 downbeat the frame dries white again: the canvas is set.
"""

from __future__ import annotations

import numpy as np

import choreo as C
from common import clamp, load_cached, out_cubic, save_atomic, smooth

BOX = C.box_rect((360, 150))
B = C.BORDER


def _maps():
    from intro import CACHE, organic_arrival, smooth_noise
    x, y, w, h = BOX
    iw, ih = w - 2 * B, h - 2 * B
    rng = np.random.default_rng(33)
    hw, hh = iw // 2, ih // 2
    coarse = smooth_noise(rng, hh, hw, 6) * .6 + smooth_noise(rng, hh, hw, 2) * .4
    coarse = (coarse - coarse.min()) / (np.ptp(coarse) + 1e-9)
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"canvas_soak_{hw}x{hh}.npy"
    arr = load_cached(path)
    if arr is None:
        a1 = organic_arrival(coarse, (hw // 2, hh // 2))
        a2 = organic_arrival(coarse, (hw // 2 - 34, hh // 2 + 10))
        a3 = organic_arrival(coarse, (hw // 2 + 38, hh // 2 - 8))
        arr = np.minimum(np.minimum(a1, a2 + .12), a3 + .2)
        arr = (arr - arr.min()) / (np.ptp(arr) + 1e-9)
        save_atomic(path, arr)
    arr = arr.repeat(2, 0).repeat(2, 1)[:ih, :iw]
    # stains through the frame (whole box rect, 2-px grain)
    st = smooth_noise(rng, h // 2 + 8, w // 2 + 8, 3)[4:4 + h // 2, 4:4 + w // 2]
    st = (st - st.min()) / (np.ptp(st) + 1e-9)
    st = st.repeat(2, 0).repeat(2, 1)[:h, :w]
    drips = rng.uniform(0, 1, (12, 4))            # side, position, length, delay
    return arr, st, drips


class Soak:
    def __init__(self):
        self.arr, self.stain, self.drips = _maps()
        self.p0 = C.PAPER_STEPS[0]

    # ------------------------------------------------------------ timing
    def frame_k(self, t):
        """How soaked the frame is (0..1): three steps, then it dries on the downbeat."""
        if t < C.SOAK_STEPS[0] or t >= C.CANVAS_SET + .14:
            return 0.0
        k = sum(out_cubic(clamp((t - s) / .07)) for s in C.SOAK_STEPS) / 3
        return k * (1 - smooth((t - C.CANVAS_SET) / .12))

    def paper_r(self, t):
        if t < self.p0:
            return 0.0
        return sum(out_cubic(clamp((t - s) / .085)) for s in C.PAPER_STEPS) / 3 * 1.02

    # ------------------------------------------------------------ drawing
    def border(self, frame, t, box):
        k = self.frame_k(t)
        if k <= 0 or tuple(box) != tuple(BOX):
            return
        x, y, w, h = box
        rect = frame[y:y + h, x:x + w]
        ring = np.zeros((h, w), bool)
        ring[:B], ring[-B:], ring[:, :B], ring[:, -B:] = True, True, True, True
        wet = ring & (self.stain < k * 1.05)
        deep = ring & (self.stain < k * .6)
        rect[wet] = 150
        rect[deep] = 96
        # ink bleeding out through the frame: a ragged grey halo just outside it
        halo_w = int(round(6 * k / 2)) * 2
        if halo_w:
            out = frame[y - halo_w:y + h + halo_w, x - halo_w:x + w + halo_w]
            m = np.zeros(out.shape, bool)
            m[:halo_w], m[-halo_w:], m[:, :halo_w], m[:, -halo_w:] = True, True, True, True
            st = np.pad(self.stain, halo_w, mode="edge")
            bleed = m & (st < k * .5) & (out == 0)
            bleed[-halo_w:] = False                  # keep the HUD rows clean
            out[bleed] = 70
        # ink running down the outside of the side frames
        for side, pos, ln, dl in self.drips:
            u = clamp((k - dl * .5) / .5)
            if u <= 0:
                continue
            px = x - 3 if side < .5 else x + w + 1
            py0 = y + int(pos * (h - 40))
            L = int((8 + 34 * ln) * out_cubic(u))
            frame[py0:py0 + L, px:px + 2] = 120
            frame[py0 + L:py0 + L + 4, px - 1:px + 3] = 150

    def paper(self, region, t, box):
        """Turn the arena interior into paper where the wet front has passed."""
        if tuple(box) != tuple(BOX) or region.shape != self.arr.shape:
            return
        from scene import PAPER
        x, y, w, h = box
        r = self.paper_r(t)
        if r <= 0:
            return
        arr = self.arr
        dist = r - arr
        paper = dist >= 0
        g = PAPER[y + B:y + h - B, x + B:x + w - B]
        dry = smooth((t - C.PAPER_STEPS[-1] - .02) / .09)
        q = np.clip(dist / .32, 0, 1)
        wet = (110 * (1 - q * q * (3 - 2 * q)) * (1 - dry)).astype(np.int16)
        val = 255 - g - wet
        # faint tide marks where each pulse stopped
        for j in (1, 2):
            tide = paper & (np.abs(arr - j / 3) < .008)
            val[tide] -= int(46 * (1 - dry * .6))
        region[paper] = np.clip(val[paper], 0, 255).astype(np.uint8)
        rim = paper & (dist < .025) & (r < 1)
        region[rim] = 64
        region[:2, :][paper[:2, :]] = np.minimum(region[:2, :][paper[:2, :]], 150)
        region[:, :2][paper[:, :2]] = np.minimum(region[:, :2][paper[:, :2]], 170)

    def soul_gain(self, t, xy, box):
        """The soul is dimmed while it is inside the ink, until the paper reaches it."""
        if not (C.ENGULF <= t < C.CANVAS_SET) or xy is None:
            return 1.0
        x, y, w, h = box
        ix, iy = int(xy[0]) - x - B, int(xy[1]) - y - B
        if 0 <= iy < self.arr.shape[0] and 0 <= ix < self.arr.shape[1]:
            if self.arr[iy, ix] <= self.paper_r(t):
                return 1.0
        return .55
