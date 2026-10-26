"""Opening (0 - 7.7 s): ink drops form the four buttons, the arena, the soul,
the HUD and finally Ink Sans himself, under a slow perspective camera.

Timings come from the first notes of maidata.txt:
  3h 4h 5h 6h (1.19-2.52 s)  -> MERCY / ITEM / ACT / FIGHT
  3>6 6 5 4    (3.56-4.90 s)  -> the four arena edges
  8b           (5.35 s)       -> the soul (break note)
  4 3 2        (5.94-6.83 s)  -> Ink Sans is painted in: legs, body, head
"""

from __future__ import annotations

import heapq
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import choreo as C
import scene as S
from common import ROOT, W, H, Sprite, clamp, load_cached, out_cubic, paste, save_atomic, smooth, sprite

PAD = 128
PLANE = (W + 2 * PAD, H + 2 * PAD)
CACHE = ROOT / "output" / "cache"

BUTTON_HITS = [C.tm(1), C.tm(1.375), C.tm(1.75), C.tm(2.125)]
EDGE_HITS = [C.tm(3), C.tm(3.375), C.tm(3.75), C.tm(4.125)]
SOUL_HIT = C.tm(4.5)
SANS_HITS = [C.tm(5), C.tm(5.375), C.tm(5.75)]
HUD_START = SOUL_HIT + .12
INTRO_END = 7.0          # camera and plane composition end here
SANS_DONE = 7.75         # ink formation of Ink Sans fully resolved
BUTTON_ORDER = ["mercy1.png", "item1.png", "act1.png", "fight1.png"]

CAMERA_KEYS = [
    (0.00, (555, 448), 1.27, 34, 7, 2),
    (BUTTON_HITS[0], (555, 448), 1.27, 34, 7, 2),
    (BUTTON_HITS[1], (400, 448), 1.25, 31, 4, 1),
    (BUTTON_HITS[2], (240, 448), 1.23, 28, -2, -1),
    (BUTTON_HITS[3], (87, 448), 1.21, 25, -6, -2),
    (2.96, (87, 448), 1.18, 24, -6, -2),
    (3.25, (320, 365), .88, 24, 6, 2),
    (3.60, (320, 310), .90, 22, 8, 1.5),
    (4.92, (320, 298), .94, 10, -5, -2),
    (5.35, (320, 285), 1.08, 0, 0, 0),
    (6.48, (320, 240), 1.00, 0, 0, 0),
    (99.0, (320, 240), 1.00, 0, 0, 0),
]


def organic_arrival(texture: np.ndarray, seed) -> np.ndarray:
    """Random-cost flood fill from one ink drop (normalised 0..1)."""
    h, w = texture.shape
    costs = .28 + 3.9 * (1 - texture) ** 2
    dist = np.full((h, w), np.inf, dtype=np.float64)
    sx, sy = seed
    dist[sy, sx] = 0
    q = [(0.0, sx, sy)]
    steps = [(-1, 0, .66), (1, 0, .66), (0, -1, 1.05), (0, 1, 1.05),
             (-1, -1, 1.25), (1, -1, 1.25), (-1, 1, 1.25), (1, 1, 1.25)]
    cl = costs.tolist()
    dl = dist.tolist()
    while q:
        d, x, y = heapq.heappop(q)
        if d > dl[y][x] + 1e-9:
            continue
        cxy = cl[y][x]
        for dx, dy, st in steps:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                nd = d + st * (cxy + cl[ny][nx]) / 2
                if nd < dl[ny][nx] - 1e-9:
                    dl[ny][nx] = nd
                    heapq.heappush(q, (nd, nx, ny))
    dist = np.array(dl)
    return dist / dist.max()


def smooth_noise(rng, h, w, cell):
    """Blobby value noise without per-pixel grain."""
    g = rng.random((h // cell + 3, w // cell + 3))
    im = Image.fromarray((g * 255).astype("uint8")).resize(((w // cell + 3) * cell, (h // cell + 3) * cell),
                                                          Image.Resampling.BICUBIC)
    return np.asarray(im, float)[:h, :w] / 255


def make_texture(rng, h, w, coarse_shape):
    fine = rng.random((h, w))
    coarse = Image.fromarray((rng.random(coarse_shape) * 255).astype("uint8"))
    coarse = np.asarray(coarse.resize((w, h), Image.Resampling.BILINEAR)) / 255
    return .58 * coarse + .42 * fine


class InkForm:
    """Ink spreads from seed drops over a sprite, then resolves pixel by pixel into it."""

    def __init__(self, spr: Sprite, x: int, y: int, seeds, spread: float, rng, name: str,
                 morph=.35, pad=6, dilate=9):
        self.x, self.y, self.pad = x - pad, y - pad, pad
        h, w = spr.h + 2 * pad, spr.w + 2 * pad
        self.v = np.zeros((h, w), np.uint8)
        self.a = np.zeros((h, w), bool)
        self.v[pad:pad + spr.h, pad:pad + spr.w] = spr.v
        self.a[pad:pad + spr.h, pad:pad + spr.w] = spr.a
        self.tex = make_texture(rng, h, w, (max(2, h // 7), max(2, w // 7)))
        self.rank = .08 + .84 * self.tex
        sil = Image.fromarray(self.a.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(dilate))
        self.sil = np.asarray(sil) > 0
        CACHE.mkdir(parents=True, exist_ok=True)
        arrival = np.full((h, w), np.inf)
        for i, (sx, sy, t_seed) in enumerate(seeds):
            path = CACHE / f"{name}_{i}.npy"
            d = load_cached(path)
            if d is None:
                d = organic_arrival(self.tex, (sx + pad, sy + pad))
                save_atomic(path, d)
            arrival = np.minimum(arrival, t_seed + d * spread)
        self.arrival = arrival
        self.morph = morph
        self.done = float(arrival[self.sil].max()) + .10 + morph

    def draw(self, frame: np.ndarray, t: float, ox=0, oy=0, gain=1.0):
        if t >= self.done:
            region = frame[self.y + oy:self.y + oy + self.v.shape[0], self.x + ox:self.x + ox + self.v.shape[1]]
            region[self.a] = (self.v[self.a] * gain).astype(np.uint8)
            return
        wet = (self.arrival <= t) & self.sil
        if not wet.any():
            return
        resolved = t >= self.arrival + .10 + self.rank * self.morph
        ink = wet & ~resolved & (self.tex > .12)
        formed = wet & resolved & self.a
        region = frame[self.y + oy:self.y + oy + self.v.shape[0], self.x + ox:self.x + ox + self.v.shape[1]]
        region[ink] = np.maximum(region[ink], (62 + self.tex[ink] * 72).astype(np.uint8))
        region[formed] = self.v[formed]


class ButtonInk:
    """One falling drop becomes a UT button: the button's own line-work spreads
    outward from where the drop lands, along an irregular wet front. No ink
    blob is laid over it; only a tiny splash marks the impact."""

    PAD = 8

    def __init__(self, spr: Sprite, x: int, y: int, t_hit: float, rng, name: str):
        P = self.PAD
        self.x, self.y, self.th = x - P, y - P, t_hit
        H2, W2 = (spr.h + 2 * P) // 2, (spr.w + 2 * P) // 2
        coarse = smooth_noise(rng, H2, W2, 5) * .65 + smooth_noise(rng, H2, W2, 2) * .35
        coarse = (coarse - coarse.min()) / (np.ptp(coarse) + 1e-9)
        CACHE.mkdir(parents=True, exist_ok=True)
        path = CACHE / f"{name}_spread.npy"
        arr = load_cached(path)
        if arr is None:
            arr = organic_arrival(coarse, (W2 // 2, H2 // 2))
            save_atomic(path, arr)
        self.arrival = np.kron(arr, np.ones((2, 2)))           # 2-px blocky front
        self.full_a = np.zeros((H2 * 2, W2 * 2), bool)
        self.full_a[P:P + spr.h, P:P + spr.w] = spr.a
        self.full_v = np.zeros(self.full_a.shape, np.uint8)
        self.full_v[P:P + spr.h, P:P + spr.w] = spr.v
        self.stroke = self.full_a & (self.full_v > 20)
        self.glint = np.minimum(255, self.full_v.astype(int) + 90).astype(np.uint8)
        ang = np.linspace(0, math.tau, 7, endpoint=False) + rng.uniform(-.3, .3, 7)
        self.drops = [(math.cos(a), math.sin(a) * .6, rng.uniform(10, 18)) for a in ang]
        self.dur = .52
        self.done = t_hit + self.dur + .02

    def draw(self, frame: np.ndarray, t: float, ox=0, oy=0, gain=1.0):
        age = t - self.th
        if age < 0:
            return
        Hh, Ww = self.full_a.shape
        X, Y = self.x + ox, self.y + oy
        region = frame[Y:Y + Hh, X:X + Ww]
        if t >= self.done:
            region[self.full_a] = self.full_v[self.full_a]
            return
        x = clamp(age / self.dur)
        u = (1 - (1 - x) ** 1.5) * 1.02
        reached = self.full_a & (self.arrival <= u)
        region[reached] = self.full_v[reached]
        front = self.stroke & reached & (self.arrival > u - .09)
        region[front] = self.glint[front]
        if age < .12:                       # tiny impact splash
            cx, cy = Ww // 2, Hh // 2
            k = out_cubic(age / .12)
            v = int(220 * (1 - age / .12)) + 30
            region[cy - 1:cy + 1, cx - 1:cx + 1] = v
            for dx, dy, dist in self.drops:
                px = int(cx + dx * dist * k) // 2 * 2
                py = int(cy + dy * dist * k) // 2 * 2
                if 0 <= px < Ww - 1 and 0 <= py < Hh - 1:
                    region[py:py + 2, px:px + 2] = v


# ------------------------------------------------------------------ camera
class Camera:
    def __init__(self, t: float, hits):
        keys = CAMERA_KEYS
        a = b = keys[0]
        for first, second in zip(keys, keys[1:]):
            a, b = first, second
            if t <= second[0]:
                break
        u = smooth((t - a[0]) / max(1e-4, b[0] - a[0]))
        mix = lambda i: a[i] + (b[i] - a[i]) * u
        self.focus = np.array(a[1], float) * (1 - u) + np.array(b[1], float) * u
        self.focal = 800.0
        self.distance = self.focal / mix(2)
        pitch, yaw, roll = (math.radians(mix(i)) for i in (3, 4, 5))
        cp, sp, cy, sy, cr, sr = (math.cos(pitch), math.sin(pitch), math.cos(yaw),
                                  math.sin(yaw), math.cos(roll), math.sin(roll))
        rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
        self.rot = rz @ ry @ rx
        self.recoil = sum(2.0 * math.sin((t - h) * 52) * math.exp(-(t - h) * 24)
                          for h in hits if 0 <= t - h < .18)
        self.identity = (abs(mix(2) - 1) < 1e-6 and abs(mix(3)) < 1e-6 and abs(mix(4)) < 1e-6
                         and abs(mix(5)) < 1e-6 and np.allclose(self.focus, (320, 240))
                         and abs(self.recoil) < .5)

    def project(self, x, y, z=0.0):
        p = self.rot @ np.array([x - self.focus[0], y - self.focus[1], z])
        s = self.focal / max(100, self.distance - p[2])
        return W / 2 + p[0] * s, H / 2 + p[1] * s + self.recoil, s

    def warp(self, plane: Image.Image) -> Image.Image:
        corners = [(0, 0), (plane.width, 0), (plane.width, plane.height), (0, plane.height)]
        eq, val = [], []
        for sx, sy in corners:
            dx, dy, _ = self.project(sx - PAD, sy - PAD)
            eq += [[dx, dy, 1, 0, 0, 0, -sx * dx, -sx * dy], [0, 0, 0, dx, dy, 1, -sy * dx, -sy * dy]]
            val += [sx, sy]
        coef = np.linalg.solve(np.array(eq), np.array(val))
        return plane.transform((W, H), Image.Transform.PERSPECTIVE, coef,
                               Image.Resampling.NEAREST, fillcolor=0)


# ------------------------------------------------------------------ intro
class Intro:
    def __init__(self):
        rng = np.random.default_rng(20261024)
        self.hits = BUTTON_HITS + EDGE_HITS + [SOUL_HIT]
        by = {b["sprite"]: b for b in S.HUD["buttons"]}
        self.buttons = []
        for i, name in enumerate(BUTTON_ORDER):
            b = by[name]
            spr = sprite(name)
            x, y = b["position"]
            form = ButtonInk(spr, x, y, BUTTON_HITS[i], rng, f"btn{i}")
            self.buttons.append((form, (x + 55, y + 21)))
        # HUD row (name, LV, HP tag, bar, numbers) forms after the soul appears
        hud = np.zeros((H, W), np.uint8)
        S.draw_hud(hud, S.HUD["hp"], buttons=False)
        ys, xs = np.nonzero(hud)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        hspr = Sprite(hud[y0:y1, x0:x1].copy(), hud[y0:y1, x0:x1] > 0)
        self.hud = InkForm(hspr, x0, y0, [(0, hspr.h // 2, HUD_START),
                                          (hspr.w // 2, hspr.h // 2, HUD_START + .12)],
                           .55, rng, "hud", morph=.3)
        # Ink Sans painted in from three drops
        comp = S.sans_composite(head="head_close_eyes.png", body="body_nobrush.png")
        self.sans_spr = comp
        ox, oy = S.INK["origin"]
        seeds = [(ox - comp.x, oy - 8 - comp.y, SANS_HITS[0]),          # feet
                 (ox - comp.x, oy - 70 - comp.y, SANS_HITS[1]),         # chest
                 (ox - comp.x, oy - 130 - comp.y, SANS_HITS[2])]        # skull
        self.sans = InkForm(comp, comp.x, comp.y, seeds, .55, rng, "sans", morph=.32, pad=6)
        # the first speech bubble and its text are painted in alongside him
        import dialogue as D
        dlg = D.Dialogue()
        text = D.LINES[0][2]
        layer = dlg.static(text)
        ys, xs = np.nonzero(layer)
        by0, by1, bx0, bx1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        bspr = Sprite(layer[by0:by1, bx0:bx1].copy(), np.ones((by1 - by0, bx1 - bx0), bool)
                      & (layer[by0:by1, bx0:bx1] > 0) | self._text_mask(layer, by0, by1, bx0, bx1))
        tip_y = D.TAIL_TIP[1] - by0
        self.bubble = InkForm(bspr, bx0, by0, [(1, tip_y, SANS_HITS[0] + .12),
                                               (bspr.w // 2, bspr.h // 2, SANS_HITS[1])],
                              .6, rng, f"bubble_{abs(hash(text)) % 99991}", morph=.32, pad=6)
        self.edge_maps = self._edges(rng)

    @staticmethod
    def _text_mask(layer, y0, y1, x0, x1):
        """Black text pixels inside the white bubble are part of the sprite too."""
        from scipy import ndimage as ndi
        white = layer[y0:y1, x0:x1] > 0
        return ndi.binary_fill_holes(white)

    def _edges(self, rng):
        x, y, w, h = C.box_at(0)
        border = C.BORDER
        ex, ey = np.meshgrid(np.arange(w + 16), np.arange(h + 16))
        maps = []
        for i in range(4):
            if i == 0:
                along, off, length = ex - 8, np.abs(ey - (8 + (border - 1) / 2)), w
                target = (ex >= 8) & (ex < 8 + w) & (ey >= 8) & (ey < 8 + border)
            elif i == 1:
                along, off, length = ey - 8, np.abs(ex - (8 + w - (border + 1) / 2)), h
                target = (ex >= 8 + w - border) & (ex < 8 + w) & (ey >= 8) & (ey < 8 + h)
            elif i == 2:
                along, off, length = 8 + w - 1 - ex, np.abs(ey - (8 + h - (border + 1) / 2)), w
                target = (ex >= 8) & (ex < 8 + w) & (ey >= 8 + h - border) & (ey < 8 + h)
            else:
                along, off, length = 8 + h - 1 - ey, np.abs(ex - (8 + (border - 1) / 2)), h
                target = (ex >= 8) & (ex < 8 + border) & (ey >= 8) & (ey < 8 + h)
            tex = make_texture(rng, h + 16, w + 16, (19, 19))
            maps.append(dict(along=along, off=off, length=length, target=target, tex=tex,
                             rank=.10 + .80 * tex))
        return maps

    # -------------------------------------------------------------- pieces
    def draw_edges(self, plane, t):
        x, y, w, h = C.box_at(0)
        for i, hit in enumerate(EDGE_HITS):
            age = t - hit
            if age < 0:
                continue
            e = self.edge_maps[i]
            if age >= .46:
                vals = e["target"].astype(np.uint8) * 255
            else:
                front = e["length"] * smooth(age / .27)
                reached = (e["along"] >= 0) & (e["along"] < e["length"])
                reached &= e["along"] <= front + (e["tex"] - .5) * 5
                wake = np.clip((front - e["along"]) / 50, 0, 1)
                settle = smooth((age - .27) / .19)
                resolved = np.maximum(wake, settle) >= e["rank"]
                vals = np.zeros(e["target"].shape, np.uint8)
                ink = reached & ~resolved & (e["off"] < 4 + e["tex"] * 5)
                vals[ink] = (66 + e["tex"][ink] * 72).astype(np.uint8)
                vals[reached & resolved & e["target"]] = 255
            region = plane[y + PAD - 8:y + PAD - 8 + h + 16, x + PAD - 8:x + PAD - 8 + w + 16]
            np.maximum(region, vals, out=region)

    def draw_soul(self, plane, t):
        age = t - SOUL_HIT
        cx, cy = S.SCENE["soul_start"]
        gather = .23
        if -gather <= age < 0:
            img = Image.fromarray(plane)
            from PIL import ImageDraw
            d = ImageDraw.Draw(img)
            u = smooth((age + gather) / gather)
            for i in range(8):
                a = i * math.tau / 8 + .35 * (1 - u)
                outer = 61 * (1 - u) + 4
                inner = outer + 14 * (1 - u)
                p1 = (round(cx + math.cos(a) * outer + PAD), round(cy + math.sin(a) * outer + PAD))
                p2 = (round(cx + math.cos(a + .05) * inner + PAD), round(cy + math.sin(a + .05) * inner + PAD))
                d.line((p1, p2), fill=round(65 + 44 * u), width=2)
            plane[:] = np.asarray(img)
        if age >= 0:
            size = 16 + round(6 * (1 - out_cubic(age / .18)))
            s = sprite("player.png")
            img = Image.fromarray(s.v).resize((size, size), Image.Resampling.NEAREST)
            al = Image.fromarray(s.a.astype(np.uint8) * 255).resize((size, size), Image.Resampling.NEAREST)
            big = Sprite(np.asarray(img), np.asarray(al) > 0)
            paste(plane, big, cx - size // 2 + PAD, cy - size // 2 + PAD)

    def draw_drops(self, frame, cam, t):
        from PIL import ImageDraw
        img = Image.fromarray(frame)
        d = ImageDraw.Draw(img)
        dur = .34
        for i, hit in enumerate(BUTTON_HITS):
            age = t - hit
            if not -dur <= age < 0:
                continue
            u = (age + dur) / dur
            cx, cy = self.buttons[i][1]
            z = 170 * (1 - u * u)
            px, py, sc = cam.project(cx + 10 * (1 - u), cy, z)
            r = max(2, round(2.4 * sc))
            tail = round((7 + 7 * u) * sc)
            px, py = round(px), round(py)
            d.polygon([(px - r, py), (px, py - tail), (px + r, py)], fill=112)
            d.ellipse((px - r, py - r, px + r, py + r), fill=137)
            d.rectangle((px - 1, py - r, px, py - r + 2), fill=184)
            for k in range(2):
                by = py - tail - 6 - k * 7
                d.rectangle((px, by, px + 1, by + 2), fill=round(69 * u))
        frame[:] = np.asarray(img)

    def flash(self, frame, t):
        b = 0
        for h in BUTTON_HITS:      # only the four ink drops hit; the arena edges are drawn
            age = t - h
            if 0 <= age < .13:
                b = max(b, round(28 * (1 - smooth(age / .13))))
        if b:
            frame[frame == 0] = b

    # -------------------------------------------------------------- frame
    def frame(self, t: float) -> np.ndarray:
        plane = np.zeros((PLANE[1], PLANE[0]), np.uint8)
        for form, _ in self.buttons:
            form.draw(plane, t, PAD, PAD)
        self.draw_edges(plane, t)
        self.draw_soul(plane, t)
        self.hud.draw(plane, t, PAD, PAD)
        self.sans.draw(plane, t, PAD, PAD)
        self.bubble.draw(plane, t, PAD, PAD)
        cam = Camera(t, BUTTON_HITS + [SOUL_HIT])
        if cam.identity:
            frame = plane[PAD:PAD + H, PAD:PAD + W].copy()
        else:
            frame = np.asarray(cam.warp(Image.fromarray(plane))).copy()
        self.draw_drops(frame, cam, t)
        return frame

    def bubble_hook(self, frame, t) -> bool:
        if t >= self.bubble.done:
            return False
        self.bubble.draw(frame, t)
        return True

    def sans_hook(self, frame, t) -> bool:
        """Draw the forming Ink Sans in the main renderer until formation is done."""
        if t >= SANS_DONE:
            return False
        self.sans.draw(frame, t)
        return True
