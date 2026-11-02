"""Plan the player's soul movement through every attack.

1. Render the hazard mask of every frame (no art needed).
2. Backward pass: S[f] = positions from which the soul can survive to the end,
   moving at UT speed (2 px per frame at 60 fps, 8 directions).
3. Forward pass: pick moves that stay inside S while keeping clear of ink,
   with inertia so the motion reads like a player holding arrow keys.
If no surviving move exists the soul takes a hit (HP drops, 1 s of blinking).
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from scipy import ndimage as ndi

import choreo as C
import scene as S
from common import ROOT, FPS

X0, X1, Y0, Y1 = 0, 640, 226, 392       # crop window that contains every arena (set per segment, see SEGMENTS)
BAND = (0, 640, 226, 392)                # ...the battle box / the canvas band, full width
WHOLE = (0, 640, 0, 480)                 # ...the descent: the arena is the whole height of the screen
T_START = 7.0
T_END = 146.5
HALF = 4                                 # 8x8 hitbox
# while the canvas is the whole picture (grid, tunnel, cube - the camera is close)
# the soul keeps its whole sprite clear of the ink: it never seems to touch black
# without being hurt
WIDE = (C.tm(20.75) + .25, C.tm(24.75))
HALF_WIDE = 12
# (and in the descent's rings and spike seas, threading close past the ink: the
# whole heart clear of it)
HALVES = [(C.tm(33), C.tm(37.875), 8),                # (M33-M37: the whole heart - it must never touch the black)
          (WIDE[0], WIDE[1], HALF_WIDE), (C.RING_DRAWS[0], C.RING_FIRES[-1] + .6, 7),
          (C.BREAK_T - .2, C.tm(C.CORE_M[-1] + .6), 7)]
SPEED = 2
DAMAGE = 4
INVULN = 1.0


def frame_range():
    return int(round(T_START * FPS)), int(round(T_END * FPS))


def segments():
    """The plan is made in independent pieces, each with its own crop window: the
    soul's frame is reset where one starts (C.SOUL_RESETS), so none depends on the
    next. (The descent needs the whole screen; a window that big for the whole song
    would not fit in memory.)"""
    f0, f1 = frame_range()
    a, b = int(round(C.SINK0 * FPS)), int(round(C.NEW_END * FPS))
    return [(f0, a - 1, BAND), (a, b - 1, WHOLE), (b, f1, BAND)]


def set_window(win):
    global X0, X1, Y0, Y1
    X0, X1, Y0, Y1 = win


def valid_mask(box):
    x, y, w, h = box
    b = C.BORDER
    m = np.zeros((Y1 - Y0, X1 - X0), bool)
    xa, xb = max(X0, x + b + 8), min(X1 - 1, x + w - b - 8)
    ya, yb = max(Y0, y + b + 8), min(Y1 - 1, y + h - b - 8)
    m[ya - Y0:yb - Y0 + 1, xa - X0:xb - X0 + 1] = True
    return m


class Valid:
    """Where the soul may be in each frame: inside the arena. Only the boxes are
    kept (a mask per frame of the full width would take too much memory); the
    masks are made when asked for (a few are cached)."""

    def __init__(self, boxes):
        self.boxes = boxes
        self.cache = {}

    def __len__(self):
        return len(self.boxes)

    def __getitem__(self, i):
        b = self.boxes[i]
        if b not in self.cache:
            if len(self.cache) > 64:
                self.cache.clear()
            self.cache[b] = valid_mask(b)
        return self.cache[b]


def safe_masks(tl, f0, f1):
    n = f1 - f0 + 1
    safe = np.zeros((n, Y1 - Y0, X1 - X0), bool)
    boxes = []
    for i in range(n):
        t = (f0 + i) / FPS
        box = C.box_at(t)
        ctx = S.render_attacks(t, tl, box)
        haz = S.hazard_inside(ctx, box)[Y0:Y1, X0:X1]
        half = next((h for a, b, h in HALVES if a <= t < b), HALF)
        unsafe = ndi.maximum_filter(haz, size=2 * half + 1) if haz.any() else haz
        boxes.append(tuple(int(v) for v in box))
        v = valid_mask(boxes[-1])
        safe[i] = v & ~unsafe
        if i % 1200 == 0:
            print(f"  hazards {t:6.1f}s", flush=True)
    return safe, Valid(boxes)


def backward(safe, valid):
    n = len(safe)
    surv = np.zeros_like(safe)
    surv[-1] = safe[-1]
    broken = []
    foot = np.ones((2 * SPEED + 1, 2 * SPEED + 1), bool)
    for i in range(n - 2, -1, -1):
        reach = ndi.binary_dilation(surv[i + 1], structure=foot)
        s = safe[i] & reach
        if not s.any():
            broken.append(i)
            s = safe[i] if safe[i].any() else valid[i]
        surv[i] = s
    return surv, broken[::-1]


MOVES = [(dx, dy) for dx in (-SPEED, 0, SPEED) for dy in (-SPEED, 0, SPEED)]


def forward(safe, surv, valid, start, f0, start_hp):
    n = len(safe)
    pos = np.zeros((n, 2), np.int32)
    hp = np.zeros(n, np.int32)
    vis = np.ones(n, bool)
    p = (start[0] - X0, start[1] - Y0)
    p = (min(max(p[0], 0), X1 - X0 - 1), min(max(p[1], 0), Y1 - Y0 - 1))
    cur_hp = start_hp
    inv_until = -1
    last = (0, 0)
    hold = 0
    hits = []
    clear_cache = {}

    def clearance(i):
        if i >= n:
            i = n - 1
        if i not in clear_cache:
            unsafe = ~safe[i] & valid[i]
            clear_cache.clear() if len(clear_cache) > 40 else None
            clear_cache[i] = ndi.distance_transform_cdt(~unsafe, metric="chessboard") \
                if unsafe.any() else np.full(unsafe.shape, 99, np.int32)
        return clear_cache[i]

    def clamp_into(q, i):
        v = valid[i]
        ys, xs = np.nonzero(v.any(1))[0], np.nonzero(v.any(0))[0]
        if len(ys) == 0:
            return q
        return (int(min(max(q[0], xs[0]), xs[-1])), int(min(max(q[1], ys[0]), ys[-1])))

    forced = {int(round(t * FPS)) - f0: dmg for t, dmg in getattr(C, "FORCED_HITS", [])}
    resets = {int(round(t * FPS)) - f0: q for t, q in getattr(C, "SOUL_RESETS", [])}
    for i in range(n):
        if i in resets:                      # the soul's frame changes (the descent): it is here in the new one
            p = (resets[i][0] - X0, resets[i][1] - Y0)
        pos[i] = (p[0] + X0, p[1] + Y0)
        if i in forced:                      # a scripted hit (the story makes it land)
            cur_hp = max(1, cur_hp - forced[i])
            inv_until = i + int(INVULN * FPS)
            hits.append((f0 + i) / FPS)
        hp[i] = cur_hp
        if i <= inv_until:
            vis[i] = ((inv_until - i) // 3) % 2 == 0
        if i == n - 1:
            break
        j = i + 1
        c1 = clearance(j)
        c2 = clearance(j + 10)
        c3 = clearance(j + 22)
        cx, cy = np.mean(np.nonzero(valid[j].any(0))[0]), np.mean(np.nonzero(valid[j].any(1))[0])
        tj = (f0 + j) / FPS
        run = next((w for a, b, w in getattr(C, "RUNS", []) if a <= tj < b), 0)
        line = next((f for a, b, f in getattr(C, "LINES", []) if a <= tj < b), None)
        lx = line(p[1] + Y0 + run * 14) - X0 if line is not None and run else None   # (the row 14 px ahead)
        goal = next((g for a, b, g in getattr(C, "GOALS", []) if a <= tj < b), None)
        held = any(a <= tj < b for a, b in getattr(C, "HOLDS", []))
        best, best_s = None, -1e9
        for mv in MOVES:
            q = clamp_into((p[0] + mv[0], p[1] + mv[1]), j)
            if not (0 <= q[0] < X1 - X0 and 0 <= q[1] < Y1 - Y0):
                continue
            ok = surv[j][q[1], q[0]]
            s = 0.0 if ok else -1000.0
            for cl, wt in ((c1, 1.0), (c2, .6), (c3, .3)):
                s -= wt * max(0, 13 - cl[q[1], q[0]])
            # drift back toward the middle like a player re-centring (dead zone 18 px)
            if run:                          # (a chase: it runs flat out the way it is driven, from the off)
                if lx is not None:           # (through the gates: weaving for the gaps ahead)
                    s -= 2.5 * max(0.0, abs(q[0] - lx) - 3)
                else:
                    s -= .6 * max(0.0, abs(q[0] - cx) - 18)
                s += 8.0 if mv[1] * run > 0 else 0.0
            elif goal is not None:           # (walking to a spot: where the run will start)
                gx, gy = goal[0] - X0, goal[1] - Y0
                s -= 1.5 * max(0.0, abs(q[0] - gx) + abs(q[1] - gy) - 2)
            elif held:                       # (between runs: it stops dead, and stays put unless it has to move)
                s -= 10.0 if mv != (0, 0) else 0.0
            else:
                s -= .6 * max(0.0, abs(q[0] - cx) + abs(q[1] - cy) * 1.2 - 18)
            vx = np.nonzero(valid[j].any(0))[0]
            vy = np.nonzero(valid[j].any(1))[0]
            wall = min(q[0] - vx[0], vx[-1] - q[0], q[1] - vy[0], vy[-1] - q[1])
            if wall < 30:
                s -= (30 - wall) * .3
            # behave like a player holding keys: commit to a direction (or to
            # standing still) for a few frames before changing it
            if mv == last:
                s += 6.0 if hold < 5 else .5
            elif mv != (0, 0) and mv == (-last[0], -last[1]):
                s -= 4.0
            if mv == (0, 0):
                s += 2.0                     # standing still is the default
            if s > best_s:
                best, best_s, best_mv = q, s, mv
        hold = hold + 1 if best_mv == last else 0
        last = best_mv
        p = best
        if not safe[j][p[1], p[0]] and j > inv_until:
            cur_hp = max(1, cur_hp - DAMAGE)
            inv_until = j + int(INVULN * FPS)
            hits.append((f0 + j) / FPS)
    # scripted soul moves (no attacks there): the blast throws the soul ("fly",
    # easing out), then it walks like a player holding the arrow keys ("walk":
    # 8 directions at UT speed, or at the script's `speed`) and stays put until `hold`
    for sc in getattr(C, "SOUL_SCRIPT", []):
        a, e = int(round(sc["t0"] * FPS)) - f0, int(round(sc["hold"] * FPS)) - f0
        if not (0 <= a < n):
            continue
        to = np.array(sc["to"], float)
        if sc["kind"] == "fly":
            p0 = pos[a - 1 if a > 0 else a].astype(float)       # from where the previous move left it
            cube = getattr(C, "CUBE", None)
            if cube is not None and cube.tx <= sc["t0"] - 1 / FPS < cube.tb + 1e-6:
                p0 = np.asarray(cube.soul_view(pos[a - 1], sc["t0"] - 1 / FPS)[0], float)
            b = max(1, int(round(sc["dur"] * FPS)))
            for i in range(a, min(e, n)):
                u = min(1.0, (i - a) / b)
                if sc.get("ease") == "smooth":            # eased in and out (settling back, not thrown)
                    u = u * u * (3 - 2 * u)
                else:                                     # thrown: fast, then slowing down
                    u = 1 - (1 - u) ** 3
                pos[i] = np.round(p0 + (to - p0) * u)
                vis[i] = True
        else:
            q = pos[a - 1].astype(float)                # from where the previous move left it
            s = sc.get("speed", SPEED)
            for i in range(a, min(e, n)):
                pos[i] = np.round(q)
                vis[i] = True
                q = q + np.clip(to - q, -s, s)
    return pos, hp, vis, hits


DESCENT_PASSES = 4


def aim_from(pos, f0):
    def aim(t):
        k = int(round(t * FPS)) - f0
        if k < 0:
            return None
        return tuple(pos[min(k, len(pos) - 1)])
    return aim


def run(tl, label, only=None, base=None):
    """Plan every segment - or only segment `only`, the rest taken from `base`
    ((pos, hp, vis, hits) of a whole earlier run)."""
    t0 = time.time()
    print(f"[{label}] timeline: {len(tl.attacks)} attacks")
    out_pos, out_hp, out_vis, out_hits = [], [], [], []
    start, cur_hp = S.SCENE["soul_start"], S.HUD["hp"]
    resets = dict((int(round(t * FPS)), q) for t, q in getattr(C, "SOUL_RESETS", []))
    f_first = frame_range()[0]
    for si, (fa, fb, win) in enumerate(segments()):
        if fa in resets:
            start = resets[fa]
        if only is not None and si != only:
            bp, bh, bv, bhits = base
            sl = slice(fa - f_first, fb - f_first + 1)
            out_pos.append(bp[sl]), out_hp.append(bh[sl]), out_vis.append(bv[sl])
            out_hits.extend(h for h in bhits if fa / FPS - 1e-6 <= h <= fb / FPS + 1e-6)
            start, cur_hp = tuple(bp[sl][-1]), int(bh[sl][-1])
            continue
        set_window(win)
        safe, valid = safe_masks(tl, fa, fb)
        surv, broken = backward(safe, valid)
        if broken:
            spans = []
            for b in broken:
                tb = (fa + b) / FPS
                if not spans or tb - spans[-1][1] > .1:
                    spans.append([tb, tb])
                spans[-1][1] = tb
            print("unavoidable at:", ", ".join(f"{a:.2f}" for a, _ in spans))
        pos, hp, vis, hits = forward(safe, surv, valid, start, fa, cur_hp)
        del safe, surv, valid
        out_pos.append(pos), out_hp.append(hp), out_vis.append(vis), out_hits.extend(hits)
        start, cur_hp = tuple(pos[-1]), int(hp[-1])
    set_window(BAND)
    print(f"[{label}] hits: {[round(h, 2) for h in out_hits]}  ({time.time() - t0:.0f}s)")
    return np.concatenate(out_pos), np.concatenate(out_hp), np.concatenate(out_vis), out_hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "output" / "plan.npz"))
    args = ap.parse_args()
    C.LOCAL = True      # (the descent: plan in the moving arena's own frame - render moves it with the arena)
    f0, _ = frame_range()
    # pass 1: attacks aim at the arena centre; pass 2: they aim where the soul was
    C.AIM = None
    pos1 = run(C.build(), "pass 1")[0]
    C.AIM = aim_from(pos1, f0)
    res = run(C.build(), "pass 2")
    # the descent's attacks chase the soul closely (the climax's tide drives it
    # across the sheet from where it is): plan the descent again - only it - aiming
    # at where the last plan put the soul, till it holds still. Everything else
    # keeps pass 1's aim and pass 2's plan
    aim = pos1.copy()
    a = int(round(C.SINK0 * FPS))
    di = next(i for i, (fa, fb, _) in enumerate(segments()) if fa <= a <= fb)
    fa, fb, _ = segments()[di]
    sl = slice(fa - f0, fb - f0 + 1)
    for k in range(DESCENT_PASSES):
        aim[sl] = res[0][sl]
        C.AIM = aim_from(aim, f0)
        new = run(C.build(), f"descent {k + 1}", only=di, base=res)
        dif = np.abs(new[0][sl].astype(int) - res[0][sl].astype(int)).max(axis=1)
        moved = int(dif.max())
        res = new
        print(f"  (the soul moved up to {moved} px from the last plan, at {(fa + int(dif.argmax())) / FPS:.2f} s)")
        if moved <= 2:
            break
    np.save(ROOT / "output" / "aim.npy", aim)
    pos, hp, vis, hits = res
    np.savez_compressed(args.out, f0=f0, pos=pos, hp=hp, vis=vis, hits=np.array(hits))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
