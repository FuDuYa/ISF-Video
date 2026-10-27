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

X0, X1, Y0, Y1 = 90, 550, 226, 392      # crop window that contains every arena
T_START = 7.0
T_END = 146.5
HALF = 4                                 # 8x8 hitbox
# while the canvas is the whole picture (grid, tunnel, cube - the camera is close)
# the soul keeps its whole sprite clear of the ink: it never seems to touch black
# without being hurt
WIDE = (C.tm(20.75) + .25, C.tm(24.75))
HALF_WIDE = 12
SPEED = 2
DAMAGE = 4
INVULN = 1.0


def frame_range():
    return int(round(T_START * FPS)), int(round(T_END * FPS))


def valid_mask(box):
    x, y, w, h = box
    b = C.BORDER
    m = np.zeros((Y1 - Y0, X1 - X0), bool)
    xa, xb = x + b + 8, x + w - b - 8
    ya, yb = y + b + 8, y + h - b - 8
    m[ya - Y0:yb - Y0 + 1, xa - X0:xb - X0 + 1] = True
    return m


def safe_masks(tl):
    f0, f1 = frame_range()
    n = f1 - f0 + 1
    safe = np.zeros((n, Y1 - Y0, X1 - X0), bool)
    valid = np.zeros_like(safe)
    for i in range(n):
        t = (f0 + i) / FPS
        box = C.box_at(t)
        ctx = S.render_attacks(t, tl, box)
        haz = S.hazard_inside(ctx, box)[Y0:Y1, X0:X1]
        half = HALF_WIDE if WIDE[0] <= t < WIDE[1] else HALF
        unsafe = ndi.maximum_filter(haz, size=2 * half + 1) if haz.any() else haz
        v = valid_mask(box)
        valid[i] = v
        safe[i] = v & ~unsafe
        if i % 1200 == 0:
            print(f"  hazards {t:6.1f}s", flush=True)
    return safe, valid


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


def forward(safe, surv, valid, start):
    n = len(safe)
    f0, _ = frame_range()
    pos = np.zeros((n, 2), np.int32)
    hp = np.zeros(n, np.int32)
    vis = np.ones(n, bool)
    p = (start[0] - X0, start[1] - Y0)
    cur_hp = S.HUD["hp"]
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


def aim_from(pos, f0):
    def aim(t):
        k = int(round(t * FPS)) - f0
        if k < 0:
            return None
        return tuple(pos[min(k, len(pos) - 1)])
    return aim


def run(tl, label):
    t0 = time.time()
    print(f"[{label}] timeline: {len(tl.attacks)} attacks")
    safe, valid = safe_masks(tl)
    surv, broken = backward(safe, valid)
    f0, _ = frame_range()
    if broken:
        spans = []
        for b in broken:
            tb = (f0 + b) / FPS
            if not spans or tb - spans[-1][1] > .1:
                spans.append([tb, tb])
            spans[-1][1] = tb
        print("unavoidable at:", ", ".join(f"{a:.2f}" for a, _ in spans))
    pos, hp, vis, hits = forward(safe, surv, valid, S.SCENE["soul_start"])
    print(f"[{label}] hits: {[round(h, 2) for h in hits]}  ({time.time() - t0:.0f}s)")
    return pos, hp, vis, hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "output" / "plan.npz"))
    args = ap.parse_args()
    C.LOCAL = True      # (the descent: plan in the moving arena's own frame - render moves it with the arena)
    f0, _ = frame_range()
    # pass 1: attacks aim at the arena centre; pass 2: they aim where the soul was
    C.AIM = None
    pos, hp, vis, hits = run(C.build(), "pass 1")
    np.save(ROOT / "output" / "aim.npy", pos)
    C.AIM = aim_from(pos, f0)
    pos, hp, vis, hits = run(C.build(), "pass 2")
    np.savez_compressed(args.out, f0=f0, pos=pos, hp=hp, vis=vis, hits=np.array(hits))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
