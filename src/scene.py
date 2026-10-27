"""Compose one 640x480 battle frame: Ink Sans, arena, attacks, soul, HUD."""

from __future__ import annotations

import bisect
import json
import math

import numpy as np

import actions
import attacks as A
import choreo as C
from PIL import Image, ImageDraw

from common import ROOT, VIRTUAL, W, H, BitmapFont, Sprite, clamp, hp_label, paste, smooth, sprite, out_cubic

SCENE = json.loads((ROOT / "scene.json").read_text(encoding="utf-8"))
INK = SCENE["ink"]
HUD = SCENE["hud"]
FONT = BitmapFont()
HP_TAG = hp_label()
SWAY_START = 7.75
_prng = np.random.default_rng(9)
PAPER = (np.kron(_prng.integers(0, 4, (H // 2, W // 2)), np.ones((2, 2), np.int16)) * 4
         + np.kron(_prng.integers(0, 2, (H // 8, W // 8)), np.ones((8, 8), np.int16)) * 6).astype(np.int16)
SANS_HOOK = None          # set by render.py: intro ink formation of Ink Sans
BACKDROP = None           # set by render.py: background mood layer (backdrop.Backdrop)
DIALOGUE = None           # set by render.py: Ink Sans speech bubble (dialogue.Dialogue)
SOAK = None               # set by render.py: the arena soaking into a canvas (soak.Soak)
CURTAIN = None            # set by render.py: the curtain of ink Ink Sans hides behind (curtain.Curtain)
BRUSH_RETURNS = C.tm(12.75) - .45  # the brush ink-forms on his back, done on the break star


def _make_nobrush() -> Sprite:
    """body.png with the brush on his back cut away (source PNG untouched)."""
    src = sprite("body.png")
    cut = Image.new("L", (src.w, src.h))
    d = ImageDraw.Draw(cut)
    for poly in INK["nobrush_cutouts"]:
        d.polygon([tuple(p) for p in poly], fill=255)
    return Sprite(src.v.copy(), src.a & ~(np.asarray(cut) > 0))


VIRTUAL["body_nobrush.png"] = _make_nobrush()

# ---------------------------------------------------------------- the brush returns
# At BRUSH_RETURNS the brush on his back is painted back in (same ink formation
# as his entrance), just before he leaps at the arena.
_BRUSH_FORM = None


def brush_form():
    global _BRUSH_FORM
    if _BRUSH_FORM is None:
        from intro import InkForm
        full, bare = sprite("body.png"), VIRTUAL["body_nobrush.png"]
        only = Sprite(full.v.copy(), full.a & ~bare.a)
        ox, oy = INK["origin"]
        px, py = INK["body"]
        rng = np.random.default_rng(21)
        _BRUSH_FORM = InkForm(only, ox + px, oy + py, [(128, 30, BRUSH_RETURNS), (20, 132, BRUSH_RETURNS + .06)],
                              .32, rng, "brush_return", morph=.22, pad=6)
    return _BRUSH_FORM


_BRUSH_FLASH = None


def brush_flash() -> Sprite:
    """The brush on his back as a white silhouette (its blink)."""
    global _BRUSH_FLASH
    if _BRUSH_FLASH is None:
        full, bare = sprite("body.png"), VIRTUAL["body_nobrush.png"]
        _BRUSH_FLASH = Sprite(np.full_like(full.v, 255), full.a & ~bare.a)
    return _BRUSH_FLASH


# ---------------------------------------------------------------- Ink Sans
def sans_pose(t: float, tl: "C.Timeline | None"):
    """Head and body for time t. A section may fix the body (`body` key); it is
    then held for the whole section (big moves come from actions.py). Elsewhere
    the brush stays out between swings that are close together, so the sprite
    never flickers back and forth."""
    sec = C.section_at(t)
    head = sec["head"]
    body = "body.png" if t >= BRUSH_RETURNS else "body_nobrush.png"
    if "body" in sec:
        body = sec["body"]
    elif tl is not None and tl.swings:
        i = bisect.bisect_right(tl.swings, t + .05) - 1
        if i >= 0:
            s = tl.swings[i]
            prev = tl.swings[i - 1] if i > 0 else -9.0
            nxt = tl.swings[i + 1] if i + 1 < len(tl.swings) else 1e9
            tau = t - s
            if -.05 <= tau < .07 and s - prev > 1.5:
                body = "body_anime1.png"
            elif tau < .6 or (nxt - s < 1.5 and t < nxt):
                body = "body_anime2.png"
    if t >= C.tm(119.5):
        body = "body.png"
    return head, body


def sans_offsets(t: float):
    """Slow vertical breathing: legs stay planted, the torso rises and falls,
    the skull follows a little later. Offsets stay on the sprites' 2-px grain."""
    if t < SWAY_START:
        return 0, 0, 0
    ph = (t - SWAY_START) / INK.get("bob_period", 2.4) * math.tau
    by = 2 * round(.5 - .5 * math.cos(ph))          # 0 or 2 px
    hy = 2 * round(.5 - .5 * math.cos(ph - .9))     # skull follows slightly later
    return 0, by, hy


def draw_sans(frame: np.ndarray, t: float, tl, gain: float = 1.0, head=None, body=None):
    if C.INK_GONE[0] <= t < C.INK_GONE[1]:
        return                                # (he steps out while the camera is too close to see him)
    h, b = sans_pose(t, tl)
    head, body = head or h, body or b
    ox, oy = INK["origin"]
    oy -= int(round(C.rise(t)))               # (the descent: he rises away as we sink)
    if oy < -260:
        return
    _, by, hy = sans_offsets(t)
    lyy = 0
    act = actions.pose(t)
    if act is not None:                       # a jump-slam drives the whole figure
        ox, oy = ox + act["dx"], oy + act["dy"]
        body, by, hy, lyy = act["body"], act["by"], act["hy"], act["ly"]
    lx, ly = INK["legs"]
    paste(frame, sprite("legs.png"), ox + lx, oy + ly + lyy, gain)
    px, py = INK["body"]
    if act is not None and abs(act.get("rot", 0)) > .5:      # a bending pose: body + skull turn at the hip
        spr, x0, y0 = bent_upper(body, head, act["rot"], by, hy)
        paste(frame, spr, ox + x0, oy + y0, gain)
        return
    form = brush_form() if BRUSH_RETURNS <= t < BRUSH_RETURNS + 1.0 and act is None else None
    if form is not None and t < form.done and body == "body.png":
        paste(frame, sprite("body_nobrush.png"), ox + px, oy + py + by, gain)
        form.draw(frame, t, ox - INK["origin"][0], oy - INK["origin"][1] + by)
    else:
        bs = actions.brush_state(t) if body == "body.png" else None
        paste(frame, sprite("body_nobrush.png" if bs == "gone" else body), ox + px, oy + py + by, gain)
        if bs == "flash":                     # the brush blinks off / back onto his back
            paste(frame, brush_flash(), ox + px, oy + py + by, gain)
    qx, qy = INK["head"]
    paste(frame, sprite(head), ox + qx, oy + qy + hy, gain)


_BENT = {}


def bent_upper(body, head, rot, by, hy):
    """Body + skull as one sprite turned about the hip (nearest neighbour), and its
    offset from Ink Sans' origin. Cached per pose."""
    rq = int(round(rot / 2)) * 2
    key = (body, head, rq, by, hy)
    if key not in _BENT:
        R = 230
        canvas_v = np.zeros((2 * R, 2 * R), np.uint8)
        canvas_a = np.zeros((2 * R, 2 * R), bool)
        px, py = INK["body"]
        qx, qy = INK["head"]
        for name, x, y in ((body, px, py + by), (head, qx, qy + hy)):
            s = sprite(name)
            region_v = canvas_v[R + y:R + y + s.h, R + x:R + x + s.w]
            region_a = canvas_a[R + y:R + y + s.h, R + x:R + x + s.w]
            region_v[s.a] = s.v[s.a]
            region_a |= s.a
        hip = (R, R - 58)
        iv = Image.fromarray(canvas_v).rotate(rq, resample=Image.Resampling.NEAREST, center=hip)
        ia = Image.fromarray(canvas_a.astype(np.uint8) * 255).rotate(rq, resample=Image.Resampling.NEAREST, center=hip)
        v, a = np.asarray(iv), np.asarray(ia) > 127
        ys, xs = np.nonzero(a)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        _BENT[key] = (Sprite(v[y0:y1, x0:x1].copy(), a[y0:y1, x0:x1].copy()), int(x0 - R), int(y0 - R))
    return _BENT[key]


def sans_composite(head="head_normal.png", body="body.png") -> Sprite:
    """Static Ink Sans as one sprite (used by the intro ink formation)."""
    canvas = np.zeros((H, W), np.uint8)
    alpha = np.zeros((H, W), bool)
    ox, oy = INK["origin"]
    for name, key in (("legs.png", "legs"), (body, "body"), (head, "head")):
        s = sprite(name)
        x, y = ox + INK[key][0], oy + INK[key][1]
        paste(canvas, s, x, y)
        alpha[y:y + s.h, x:x + s.w] |= s.a
    ys, xs = np.nonzero(alpha)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    spr = Sprite(canvas[y0:y1, x0:x1].copy(), alpha[y0:y1, x0:x1].copy())
    spr.x, spr.y = int(x0), int(y0)
    return spr


# ---------------------------------------------------------------- HUD
def draw_hud(frame: np.ndarray, hp: int, gain: float = 1.0, buttons: bool = True, selected=None):
    FONT.draw(frame, HUD["name"], *HUD["name_position"], HUD["font_tracking"], gain)
    FONT.draw(frame, f"LV {HUD['lv']}", *HUD["lv_position"], HUD["font_tracking"], gain)
    paste(frame, HP_TAG, *HUD["hp_label_position"], gain)
    bar = HUD["hp_bar"]
    bx, by, bw, bh = bar["x"], bar["y"], bar["width"], bar["height"]
    g = lambda v: int(v * gain)
    frame[by:by + bh, bx:bx + bw] = g(bar["empty"])
    fill = round(bw * hp / HUD["max_hp"])
    if fill > 0:
        frame[by:by + bh, bx:bx + fill] = g(bar["fill"])
    FONT.draw(frame, f"{hp:02} / {HUD['max_hp']:02}", bx + bw + 14, HUD["hp_value_y"],
              HUD["font_tracking"], gain)
    if buttons:
        for i, btn in enumerate(HUD["buttons"]):
            name = btn["sprite"].replace("1.png", "2.png") if i == selected else btn["sprite"]
            paste(frame, sprite(name), *btn["position"], gain)


# ---------------------------------------------------------------- arena
def draw_box(frame: np.ndarray, box, fill_inside=True):
    x, y, w, h = box
    b = C.BORDER
    if fill_inside:
        frame[y + b:y + h - b, x + b:x + w - b] = 0
    frame[y:y + b, x:x + w] = 255
    frame[y + h - b:y + h, x:x + w] = 255
    frame[y:y + h, x:x + b] = 255
    frame[y:y + h, x + w - b:x + w] = 255


def up2(img) -> np.ndarray:
    a = np.asarray(img)
    return a.repeat(2, 0).repeat(2, 1)


FINE = (C.tm(21) - .05, C.tm(23) + .05)    # deep zoom into the canvas: draw at full resolution


def render_attacks(t: float, tl, box):
    ctx = A.Ctx(t, box, fine=FINE[0] <= t < FINE[1])
    for a in sorted(tl.active(t), key=lambda a: getattr(a, "z", 0)):    # stable: start order within a layer
        a.draw(ctx)
    return ctx


def hazard_inside(ctx, box) -> np.ndarray:
    """Full-resolution hazard mask restricted to the arena interior."""
    haz = ctx.layer("haz") > 0
    x, y, w, h = box
    b = C.BORDER
    out = np.zeros_like(haz)
    out[y + b:y + h - b, x + b:x + w - b] = haz[y + b:y + h - b, x + b:x + w - b]
    return out


# ---------------------------------------------------------------- the canvas
# From CANVAS_SET on the arena is a sheet of paper with no drawn border. While
# we fly through the tunnel (M21-M22.75) the paper spreads out over the screen;
# after the cube bursts it draws back to the arena.
SPREAD = (C.tm(21), C.tm(22.75))
HUD_TOP_ROW = 394
RETRACT = (C.tm(24.75) + .1, C.tm(24.75) + .8)
_YY, _XX = np.mgrid[0:H, 0:W]
_erng = np.random.default_rng(31)
_EDGE = np.kron(_erng.uniform(-1, 1, (H // 16 + 1, W // 16 + 1)), np.ones((16, 16)))[:H, :W]
_EDGE = (_EDGE + np.kron(_erng.uniform(-1, 1, (H // 4, W // 4)), np.ones((4, 4))) * .45) / 1.45


def paper_spread(t: float) -> float:
    # it creeps out of the arena slowly and only gathers pace near the end (the
    # camera pushing in at the same time makes any spread look quicker)
    u = clamp((t - SPREAD[0]) / (SPREAD[1] - SPREAD[0]))
    return u ** 2.2 * (1 - smooth((t - RETRACT[0]) / (RETRACT[1] - RETRACT[0])))


def paper_mask(t: float, box, grid=None):
    """(paper, wet rim) boolean masks for the canvas at time t. The paper spreads
    over the whole screen for the tunnel - the buttons too, they are part of the
    canvas from then on; after the cube it draws back from the top only, to a
    sheet that still wraps the buttons (C.REST_RECT). `grid`: (XX, YY, EDGE) for
    a bigger plane than the screen (scene.overscan)."""
    XX, YY, EDGE = grid if grid is not None else (_XX, _YY, _EDGE)
    if C.SINK0 <= t < C.NEW_END:
        return sink_paper(t, box, XX, YY, EDGE)
    x, y, w, h = box
    big = (-150, -150, W + 150, H + 150)
    if t < RETRACT[0]:
        s = paper_spread(t)
        if s <= 0:
            m = (XX >= x) & (XX < x + w) & (YY >= y) & (YY < y + h)
            push = edge_push(t)
            if push > .5:                                   # the left edge bowed out by a wall
                prof = np.sin(np.pi * np.clip((YY - y) / h, 0, 1)) ** .8
                m |= (XX < x) & (XX >= x - push * prof) & (YY >= y) & (YY < y + h)
            return m, None
        r0, r1, moving = (x, y, x + w, y + h), big, s
    else:
        s = smooth((t - RETRACT[0]) / (RETRACT[1] - RETRACT[0]))
        r0, r1, moving = big, C.REST_RECT, s
    x0, y0, x1, y1 = (a + (b - a) * moving for a, b in zip(r0, r1))
    d = np.maximum(np.maximum(x0 - XX, XX - x1), np.maximum(y0 - YY, YY - y1))
    ragged = d - EDGE * 18 * min(1.0, 4 * moving * (1 - moving) + .15)
    return ragged < 0, (ragged > -5) & (ragged < 0)


def _grain_shift(t):
    dx, dy = C.paper_offset(t)
    return int(round(dy)) % H, int(round(dx)) % W


def paper_grain(t):
    """The paper's grain, carried along with the sheet in the run and the climb
    (C.paper_offset)."""
    sy, sx = _grain_shift(t)
    return PAPER if sx == 0 and sy == 0 else np.roll(PAPER, (sy, sx), axis=(0, 1))


def edge_push(t):
    """How far (px) the paper's left edge bulges out: a wall drunk there kicks it
    out, it springs back with a little wobble."""
    amp = 0.0
    for tp, a0 in C.EDGE_PUSH:
        a = t - tp
        if 0 <= a < .6:
            amp += a0 * (math.sin(a / .05 * math.pi / 2) if a < .05
                         else math.exp(-(a - .05) * 9) * math.cos((a - .05) * 26))
    return max(0.0, amp)


def sink_paper(t, box, XX, YY, EDGE):
    """The descent: the torn-out piece (the arena, moving) and - until it has risen
    away - the rest of the sheet with a hole where the piece was. The tear runs
    round the piece one side a note on the 5-4-5-4 sixteenths (a dark wet line)."""
    x, y, w, h = box
    d = np.maximum(np.maximum(x - XX, XX - (x + w)), np.maximum(y - YY, YY - (y + h)))
    piece = d - EDGE * 4 < 0
    edge = (d - EDGE * 4 > -2.5) & piece                  # its torn edge, wet and dark (seen over the sheet too)
    up = C.rise(t)
    if up > H + 300:
        return piece, (edge if t < C.ARRIVE + .4 else None)
    x0, y0, x1, y1 = C.REST_RECT
    sheet = (XX >= x0) & (XX < x1) & (YY >= y0 - up) & (YY < y1 - up)
    hx, hy, hw, hh = C.arena_box(C.SINK0)                 # where the piece was cut out of the sheet
    hy -= up
    dh = np.maximum(np.maximum(hx - XX, XX - (hx + hw)), np.maximum(hy - YY, YY - (hy + hh)))
    hole = dh - EDGE * 4 < 1
    s16 = C.MEAS / 16
    k = (t - C.SINK0) / s16                               # how much of the tear is done (sides)
    rim = None
    if t < C.RISE[0]:
        band = (dh - EDGE * 4 > -2) & (dh - EDGE * 4 < 2)
        sides = [YY < hy + hh * .25, XX > hx + hw * .75, YY > hy + hh * .75, XX < hx + hw * .25]
        cut = np.zeros_like(band)
        for j in range(4):
            if k >= j:
                cut |= sides[j]
        rim = band & cut
        return piece | sheet, rim                         # (still in one piece: only the tear shows)
    return piece | (sheet & ~hole), edge


def compose_canvas(frame, t, tl, box, ctx, under_top=None):
    pm, rim = paper_mask(t, box)
    compose_canvas.pm = pm
    g = paper_grain(t)
    paper = np.clip(255 - g, 0, 255).astype(np.uint8)
    frame[pm] = paper[pm]
    if rim is not None:
        frame[rim] = np.minimum(frame[rim], 186)               # the wet edge as it spreads
    art = ctx.layer("art")
    am = (art > 0) & pm
    frame[am] = np.clip(255 - art[am].astype(np.int16) - g[am], 0, 255).astype(np.uint8)
    if CURTAIN is not None and CURTAIN.active(t):
        # he is hiding behind the curtain of ink over the paper: drawn only where
        # there is no paper (behind its edge), then the curtain over him
        tmp = frame.copy()
        draw_sans(tmp, t, tl)
        frame[~pm] = tmp[~pm]
        CURTAIN.draw(frame, t, pm)
    elif not (SANS_HOOK and SANS_HOOK(frame, t)):
        draw_sans(frame, t, tl)                                 # he stands in front of the canvas
    compose_canvas.hud_done = bool(under_top and under_top(frame, pm))
    top = ctx.layer("top").copy()
    tmask = top > 0
    inv = tmask & pm                                            # over the paper, white ink reads dark
    top[inv] = 255 - top[inv]
    frame[tmask] = top[tmask]
    if ctx.sprites:                                             # sprites turn to ink on the paper
        sv = np.zeros_like(frame)
        sa = np.zeros(frame.shape, bool)
        for spr, sx, sy, gain, mask in ctx.sprites:
            paste(sv, spr, sx, sy, gain, mask)
            paste_alpha(sa, spr, sx, sy, mask)
        si = sa & pm
        sv[si] = 255 - sv[si]
        frame[sa] = sv[sa]
    return frame


def compose(t: float, tl, soul_xy, hp: int, soul_visible: bool = True) -> np.ndarray:
    """One finished 640x480 frame (before the final 1080p enlargement)."""
    frame = np.full((H, W), tl.flash(t), np.uint8)
    if BACKDROP is not None:
        BACKDROP.draw(frame, t)
    box = C.box_at(t)
    ctx = render_attacks(t, tl, box)
    c0, c1 = C.CANVAS()
    on_canvas = c0 <= t < c1
    sel = 0 if C.FIGHT_SELECT[0] <= t < C.FIGHT_SELECT[1] else None

    def paper_hud(fr, pm):
        """When the canvas wraps the buttons the HUD is painted on the paper, under
        whatever is drawn over the canvas."""
        sinking = C.SINK0 <= t < C.NEW_END
        if not sinking and not pm[HUD_TOP_ROW:].any():
            return False
        hud = np.zeros_like(fr)
        draw_hud(hud, hp, selected=sel)
        if sinking:                                       # it rises away with the rest of the sheet
            up = int(round(C.rise(t)))
            if up >= H:
                hud[:] = 0
            elif up:
                hud = np.roll(hud, -up, axis=0)
                hud[H - up:] = 0
        hm = hud > 0
        on = hm & pm
        fr[hm & ~pm] = hud[hm & ~pm]
        fr[on] = np.clip(255 - hud[on].astype(np.int16) * 2.2, 0, 255).astype(np.uint8)
        return True

    if on_canvas:
        frame = compose_canvas(frame, t, tl, box, ctx, under_top=paper_hud)
    else:
        if not (SANS_HOOK and SANS_HOOK(frame, t)):
            draw_sans(frame, t, tl)
        draw_box(frame, box)
        if SOAK is not None:
            SOAK.border(frame, t, box)
        x, y, w, h = box
        b = C.BORDER
        art = ctx.layer("art")
        inner = (slice(y + b, y + h - b), slice(x + b, x + w - b))
        region = frame[inner]
        a = art[inner]
        region[a > 0] = a[a > 0]
        if SOAK is not None and SOAK.p0 <= t < c0:
            SOAK.paper(region, t, box)
        top = ctx.layer("top")
        frame[top > 0] = top[top > 0]
        for spr, sx, sy, gain, mask in ctx.sprites:
            paste(frame, spr, sx, sy, gain, mask)
    if soul_visible and soul_xy is not None:
        sx, sy = soul_xy
        sg = SOAK.soul_gain(t, soul_xy, box) if SOAK is not None else 1.0
        cube = C.CUBE
        if cube is not None and cube.tx <= t < cube.tb:
            # the soul rides the turning box as a solid heart on its front face,
            # in the box's own perspective
            for spr, lx, ly in cube.soul_layers(soul_xy, t):
                paste(frame, spr, lx, ly, sg)
        else:
            paste(frame, sprite("player.png"), int(round(sx)) - 8, int(round(sy)) - 8, sg)
    pm = compose_canvas.pm if on_canvas else None
    if not (on_canvas and compose_canvas.hud_done):
        draw_hud(frame, hp, selected=sel)
    press_flash(frame, t, pm)
    if DIALOGUE is not None:
        DIALOGUE.draw(frame, t)
    dx, dy = tl.shake(t)
    global LAST_PLANE
    LAST_PLANE = None
    if on_canvas and t >= SPREAD[0]:
        # once the paper has spread, the sheet runs on past the screen (and the
        # living ink may circle out there): give the camera that bigger picture
        plane = overscan(frame, t, tl, box)
        if dx or dy:
            plane = shift(plane, dx, dy)
        LAST_PLANE = (t, plane)
        return plane[fx_pad():fx_pad() + H, fx_pad():fx_pad() + W].copy()
    if dx or dy:
        frame = shift(frame, dx, dy)
    return frame


LAST_PLANE = None          # (t, padded plane) when the frame has content past its edges


def fx_pad():
    import fx
    return fx.PAD


class PlaneCtx:
    """A Ctx stand-in drawing at full resolution into a padded plane ("top" and
    sprites only) - for what lies past the screen's edges."""

    def __init__(self, t, pad, size):
        self.t, self.P, self.k = t, pad, 1
        self.top = Image.new("L", size)
        self.d = ImageDraw.Draw(self.top)
        self.sprites = []

    def _ok(self, layer):
        return layer == "top"

    def circle(self, layer, x, y, size, fill):
        if self._ok(layer):
            r = max(size, 1.0) / 2
            x, y = x + self.P, y + self.P
            self.d.ellipse((x - r, y - r, x + r - .01, y + r - .01), fill=fill)

    def line(self, layer, pts, width, fill):
        if self._ok(layer):
            self.d.line([(px + self.P, py + self.P) for px, py in pts], fill=fill, width=max(1, int(round(width))))

    def poly(self, layer, pts, fill):
        if self._ok(layer):
            self.d.polygon([(px + self.P, py + self.P) for px, py in pts], fill=fill)

    def rect(self, layer, x0, y0, x1, y1, fill):
        if self._ok(layer):
            self.d.rectangle((x0 + self.P, y0 + self.P, x1 + self.P - .01, y1 + self.P - .01), fill=fill)

    def ring(self, layer, x, y, r, width, fill):
        if self._ok(layer):
            x, y = x + self.P, y + self.P
            self.d.ellipse((x - r, y - r, x + r, y + r), outline=fill, width=max(1, int(round(width))))


_PLANE_TEX = None


def overscan(frame, t, tl, box):
    """The composed frame inside a padded plane: the paper continues into the
    padding (same sheet, same edge), with the living ink drawn on it."""
    global _PLANE_TEX
    P = fx_pad()
    Hp, Wp = H + 2 * P, W + 2 * P
    off = _grain_shift(t)                                    # same grain as on screen (carried along)
    if _PLANE_TEX is None or _PLANE_TEX[2] != off:
        tex = np.clip(255 - np.pad(paper_grain(t), P, mode="wrap"), 0, 255).astype(np.uint8)
        yy, xx = np.mgrid[-P:H + P, -P:W + P]
        _PLANE_TEX = (tex, (xx, yy, np.pad(_EDGE, P, mode="wrap")), off)
    tex, grid, _ = _PLANE_TEX
    paper, rim = paper_mask(t, box, grid)
    plane = np.full((Hp, Wp), tl.flash(t), np.uint8)
    plane[paper] = tex[paper]
    if rim is not None:
        plane[rim] = np.minimum(plane[rim], 186)
    pc = PlaneCtx(t, P, (Wp, Hp))
    for a in tl.active(t):
        if getattr(a, "overscan", False):
            a.draw(pc)
    top = np.asarray(pc.top).copy()
    tmask = top > 0
    inv = tmask & paper
    top[inv] = 255 - top[inv]
    plane[tmask] = top[tmask]
    if pc.sprites:
        sv = np.zeros_like(plane)
        sa = np.zeros(plane.shape, bool)
        for spr, sx, sy, gain, mask in pc.sprites:
            paste(sv, spr, sx + P, sy + P, gain, mask)
            paste_alpha(sa, spr, sx + P, sy + P, mask)
        si = sa & paper
        sv[si] = 255 - sv[si]
        plane[sa] = sv[sa]
    plane[P:P + H, P:P + W] = frame
    return plane


def press_flash(frame, t, pm):
    """Each press of FIGHT (C.PRESSES): the button flashes - a few frames inverted,
    then a frame of ink spreading out from it and fading."""
    x, y = HUD["buttons"][0]["position"]
    w, h = sprite(HUD["buttons"][0]["sprite"]).w, sprite(HUD["buttons"][0]["sprite"]).h
    for tp in C.PRESSES:
        a = t - tp
        if not (0 <= a < .26):
            continue
        if a < .05:
            frame[y:y + h, x:x + w] = 255 - frame[y:y + h, x:x + w]
        u = out_cubic(min(1.0, a / .26))
        g = int(4 + 18 * u)
        on_paper = pm is not None and pm[y + h // 2, x + w // 2]
        v = int(255 - 210 * (1 - u)) if on_paper else int(230 * (1 - u))
        x0, y0, x1, y1 = max(0, x - g), max(0, y - g), min(W, x + w + g), min(H, y + h + g)
        for sl in ((slice(y0, y0 + 2), slice(x0, x1)), (slice(y1 - 2, y1), slice(x0, x1)),
                   (slice(y0, y1), slice(x0, x0 + 2)), (slice(y0, y1), slice(x1 - 2, x1))):
            frame[sl] = np.minimum(frame[sl], v) if on_paper else np.maximum(frame[sl], v)


_SOULS = {}


def soul_sprite(k: float) -> Sprite:
    """player.png scaled by perspective (nearest neighbour, even sizes)."""
    n = max(6, int(round(16 * k / 2)) * 2)
    if n not in _SOULS:
        src = sprite("player.png")
        idx = (np.arange(n) * 16 / n).astype(int)
        _SOULS[n] = Sprite(src.v[idx][:, idx].copy(), src.a[idx][:, idx].copy())
    return _SOULS[n]


def paste_alpha(dst: np.ndarray, spr, x: int, y: int, mask=None):
    """OR a sprite's alpha into a boolean layer (same clipping as common.paste)."""
    x, y = int(round(x)), int(round(y))
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(dst.shape[1], x + spr.w), min(dst.shape[0], y + spr.h)
    if x0 >= x1 or y0 >= y1:
        return
    sa = spr.a[y0 - y:y1 - y, x0 - x:x1 - x]
    if mask is not None:
        sa = sa & mask[y0 - y:y1 - y, x0 - x:x1 - x]
    dst[y0:y1, x0:x1] |= sa


def shift(frame, dx, dy):
    out = np.zeros_like(frame)
    hh, ww = frame.shape
    xs0, xs1 = max(0, -dx), min(ww, ww - dx)
    ys0, ys1 = max(0, -dy), min(hh, hh - dy)
    out[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx] = frame[ys0:ys1, xs0:xs1]
    return out
