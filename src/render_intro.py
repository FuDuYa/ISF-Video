"""Render a deterministic, music-synchronised monochrome opening with 2.5D camera motion.

Native pixel canvas: 640x480. Perspective projection uses nearest-neighbour
sampling. The final 4:3 image fills the height of a 1920x1080 frame, with
240-pixel black bars on each side. Source sprites keep their luminance.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageChops, ImageFilter

from render_preview import ROOT, SPRITES, gray_sprite, render_scene


W, H, PAD = 640, 480, 128
PLANE_SIZE = (W + 2 * PAD, H + 2 * PAD)


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def smooth(value: float) -> float:
    value = clamp(value)
    return value * value * (3 - 2 * value)


def out_cubic(value: float) -> float:
    return 1 - (1 - clamp(value)) ** 3


def scaled_luma(image: Image.Image, gain: float) -> Image.Image:
    return image.point([round(x * clamp(gain)) for x in range(256)])


def organic_arrival(texture: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    """Random-cost flood from one ink drop; every reached cell has a connected path.

    Broad and fine texture change the cost of neighbouring cells. Dijkstra's
    shortest-path growth creates branches and asymmetric pockets, without a
    radial or elliptical reveal mask. The field is fixed across frames, so its
    edge moves continuously instead of resampling noise and flickering.
    """
    height, width = texture.shape
    costs = .28 + 3.9 * (1 - texture) ** 2
    distances = np.full((height, width), np.inf, dtype=np.float32)
    sx, sy = seed
    distances[sy, sx] = 0
    queue: list[tuple[float, int, int]] = [(0.0, sx, sy)]
    steps = [(-1, 0, .66), (1, 0, .66), (0, -1, 1.05), (0, 1, 1.05),
             (-1, -1, 1.25), (1, -1, 1.25), (-1, 1, 1.25), (1, 1, 1.25)]
    while queue:
        distance, x, y = heapq.heappop(queue)
        if distance > distances[y, x] + 1e-5:
            continue
        for dx, dy, step in steps:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            candidate = distance + step * (float(costs[y, x]) + float(costs[ny, nx])) / 2
            if candidate < distances[ny, nx] - 1e-5:
                distances[ny, nx] = candidate
                heapq.heappush(queue, (candidate, nx, ny))
    return distances / float(distances.max())


class Camera:
    """Perspective camera looking toward the pixel-art plane, with a real z axis."""

    def __init__(self, keys: list[dict], time: float, hits: list[float]):
        a, b = keys[0], keys[0]
        for first, second in zip(keys, keys[1:]):
            a, b = first, second
            if time <= second["time"]:
                break
        u = smooth((time - a["time"]) / max(0.0001, b["time"] - a["time"]))
        mix = lambda key: a[key] + (b[key] - a[key]) * u
        self.focus = np.array(a["focus"], float) * (1 - u) + np.array(b["focus"], float) * u
        self.focal = 800.0
        self.distance = self.focal / mix("zoom")
        pitch, yaw, roll = [math.radians(mix(key)) for key in ("pitch", "yaw", "roll")]
        cp, sp, cy, sy, cr, sr = math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw), math.cos(roll), math.sin(roll)
        rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
        self.rotation = rz @ ry @ rx
        # A restrained, short recoil accents each impact without obscuring the UI.
        self.recoil = sum(2.0 * math.sin((time - hit) * 52) * math.exp(-(time - hit) * 24)
                          for hit in hits if 0 <= time - hit < 0.18)

    def project(self, x: float, y: float, z: float = 0) -> tuple[float, float, float]:
        point = self.rotation @ np.array([x - self.focus[0], y - self.focus[1], z])
        scale = self.focal / max(100, self.distance - point[2])
        return W / 2 + point[0] * scale, H / 2 + point[1] * scale + self.recoil, scale

    def warp(self, plane: Image.Image) -> Image.Image:
        # Solve inverse screen -> texture homography for Pillow's perspective sampler.
        corners = [(0, 0), (plane.width, 0), (plane.width, plane.height), (0, plane.height)]
        equations, values = [], []
        for sx, sy in corners:
            dx, dy, _ = self.project(sx - PAD, sy - PAD)
            equations.extend([[dx, dy, 1, 0, 0, 0, -sx * dx, -sx * dy],
                              [0, 0, 0, dx, dy, 1, -sy * dx, -sy * dy]])
            values.extend([sx, sy])
        coefficients = np.linalg.solve(np.array(equations), np.array(values))
        return plane.transform((W, H), Image.Transform.PERSPECTIVE, coefficients,
                               Image.Resampling.NEAREST, fillcolor=0)


class IntroRenderer:
    def __init__(self, scene: dict, config: dict):
        self.scene, self.config = scene, config
        timing = config["timing"]
        offset = timing["offset_seconds"]
        self.button_hits = [x + offset for x in timing["button_hits"]]
        self.edge_hits = [x + offset for x in timing["edge_hits"]]
        self.soul_hit = timing["soul_hit"] + offset
        if len(self.button_hits) != 4 or len(self.edge_hits) != 4:
            raise ValueError("The opening needs exactly four button hits and four edge hits.")
        self.hits = self.button_hits + self.edge_hits + [self.soul_hit]
        self.animation = config["animation"]
        self.final = render_scene(scene).convert("L")
        self.actor = self.final.crop((0, 0, W, scene["arena"]["y"]))
        self.hud = self.final.crop((0, 394, W, 427))
        self.soul = gray_sprite(SPRITES / scene["soul"]["sprite"])
        self.buttons = []
        self.random = np.random.default_rng(20261024)
        by_sprite = {button["sprite"]: button for button in scene["hud"]["buttons"]}
        order = config["button_order"]
        if len(order) != 4 or set(order) != set(by_sprite):
            raise ValueError("button_order must list each of the four scene buttons once.")
        self.button_order = order
        for button_name in order:
            button = by_sprite[button_name]
            source = gray_sprite(SPRITES / button["sprite"])
            # Ink and sprite occupy one small patch. A fixed texture controls
            # which ink pixels become exact source-sprite pixels at each instant.
            target = np.zeros((58, 126), dtype=np.uint8)
            target[8:50, 8:118] = np.array(source.convert("L"))
            fine = self.random.random((58, 126))
            coarse = Image.fromarray((self.random.random((8, 18)) * 255).astype("uint8"))
            coarse = np.asarray(coarse.resize((126, 58), Image.Resampling.BILINEAR)) / 255
            texture = .58 * coarse + .42 * fine
            rank = .08 + .84 * texture
            silhouette = Image.fromarray((target > 0).astype(np.uint8) * 255)
            silhouette = np.asarray(silhouette.filter(ImageFilter.MaxFilter(9))) > 0
            x, y = button["position"]
            self.buttons.append({"image": source, "target": target, "rank": rank,
                                 "texture": texture,
                                 "arrival": organic_arrival(texture, (63, 29)),
                                 "silhouette": silhouette,
                                 "center": (x + 55, y + 21), "position": (x, y)})

        arena = scene["arena"]
        self.edge_maps = []
        ew, eh, border = (arena[key] for key in ("width", "height", "border"))
        ex, ey = np.meshgrid(np.arange(ew + 16), np.arange(eh + 16))
        for i in range(4):
            if i == 0:  # top, left to right
                along, off, length = ex - 8, np.abs(ey - (8 + (border - 1) / 2)), ew
                target = (ex >= 8) & (ex < 8 + ew) & (ey >= 8) & (ey < 8 + border)
            elif i == 1:  # right, top to bottom
                along, off, length = ey - 8, np.abs(ex - (8 + ew - (border + 1) / 2)), eh
                target = (ex >= 8 + ew - border) & (ex < 8 + ew) & (ey >= 8) & (ey < 8 + eh)
            elif i == 2:  # bottom, right to left
                along, off, length = 8 + ew - 1 - ex, np.abs(ey - (8 + eh - (border + 1) / 2)), ew
                target = (ex >= 8) & (ex < 8 + ew) & (ey >= 8 + eh - border) & (ey < 8 + eh)
            else:  # left, bottom to top
                along, off, length = 8 + eh - 1 - ey, np.abs(ex - (8 + (border - 1) / 2)), eh
                target = (ex >= 8) & (ex < 8 + border) & (ey >= 8) & (ey < 8 + eh)
            fine = self.random.random((eh + 16, ew + 16))
            coarse = Image.fromarray((self.random.random((19, 19)) * 255).astype("uint8"))
            coarse = np.asarray(coarse.resize((ew + 16, eh + 16), Image.Resampling.BILINEAR)) / 255
            texture = .6 * coarse + .4 * fine
            rank = .10 + .80 * texture
            self.edge_maps.append({"along": along, "off": off, "length": length,
                                   "target": target, "texture": texture, "rank": rank})

    def draw_background_flash(self, frame: Image.Image, time: float) -> Image.Image:
        """A low-luma pulse behind the first eight impacts; no radiating lines."""
        duration = self.animation["flash_duration"]
        brightness = 0
        for index in self.animation["flash_hit_indices"]:
            age = time - self.hits[index]
            if 0 <= age < duration:
                brightness = max(brightness, round(self.animation["flash_luma"] *
                                                      (1 - smooth(age / duration))))
        if brightness == 0:
            return frame
        # Keep the flash in black screen space, behind the battle art.
        background = np.asarray(frame)
        return Image.fromarray(np.where(background == 0, brightness, background).astype(np.uint8))

    def draw_brush_sequence(self, plane: Image.Image, time: float) -> None:
        """An independent pixel brush paints a gray rectangle outside the arena."""
        cfg = self.config["brush"]
        if time < cfg["fade_start"]:
            return
        x0, y0, x1, y1 = cfg["rect"]
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
        draw = ImageDraw.Draw(plane)
        draw_start, side_duration = cfg["draw_start"], cfg["side_duration"]
        current = corners[0]
        direction = (1, 0)
        for side in range(4):
            start, end = corners[side], corners[side + 1]
            age = time - (draw_start + side * side_duration)
            if age < 0:
                break
            fraction = smooth(age / side_duration)
            tip = (round(start[0] + (end[0] - start[0]) * fraction),
                   round(start[1] + (end[1] - start[1]) * fraction))
            if tip != start:
                line = (start[0] + PAD, start[1] + PAD, tip[0] + PAD, tip[1] + PAD)
                draw.line(line, fill=cfg["stroke_luma"], width=3)
                draw.line(line, fill=min(255, cfg["stroke_luma"] + 24), width=1)
            current = tip
            direction = (int(math.copysign(1, end[0] - start[0])) if end[0] != start[0] else 0,
                         int(math.copysign(1, end[1] - start[1])) if end[1] != start[1] else 0)
            if age < side_duration:
                break

        gain = smooth((time - cfg["fade_start"]) / cfg["fade_duration"])
        gain *= 1 - smooth((time - cfg["fade_out_start"]) / cfg["fade_out_duration"])
        if gain <= 0:
            return
        # On the lower and left sides the handle tilts upward, clear of FDY's
        # HP row and command buttons while the tip stays on the paint path.
        ux, uy = direction
        if direction == (-1, 0):
            ux, uy = -math.sqrt(.5), math.sqrt(.5)
        elif direction == (0, -1):
            ux, uy = math.sqrt(.5), math.sqrt(.5)
        tx, ty = current
        def points(shape):
            return [(round(tx + ux * x - uy * y) + PAD,
                     round(ty + uy * x + ux * y) + PAD) for x, y in shape]
        def polygon(shape, shade):
            draw.polygon(points(shape), fill=round(shade * gain))
        # The brush is drawn from handle (negative local x) toward its ink tip.
        polygon([(-51, -3), (-47, -5), (-20, -5), (-18, -6), (-10, -6),
                 (-10, 6), (-18, 6), (-20, 5), (-47, 5), (-51, 3)], 49)
        polygon([(-49, -2), (-20, -3), (-20, 2), (-49, 2)], 101)
        polygon([(-46, -2), (-23, -2), (-23, -1), (-46, -1)], 147)
        polygon([(-20, -6), (-11, -6), (-11, 6), (-20, 6)], 154)
        polygon([(-18, -6), (-16, -6), (-16, 6), (-18, 6)], 216)
        polygon([(-13, -6), (-11, -6), (-11, 6), (-13, 6)], 205)
        polygon([(-11, -6), (-5, -7), (0, -6), (2, -4), (1, 3),
                 (-2, 5), (-6, 7), (-11, 6)], 71)
        polygon([(-10, -3), (-3, -4), (0, -3), (-3, -1), (-10, 0)], 109)
        polygon([(-9, 2), (-4, 1), (1, 2), (-3, 4), (-9, 4)], 98)
        polygon([(0, -5), (4, -5), (3, -3), (0, -3)], 64)
        polygon([(1, 0), (5, 0), (4, 2), (0, 2)], 58)

    def draw_button(self, plane: Image.Image, index: int, time: float) -> None:
        item = self.buttons[index]
        age = time - self.button_hits[index]
        if age < 0:
            return
        x, y = item["position"]
        if age >= self.animation["button_form"]:
            image = item["image"]
            plane.paste(image.convert("L"), (x + PAD, y + PAD), image.getchannel("A"))
            return
        spread = smooth(age / self.animation["button_spread"])
        # A connected random-cost flood replaces the old regular ellipse.
        # The same reached pixels then reverse-dissolve into exact sprite cells.
        wet = item["arrival"] <= .008 + .995 * spread
        wet &= item["silhouette"]
        morph = smooth((age - .02) / (self.animation["button_form"] - .12))
        resolved = morph >= item["rank"]
        values = np.zeros((58, 126), dtype=np.uint8)
        ink = wet & ~resolved & (item["texture"] > .12)
        values[ink] = (62 + item["texture"][ink] * 72).astype(np.uint8)
        formed = wet & resolved & (item["target"] > 0)
        values[formed] = item["target"][formed]
        patch = Image.fromarray(values)
        cx, cy = item["center"]
        px, py = cx - 63 + PAD, cy - 29 + PAD
        # Transparent negative space must not erase a previously formed neighbour.
        underneath = plane.crop((px, py, px + patch.width, py + patch.height))
        plane.paste(ImageChops.lighter(underneath, patch), (px, py))

    def draw_edges(self, plane: Image.Image, time: float) -> None:
        a = self.scene["arena"]
        x, y, w, h = [a[k] for k in ("x", "y", "width", "height")]
        for i, hit in enumerate(self.edge_hits):
            age = time - hit
            if age < 0:
                continue
            edge = self.edge_maps[i]
            front = edge["length"] * smooth(age / self.animation["edge_draw"])
            reached = (edge["along"] >= 0) & (edge["along"] < edge["length"])
            reached &= edge["along"] <= front + (edge["texture"] - .5) * 5
            wake = np.clip((front - edge["along"]) / 50, 0, 1)
            settling = smooth((age - self.animation["edge_draw"]) /
                               self.animation["edge_settle"])
            resolved = np.maximum(wake, settling) >= edge["rank"]
            values = np.zeros_like(edge["target"], dtype=np.uint8)
            ink = reached & ~resolved & (edge["off"] < 4 + edge["texture"] * 5)
            values[ink] = (66 + edge["texture"][ink] * 72).astype(np.uint8)
            values[reached & resolved & edge["target"]] = 255
            # Once the short trailing texture has settled, the original
            # five-pixel border remains exactly, with no residue outside it.
            if age >= self.animation["edge_draw"] + self.animation["edge_settle"]:
                values[:] = edge["target"].astype(np.uint8) * 255
            patch = Image.fromarray(values)
            px, py = x + PAD - 8, y + PAD - 8
            underneath = plane.crop((px, py, px + w + 16, py + h + 16))
            plane.paste(ImageChops.lighter(underneath, patch), (px, py))

    def draw_soul(self, plane: Image.Image, time: float) -> None:
        age = time - self.soul_hit
        gather = self.animation["soul_gather"]
        cx, cy = self.scene["soul"]["center"]
        if -gather <= age < 0:
            d = ImageDraw.Draw(plane)
            u = smooth((age + gather) / gather)
            for i in range(8):
                angle = i * math.tau / 8 + .35 * (1 - u)
                outer = 61 * (1 - u) + 4
                inner = outer + 14 * (1 - u)
                p1 = (round(cx + math.cos(angle) * outer + PAD), round(cy + math.sin(angle) * outer + PAD))
                p2 = (round(cx + math.cos(angle + .05) * inner + PAD), round(cy + math.sin(angle + .05) * inner + PAD))
                d.line((p1, p2), fill=round(65 + 44 * u), width=2)
        if age >= 0:
            size = 16 + round(6 * (1 - out_cubic(age / .18)))
            image = self.soul.resize((size, size), Image.Resampling.NEAREST)
            plane.paste(image.convert("L"), (cx - size // 2 + PAD, cy - size // 2 + PAD), image.getchannel("A"))

    def draw_falling_drops(self, frame: Image.Image, camera: Camera, time: float) -> None:
        d = ImageDraw.Draw(frame)
        duration = self.animation["drop_fall"]
        for i, hit in enumerate(self.button_hits):
            age = time - hit
            if not -duration <= age < 0:
                continue
            u = (age + duration) / duration
            cx, cy = self.buttons[i]["center"]
            # z is perpendicular to the UI plane, so the fall responds to the camera angle.
            z = 170 * (1 - u * u)
            px, py, scale = camera.project(cx + 10 * (1 - u), cy, z)
            radius = max(2, round(2.4 * scale))
            tail = round((7 + 7 * u) * scale)
            px, py = round(px), round(py)
            # Short, separated droplets in the wake, not a glow or a particle field.
            d.polygon([(px - radius, py), (px, py - tail), (px + radius, py)], fill=112)
            d.ellipse((px - radius, py - radius, px + radius, py + radius), fill=137)
            d.rectangle((px - 1, py - radius, px, py - radius + 2), fill=184)
            for n in range(2):
                by = py - tail - 6 - n * 7
                d.rectangle((px, by, px + 1, by + 2), fill=round(69 * u))

    def frame(self, time: float) -> Image.Image:
        plane = Image.new("L", PLANE_SIZE)
        for i in range(4):
            self.draw_button(plane, i, time)
        self.draw_edges(plane, time)
        self.draw_soul(plane, time)
        hud_gain = smooth((time - self.animation["hud_start"]) / self.animation["hud_fade"])
        actor_gain = smooth((time - self.animation["ink_start"]) / self.animation["ink_fade"])
        if hud_gain > 0:
            plane.paste(scaled_luma(self.hud, hud_gain), (PAD, PAD + 394))
        if actor_gain > 0:
            plane.paste(scaled_luma(self.actor, actor_gain), (PAD, PAD))
        self.draw_brush_sequence(plane, time)
        camera = Camera(self.config["camera_keys"], time, self.hits)
        frame = camera.warp(plane)
        frame = self.draw_background_flash(frame, time)
        self.draw_falling_drops(frame, camera, time)
        return frame


def fullscreen_frame(frame: Image.Image, export: dict) -> Image.Image:
    """Fit the native 4:3 battle image into centered 1080p without smoothing."""
    width, height = export["width"], export["height"]
    view_width, view_height = export["viewport_width"], export["viewport_height"]
    left, right = export["left_bar"], export["right_bar"]
    if (width, height) != (1920, 1080) or (view_width, view_height) != (1440, 1080):
        raise ValueError("The requested fullscreen layout is 1920x1080 with a 1440x1080 battle view.")
    if left != right or left + view_width + right != width or H * view_width != W * view_height:
        raise ValueError("The 4:3 battle view must be centered between equal black bars.")
    full = Image.new("L", (width, height))
    full.paste(frame.resize((view_width, view_height), Image.Resampling.NEAREST), (left, 0))
    return full


def render_video(renderer: IntroRenderer, destination: Path) -> None:
    if not shutil.which("ffmpeg"):
        raise SystemExit("FFmpeg is required. Install it, then run this command again.")
    cfg = renderer.config
    fps, duration = cfg["fps"], cfg["duration"]
    export = cfg["export"]
    width, height = export["width"], export["height"]
    fullscreen_frame(Image.new("L", (W, H)), export)  # validate dimensions before launching FFmpeg
    frames = round(duration * fps)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fade = cfg["audio_fade_out"]
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pixel_format", "gray", "-video_size", f"{width}x{height}",
               "-framerate", str(fps), "-i", "pipe:0", "-i", str(ROOT / cfg["music"]),
               "-map", "0:v:0", "-map", "1:a:0", "-t", str(duration),
               "-c:v", "libx264", "-preset", "medium", "-crf", "13", "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-b:a", "256k", "-af", f"afade=t=out:st={duration - fade}:d={fade}",
               "-movflags", "+faststart", str(destination)]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        try:
            for i in range(frames):
                full = fullscreen_frame(renderer.frame(i / fps), export)
                encoder.stdin.write(full.tobytes())
                if i % fps == 0:
                    print(f"Rendered {i / fps:.0f}s / {duration}s", flush=True)
        finally:
            encoder.stdin.close()
        code = encoder.wait()
        if code:
            raise RuntimeError(f"FFmpeg exited with status {code}.")
    print(f"Saved {destination}", flush=True)


def storyboard(renderer: IntroRenderer, destination: Path) -> None:
    times = [.99, 1.25, 1.84, 2.29, 2.73, 3.78, 4.55, 5.38, 6.50,
             7.22, 7.90, 8.80, 9.70, 10.50, 11.60]
    canvas = Image.new("RGB", (1280, math.ceil(len(times) / 3) * 267), (20, 20, 20))
    d = ImageDraw.Draw(canvas)
    for i, time in enumerate(times):
        x, y = i % 3 * 426, i // 3 * 267
        # Keep the same 4:3 composition in the reference thumbnails.
        image = renderer.frame(time).resize((320, 240), Image.Resampling.NEAREST)
        canvas.paste(image.convert("RGB"), (x + 53, y + 22))
        d.text((x + 53, y + 5), f"{time:.2f}s", fill=(200, 200, 200))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "intro.json")
    parser.add_argument("--scene", type=Path, default=ROOT / "scene.json")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "phase4-intro.mp4")
    parser.add_argument("--storyboard-only", action="store_true")
    parser.add_argument("--frame", type=float, help="Export one 1920x1080 PNG at this time.")
    args = parser.parse_args()
    scene = json.loads(args.scene.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    renderer = IntroRenderer(scene, config)
    if args.frame is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fullscreen_frame(renderer.frame(args.frame), config["export"]).save(args.output)
    elif args.storyboard_only:
        storyboard(renderer, ROOT / "output" / "intro-storyboard.png")
    else:
        render_video(renderer, args.output)


if __name__ == "__main__":
    main()
