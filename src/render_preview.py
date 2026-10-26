"""Render the phase-four idle frame with original sprites and bitmap UI text.

All composition happens at 640 x 480, on integer pixel coordinates. Output
scaling uses nearest-neighbour sampling. No system fonts or antialiasing enter
the battle frame. The supplied sprite files are read without modification.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageOps, ImageChops


ROOT = Path(__file__).resolve().parents[1]
SPRITES = ROOT / "sprite"
FONTS = ROOT / "assets" / "fonts"


def gray_sprite(path: Path) -> Image.Image:
    """Preserve the source alpha and pixel boundaries; remove chroma only."""
    with Image.open(path) as source:
        rgba = source.convert("RGBA")
    value = ImageOps.grayscale(rgba)
    return Image.merge("RGBA", (value, value, value, rgba.getchannel("A")))


def actor_layer(layer: dict) -> Image.Image:
    """Clear configured prop and matte cutouts without changing the source PNG.

    The supplied torso has an opaque black patch inside its collar opening.
    It is a background matte, not clothing. Making that limited patch transparent
    allows the head to sit behind the actual collar without losing its mouth.
    The source PNG is never changed.
    """
    image = gray_sprite(SPRITES / layer["sprite"])
    for polygon in layer.get("alpha_cutouts", []):
        cutout = Image.new("L", image.size)
        ImageDraw.Draw(cutout).polygon([tuple(point) for point in polygon], fill=255)
        image.putalpha(ImageChops.subtract(image.getchannel("A"), cutout))
    for polygon in layer.get("black_matte_cutouts", []):
        region = Image.new("L", image.size)
        ImageDraw.Draw(region).polygon([tuple(point) for point in polygon], fill=255)
        black_pixels = image.getchannel("R").point(lambda value: 255 if value == 0 else 0)
        cutout = ImageChops.multiply(region, black_pixels)
        image.putalpha(ImageChops.subtract(image.getchannel("A"), cutout))
    return image


class BitmapFont:
    """Read CYF's sprite font directly, without substituting a desktop font."""

    ALIASES = {
        " ": "space", "/": "slash", "?": "questionmark", ":": "colon",
        "&": "ampersand", "*": "asterisk", "'": "quote",
        '"': "doublequote", "\\": "backslash", ">": "gt", "<": "lt",
    }

    def __init__(self, stem: str = "uibattlesmall") -> None:
        self.atlas = gray_sprite(FONTS / f"{stem}.png")
        tree = ET.parse(FONTS / f"{stem}.xml")
        self.glyphs = {}
        for item in tree.findall("./spritesheet/sprite"):
            rect = item.find("rect")
            if rect is None:
                continue
            x, y, w, h = (int(rect.attrib[key]) for key in ("x", "y", "w", "h"))
            # CYF/Unity sprite rectangles use a bottom-left origin.
            top = self.atlas.height - y - h
            self.glyphs[item.attrib["name"]] = self.atlas.crop((x, top, x + w, top + h))

    def draw(self, frame: Image.Image, value: str, position: list[int], tracking: int = 2) -> None:
        x, y = position
        for char in value:
            key = self.ALIASES.get(char, char)
            if key not in self.glyphs:
                raise ValueError(f"The battle bitmap font does not contain {char!r}.")
            glyph = self.glyphs[key]
            if char != " ":
                frame.alpha_composite(glyph, (x, y + 15 - glyph.height))
            x += glyph.width + tracking


def render_scene(scene: dict) -> Image.Image:
    """Return one unscaled RGB frame. This is the future frame-rendering entry point."""
    canvas = scene["canvas"]
    if (canvas["width"], canvas["height"]) != (640, 480):
        raise ValueError("Keep the battle canvas at 640 x 480; change export_scale instead.")
    frame = Image.new("RGBA", (640, 480), (0, 0, 0, 255))

    # Separate body parts remain independently movable for later idle animation.
    ox, oy = scene["ink"]["origin"]
    for layer in scene["ink"]["layers"]:
        dx, dy = layer["offset"]
        frame.alpha_composite(actor_layer(layer), (ox + dx, oy + dy))

    arena = scene["arena"]
    x, y, w, h, border = (arena[key] for key in ("x", "y", "width", "height", "border"))
    if min(w, h) <= 2 * border or border < 1:
        raise ValueError("The arena must have a positive border and a usable interior.")
    draw = ImageDraw.Draw(frame)
    draw.rectangle((x, y, x + w - 1, y + h - 1), fill="black", outline="white", width=border)

    soul = scene["soul"]
    heart = gray_sprite(SPRITES / soul["sprite"])
    cx, cy = soul["center"]
    sx, sy = cx - heart.width // 2, cy - heart.height // 2
    if not (x + border <= sx and sx + heart.width <= x + w - border
            and y + border <= sy and sy + heart.height <= y + h - border):
        raise ValueError("The idle soul must fit inside the arena.")
    frame.alpha_composite(heart, (sx, sy))

    hud = scene["hud"]
    if not 0 <= hud["hp"] <= hud["max_hp"] or hud["max_hp"] <= 0:
        raise ValueError("HP must be between zero and the positive maximum HP.")
    bar = hud["hp_bar"]
    bx, by, bw, bh = (bar[key] for key in ("x", "y", "width", "height"))
    draw.rectangle((bx, by, bx + bw - 1, by + bh - 1), fill=(bar["empty"],) * 3)
    filled_width = round(bw * hud["hp"] / hud["max_hp"])
    if filled_width:
        draw.rectangle((bx, by, bx + filled_width - 1, by + bh - 1), fill=(bar["fill"],) * 3)

    font = BitmapFont()
    for value, position_key in (
        (hud["name"], "name_position"),
        (f"LV {hud['lv']}", "lv_position"),
        ("HP", "hp_label_position"),
        (f"{hud['hp']:02} / {hud['max_hp']:02}", "hp_value_position"),
    ):
        font.draw(frame, value, hud[position_key], hud["font_tracking"])

    for button in hud["buttons"]:
        frame.alpha_composite(gray_sprite(SPRITES / button["sprite"]), tuple(button["position"]))
    return frame.convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "scene.json")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "phase4-idle.png")
    parser.add_argument("--scale", type=int, help="Positive integer export scale; default from scene.json.")
    args = parser.parse_args()
    scene = json.loads(args.config.read_text(encoding="utf-8"))
    scale = args.scale if args.scale is not None else scene["canvas"]["export_scale"]
    if not isinstance(scale, int) or scale < 1:
        parser.error("--scale must be a positive integer.")
    frame = render_scene(scene)
    output = frame.resize((frame.width * scale, frame.height * scale), Image.Resampling.NEAREST)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.save(args.output)
    print(f"Saved {args.output} ({output.width} x {output.height})")


if __name__ == "__main__":
    main()
