"""Render the Ink! Sans Fight phase-four video.

    python3 src/plan.py                 # plan the soul's dodges (writes output/plan.npz)
    python3 src/render.py               # full video -> output/ink-sans-phase4.mp4
    python3 src/render.py --still 42.5  # one 1920x1080 PNG
    python3 src/render.py --sheet 10 20 # contact sheet of 10..20 s
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

import choreo as C
import cinema
import intro as I
import scene as S
from common import ROOT, FPS, W, H, smooth

DURATION = 147.04
FADE_START, FADE_END = 144.7, 146.6

_state = {}


def init():
    import plan as P
    aim = np.load(ROOT / "output" / "aim.npy")
    C.AIM = P.aim_from(aim, P.frame_range()[0])
    tl = C.build()
    intro = I.Intro()
    S.SANS_HOOK = intro.sans_hook
    import backdrop
    S.BACKDROP = backdrop.Backdrop()
    import dialogue
    S.DIALOGUE = dialogue.Dialogue()
    dialogue.INK_HOOK = intro.bubble_hook
    import soak
    S.SOAK = soak.Soak()
    cinema.SOUL = lambda t: soul_at(int(round(t * FPS)))[0]
    plan = np.load(ROOT / "output" / "plan.npz")
    _state.update(tl=tl, intro=intro, plan=plan)


def soul_at(i: int):
    plan = _state["plan"]
    f0 = int(plan["f0"])
    k = min(max(i - f0, 0), len(plan["pos"]) - 1)
    return tuple(plan["pos"][k]), int(plan["hp"][k]), bool(plan["vis"][k])


def frame(i: int) -> np.ndarray:
    if not _state:
        init()
    t = i / FPS
    if t < I.INTRO_END:
        out = _state["intro"].frame(t)
    else:
        pos, hp, vis = soul_at(i)
        out = S.compose(t, _state["tl"], pos, hp, vis)
        out = cinema.apply(out, t)
    if t > FADE_START:
        g = 1 - smooth((t - FADE_START) / (FADE_END - FADE_START))
        out = (out.astype(np.float32) * g).astype(np.uint8)
    return out


def to_1080(img: np.ndarray) -> Image.Image:
    full = Image.new("L", (1920, 1080))
    full.paste(Image.fromarray(img).resize((1440, 1080), Image.Resampling.NEAREST), (240, 0))
    return full


def ffmpeg_exe() -> str:
    """ffmpeg on the PATH, or the one bundled with the imageio-ffmpeg package."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise SystemExit("FFmpeg is required (install it, or: pip install imageio-ffmpeg).")


def render_video(out: Path, start: float, end: float, workers: int):
    exe = ffmpeg_exe()
    init()                       # builds the caches once before the workers start
    f0, f1 = int(round(start * FPS)), int(math.ceil(end * FPS))
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [exe, "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pixel_format", "gray", "-video_size", f"{W}x{H}",
           "-framerate", str(FPS), "-i", "pipe:0",
           "-ss", f"{start:.4f}", "-i", str(ROOT / "Music" / "Larva.mp3"),
           "-map", "0:v:0", "-map", "1:a:0", "-t", f"{(f1 - f0) / FPS:.4f}",
           "-vf", "scale=1440:1080:flags=neighbor,pad=1920:1080:240:0:black,format=yuv420p",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "14", "-tune", "animation",
           "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart", str(out)]
    t0 = time.time()
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as enc:
        with mp.Pool(workers, initializer=init) as pool:
            for k, img in enumerate(pool.imap(frame, range(f0, f1), chunksize=24)):
                enc.stdin.write(img.tobytes())
                if k % (FPS * 10) == 0:
                    print(f"  {(f0 + k) / FPS:6.1f}s  ({time.time() - t0:.0f}s elapsed)", flush=True)
        enc.stdin.close()
        if enc.wait():
            raise RuntimeError("ffmpeg failed")
    print(f"saved {out}")


def sheet(times, path: Path, scale=2):
    init()
    tiles = [frame(int(round(t * FPS))) for t in times]
    cols = 4
    rows = math.ceil(len(tiles) / cols)
    th, tw = H // scale, W // scale
    canvas = np.full((rows * (th + 4), cols * (tw + 4)), 40, np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        small = np.asarray(Image.fromarray(tile).resize((tw, th), Image.Resampling.NEAREST))
        canvas[r * (th + 4):r * (th + 4) + th, c * (tw + 4):c * (tw + 4) + tw] = small
    Image.fromarray(canvas).save(path)
    print(f"saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--still", type=float)
    ap.add_argument("--sheet", type=float, nargs=2)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=DURATION)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count()))
    ap.add_argument("--out", type=Path, default=ROOT / "output" / "ink-sans-phase4.mp4")
    a = ap.parse_args()
    if a.still is not None:
        init()
        p = a.out if a.out.suffix == ".png" else ROOT / "output" / f"still-{a.still:.2f}.png"
        to_1080(frame(int(round(a.still * FPS)))).save(p)
        print(f"saved {p}")
    elif a.sheet:
        ts = np.linspace(a.sheet[0], a.sheet[1], a.n)
        p = a.out if a.out.suffix == ".png" else ROOT / "output" / "sheet.png"
        sheet(ts, p)
    else:
        render_video(a.out, a.start, a.end, a.workers)


if __name__ == "__main__":
    main()
