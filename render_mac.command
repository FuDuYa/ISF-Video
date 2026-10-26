#!/bin/bash
# Render a preview on this Mac. Double-click it (or: bash render_mac.command [start] [end]).
# The time range comes from the arguments, else from render_range.txt ("start end"),
# else 0-33.85 s. The first run sets up a Python environment in .venv.
cd "$(dirname "$0")" || exit 1
set -e
export PATH="/opt/homebrew/bin:/usr/local/bin:/Library/Frameworks/Python.framework/Versions/Current/bin:$PATH"
if [ ! -x .venv/bin/python ]; then
  echo "First run: setting up the Python environment..."
  PY=""
  for c in python3.14 python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
  done
  [ -n "$PY" ] || { echo "Python 3 not found - install it from python.org"; exit 1; }
  echo "Using $(command -v "$PY") ($("$PY" --version))"
  "$PY" -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r requirements.txt imageio-ffmpeg
fi
read -r R0 R1 < render_range.txt 2>/dev/null || true
START="${1:-${R0:-0}}"
END="${2:-${R1:-33.85}}"
OUT="output/preview-${START}-${END}s.mp4"
echo "Rendering ${START}-${END} s -> ${OUT}"
.venv/bin/python src/render.py --start "$START" --end "$END" --out "$OUT"
echo "Done: ${OUT}"
open "$OUT"
