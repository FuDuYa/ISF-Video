#!/bin/bash
# 一键生成预览视频（Mac）。双击运行；也可以在终端里：bash 一键生成视频.command 0 33.85
#
# 它会依次：
#   1. 拉取 GitHub 上的最新代码（只快进；你本地没提交的改动不会被覆盖，拉不下来就用本地代码继续）
#   2. 第一次运行时自动建好 Python 环境（.venv）；requirements.txt 变了会自动补装
#   3. 代码、谱面、贴图有变化（或者还没规划过）就先跑走位规划 plan.py（约 2 分钟），没变就跳过
#   4. 渲染 output/preview-<开始>-<结束>s.mp4，完成后自动打开
#
# 时间范围（秒）：命令后面的两个参数 > render_range.txt 里的“开始 结束” > 默认 0 33.85
# 不想拉取代码：NO_PULL=1 bash 一键生成视频.command

cd "$(dirname "$0")" || exit 1
export PATH="/opt/homebrew/bin:/usr/local/bin:/Library/Frameworks/Python.framework/Versions/Current/bin:$PATH"

fail() {
  echo
  echo "出错了：$1"
  echo "把上面的输出截图发给 Claude 就行。"
  exit 1
}

# ---- 1. 拉取最新代码 ----
if [ -z "$NO_PULL" ] && [ -d .git ] && [ "$(uname)" = Darwin ] \
   && xcode-select -p >/dev/null 2>&1 && command -v git >/dev/null 2>&1; then
  echo "== 拉取最新代码 =="
  if ! GIT_TERMINAL_PROMPT=0 git pull --ff-only; then
    echo "（没能拉取：可能没联网，或者本地改动和远程冲突。先用本地现有的代码继续。）"
  fi
fi

# ---- 2. Python 环境 ----
if [ ! -x .venv/bin/python ]; then
  echo "== 第一次运行：建立 Python 环境 =="
  PY=""
  for c in python3.14 python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
  done
  [ -n "$PY" ] || fail "找不到 Python 3，请先从 python.org 安装"
  echo "使用 $(command -v "$PY")（$("$PY" --version)）"
  "$PY" -m venv .venv || fail "建立 Python 环境失败"
  rm -f .venv/.req-stamp
fi
REQ_SIG="$(cat requirements.txt) imageio-ffmpeg"
if [ "$(cat .venv/.req-stamp 2>/dev/null)" != "$REQ_SIG" ]; then
  echo "== 安装依赖 =="
  .venv/bin/python -m pip install --upgrade pip >/dev/null 2>&1 || true
  .venv/bin/python -m pip install -r requirements.txt imageio-ffmpeg || fail "安装依赖失败（需要联网）"
  printf '%s' "$REQ_SIG" > .venv/.req-stamp
fi

# ---- 时间范围 ----
R0=""; R1=""
read -r R0 R1 < render_range.txt 2>/dev/null
R0="${R0%$'\r'}"; R1="${R1%$'\r'}"
START="${1:-${R0:-0}}"
END="${2:-${R1:-33.85}}"

# ---- 3. 走位规划（代码有变化才重跑）----
mkdir -p output
SIG="$(.venv/bin/python - <<'EOF'
import hashlib
from pathlib import Path
h = hashlib.sha256()
files = sorted(Path("src").glob("*.py")) + [Path("maidata.txt"), Path("scene.json")]
files += sorted(p for p in Path("sprite").rglob("*") if p.is_file())
for p in files:
    h.update(p.as_posix().encode())
    h.update(p.read_bytes())
print(h.hexdigest())
EOF
)" || fail "检查代码变化失败"
if [ -f output/plan.npz ] && [ -f output/aim.npy ] && [ "$(cat output/plan.sig 2>/dev/null)" = "$SIG" ]; then
  echo "== 走位规划：代码没变，沿用上次的结果 =="
else
  echo "== 走位规划（约 2 分钟）=="
  rm -f output/plan.sig
  .venv/bin/python src/plan.py || fail "走位规划失败"
  echo "$SIG" > output/plan.sig
fi

# ---- 4. 渲染 ----
OUT="output/preview-${START}-${END}s.mp4"
echo "== 渲染 ${START}–${END} 秒 -> ${OUT} =="
.venv/bin/python src/render.py --start "$START" --end "$END" --out "$OUT" || fail "渲染失败"
echo
echo "完成：${OUT}（用时 $((SECONDS / 60)) 分 $((SECONDS % 60)) 秒）"
if [ "$(uname)" = Darwin ]; then open "$OUT"; fi
