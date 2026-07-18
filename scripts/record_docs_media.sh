#!/usr/bin/env bash
#
# Convert a screen recording into a documentation GIF.
#
#   scripts/record_docs_media.sh raw.mov lasso-selection
#
# Writes docs/_static/media/<name>.gif. See scripts/DOCS_MEDIA.md for the
# shot list, the conventions and how to swap the placeholder out of the page.
#
# gifski is used rather than ffmpeg's palettegen/paletteuse because the content
# here - thousands of small saturated points on black - is exactly what a global
# 256-colour palette handles worst.

set -euo pipefail

FPS="${FPS:-15}"
WIDTH="${WIDTH:-1280}"
QUALITY="${QUALITY:-90}"

usage() {
    cat >&2 <<'EOF'
usage: scripts/record_docs_media.sh <input.mov> <output-name>

  <input.mov>    a screen recording (macOS: Cmd+Shift+5)
  <output-name>  basename without extension, e.g. lasso-selection

environment:
  FPS=15         frames per second
  WIDTH=1280     output width in pixels (height follows aspect ratio)
  QUALITY=90     gifski quality, 1-100

Drop FPS or WIDTH before dropping QUALITY if the result is over 3 MB.
EOF
    exit 2
}

[ $# -eq 2 ] || usage

INPUT="$1"
NAME="$2"

[ -f "$INPUT" ] || { echo "error: no such file: $INPUT" >&2; exit 1; }

for tool in ffmpeg gifski; do
    command -v "$tool" >/dev/null 2>&1 || {
        echo "error: $tool is not installed" >&2
        echo "  brew install $tool" >&2
        exit 1
    }
done

# Resolve relative to the repo root, so the script works from any directory.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTDIR="$ROOT/docs/_static/media"
OUT="$OUTDIR/$NAME.gif"

mkdir -p "$OUTDIR"

echo "converting $INPUT -> $OUT (${WIDTH}px, ${FPS}fps, q${QUALITY})"

ffmpeg -loglevel error -i "$INPUT" \
    -vf "fps=$FPS,scale=$WIDTH:-1:flags=lanczos" \
    -f yuv4mpegpipe - \
  | gifski -o "$OUT" --fps "$FPS" --quality "$QUALITY" -

SIZE_BYTES=$(wc -c < "$OUT" | tr -d ' ')
SIZE_MB=$(( SIZE_BYTES / 1024 / 1024 ))

echo "wrote $OUT (${SIZE_MB} MB)"

if [ "$SIZE_BYTES" -gt 3145728 ]; then
    cat >&2 <<EOF

warning: ${SIZE_MB} MB is over the 3 MB target. Try:

    FPS=12 $0 "$INPUT" "$NAME"
    WIDTH=1000 $0 "$INPUT" "$NAME"

or trim the recording - clips should be 8-15 seconds.
EOF
fi
