#!/bin/zsh
# Same as esnupi reports-src/render.sh: index.html -> <id>.pdf via headless Chrome.
set -e
here=${0:A:h}; id=${1:-the-rational-transformer}
src="$here/$id/index.html"; out="$here/$id/$id.pdf"; rm -f "$out"
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --disable-gpu --no-pdf-header-footer \
  --allow-file-access-from-files --run-all-compositor-stages-before-draw --virtual-time-budget=8000 \
  --user-data-dir="$(mktemp -d)" --print-to-pdf="$out" "file://$src" >/dev/null 2>&1 &
pid=$!; for _ in {1..90}; do kill -0 $pid 2>/dev/null || break; sleep 1; done
kill $pid 2>/dev/null || true; wait $pid 2>/dev/null || true
[[ -s "$out" ]] || { echo "no PDF" >&2; exit 1; }
echo "$id: $(uv run --quiet --with pypdf python -c 'import sys,pypdf;print(len(pypdf.PdfReader(sys.argv[1]).pages))' "$out") pages, $(du -h "$out" | cut -f1)"
