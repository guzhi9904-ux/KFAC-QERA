#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMMIT=3823cfec41c016378acbcc8616dd1ac92c15edd4
DESTINATION="${HARNESS_SOURCE:-/share/home/tm902089733300000/a913520780/chengkang/QERA-harness-3823cfe}"
PARENT="$(dirname "$DESTINATION")"
mkdir -p "$PARENT"
TEMPORARY="$(mktemp -d "$PARENT/harness-install.XXXXXX")"
# Keep this small staging directory (constraints and environment snapshot) for audit.
echo "Installation audit directory: $TEMPORARY"
python "$HERE/harness_word_ppl.py" protect-env \
  --constraints "$TEMPORARY/core-constraints.txt" --snapshot "$TEMPORARY/core-before.json"

if [[ ! -e "$DESTINATION" ]]; then
  echo "Downloading QERA's pinned lm-evaluation-harness: $COMMIT"
  curl --http1.1 -fL --retry 3 --retry-delay 3 --connect-timeout 15 --max-time 1800 \
    -o "$TEMPORARY/harness.tar.gz" \
    "https://codeload.github.com/ChengZhang-98/lm-evaluation-harness/tar.gz/$COMMIT"
  tar -xzf "$TEMPORARY/harness.tar.gz" -C "$TEMPORARY"
  mv "$TEMPORARY/lm-evaluation-harness-$COMMIT" "$DESTINATION"
fi
python "$HERE/harness_word_ppl.py" verify --source "$DESTINATION"

# Exact constraints protect the user's installed torch/CUDA and core libraries.
# The pinned evaluate version supports the existing datasets 2.21 environment.
python -m pip install --progress-bar on \
  --constraint "$TEMPORARY/core-constraints.txt" \
  "evaluate==0.4.3" -e "$DESTINATION"
python "$HERE/harness_word_ppl.py" verify --source "$DESTINATION" \
  --check-import --snapshot "$TEMPORARY/core-before.json"
echo "Harness installation complete; existing core package versions are unchanged."
