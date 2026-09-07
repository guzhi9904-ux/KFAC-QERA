#!/usr/bin/env bash
set -euo pipefail

COMMIT="bd7fc86a2e44d41f95b9b0421f27f5624dd37064"
DESTINATION="${1:-/share/home/tm902089733300000/a913520780/chengkang/QERA-official-bd7fc86}"

if [[ -e "$DESTINATION" ]]; then
  echo "Destination already exists: $DESTINATION"
  echo "Run doctor to validate it; this script will not overwrite it."
  exit 0
fi

PARENT="$(dirname "$DESTINATION")"
mkdir -p "$PARENT"
TEMPORARY="$(mktemp -d "$PARENT/qera-download.XXXXXX")"
trap 'rm -rf "$TEMPORARY"' EXIT

echo "Downloading official QERA commit $COMMIT"
curl --http1.1 -fL \
  --retry 10 \
  --retry-all-errors \
  --retry-delay 3 \
  --connect-timeout 15 \
  --max-time 1800 \
  -o "$TEMPORARY/qera.tar.gz" \
  "https://codeload.github.com/ChengZhang-98/QERA/tar.gz/$COMMIT"

tar -xzf "$TEMPORARY/qera.tar.gz" -C "$TEMPORARY"
SOURCE="$TEMPORARY/QERA-$COMMIT"
printf '%s\n' "$COMMIT" > "$SOURCE/.qera_commit"
mv "$SOURCE" "$DESTINATION"
echo "Official QERA source installed at $DESTINATION"
