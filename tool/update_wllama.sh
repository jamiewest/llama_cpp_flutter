#!/usr/bin/env bash
#
# Updates the vendored wllama wasm (lib/assets/wasm/wllama.wasm) from the
# @wllama/wllama npm release and records the provenance in tool/versions.env.
#
# The wasm must match the @wllama/wllama JS version the consuming page
# loads — WLLAMA_VERSION in versions.env is that pairing.
#
# Usage:
#   tool/update_wllama.sh            # latest npm release
#   tool/update_wllama.sh 3.5.1     # specific version
#
# Note: the npm wasm embeds whatever llama.cpp wllama's release pinned,
# which usually lags the native xcframework pin. The scheduled
# update-deps workflow rebuilds this artifact from wllama's sources with
# the llama.cpp submodule set to LLAMA_CPP_TAG, so both platforms track
# the same upstream; this script is the manual/fallback path.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PACKAGE_DIR="$(dirname "$SCRIPT_DIR")"
VERSIONS="$SCRIPT_DIR/versions.env"
DEST="$PACKAGE_DIR/lib/assets/wasm/wllama.wasm"

if [ -n "${1:-}" ]; then
  VERSION="$1"
else
  VERSION="$(curl --fail --silent --show-error \
    https://registry.npmjs.org/@wllama/wllama/latest \
    | sed -n 's/.*"version": *"\([^"]*\)".*/\1/p' | head -n1)"
  [ -n "$VERSION" ] || { echo "error: could not resolve latest version" >&2; exit 1; }
fi

WORK="$(mktemp -d)"
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

URL="https://registry.npmjs.org/@wllama/wllama/-/wllama-${VERSION}.tgz"
echo "Downloading $URL"
curl --fail --location --silent --show-error "$URL" --output "$WORK/wllama.tgz"
tar xzf "$WORK/wllama.tgz" -C "$WORK" package/src/wasm/wllama.wasm

if command -v sha256sum >/dev/null 2>&1; then
  SHA="$(sha256sum "$WORK/package/src/wasm/wllama.wasm" | awk '{print $1}')"
else
  SHA="$(shasum -a 256 "$WORK/package/src/wasm/wllama.wasm" | awk '{print $1}')"
fi

mkdir -p "$(dirname "$DEST")"
cp "$WORK/package/src/wasm/wllama.wasm" "$DEST"

record() {
  local key="$1" value="$2"
  if grep -q "^$key=" "$VERSIONS"; then
    sed -i.bak "s|^$key=.*|$key=$value|" "$VERSIONS" && rm -f "$VERSIONS.bak"
  else
    printf '%s=%s\n' "$key" "$value" >> "$VERSIONS"
  fi
}
record WLLAMA_VERSION "$VERSION"
record WLLAMA_WASM_SOURCE "npm"
record WLLAMA_WASM_SHA256 "$SHA"

echo "Installed wllama.wasm from @wllama/wllama@$VERSION (sha256: $SHA)"
