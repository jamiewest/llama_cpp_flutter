#!/usr/bin/env bash
#
# Installs the prebuilt llama.xcframework published on ggml-org/llama.cpp
# GitHub releases into darwin/Frameworks. The pinned tag and expected zip
# checksum live in tool/versions.env at the repo root (see CLAUDE.md).
#
# Idempotent: skips the download when the installed framework already matches
# the pinned tag + checksum. Override the pin with LLAMA_CPP_TAG_OVERRIDE,
# which requires the matching LLAMA_XCFRAMEWORK_ZIP_SHA256_OVERRIDE — the
# script refuses to download an unverified binary unless
# LLAMA_CPP_ALLOW_UNVERIFIED=1 explicitly opts out (development only).

set -euo pipefail

# pwd -P: resolve symlinks — under Flutter/CocoaPods this script runs through
# .symlinks/plugins/llama_cpp_flutter, and tool/versions.env only exists
# relative to the physical package location.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PACKAGE_DIR="$(dirname "$SCRIPT_DIR")"
FRAMEWORKS_DIR="$PACKAGE_DIR/darwin/Frameworks"
DEST="$FRAMEWORKS_DIR/llama.xcframework"
MARKER="$FRAMEWORKS_DIR/.llama.xcframework.sha256"

# shellcheck source=../tool/versions.env
source "$PACKAGE_DIR/tool/versions.env"

TAG="${LLAMA_CPP_TAG_OVERRIDE:-$LLAMA_CPP_TAG}"
EXPECTED="$LLAMA_XCFRAMEWORK_ZIP_SHA256"
if [ -n "${LLAMA_CPP_TAG_OVERRIDE:-}" ]; then
  if [ -n "${LLAMA_XCFRAMEWORK_ZIP_SHA256_OVERRIDE:-}" ]; then
    EXPECTED="$LLAMA_XCFRAMEWORK_ZIP_SHA256_OVERRIDE"
  elif [ "${LLAMA_CPP_ALLOW_UNVERIFIED:-0}" = "1" ]; then
    EXPECTED=""
    echo "##########################################################" >&2
    echo "# WARNING: downloading llama.xcframework for $TAG" >&2
    echo "# WITHOUT checksum verification" >&2
    echo "# (LLAMA_CPP_ALLOW_UNVERIFIED=1). Development only —" >&2
    echo "# never ship a build produced this way." >&2
    echo "##########################################################" >&2
  else
    echo "error: LLAMA_CPP_TAG_OVERRIDE=$TAG has no pinned checksum." >&2
    echo "Refusing to download an unverified binary. Either supply the" >&2
    echo "zip's checksum:" >&2
    echo "  LLAMA_XCFRAMEWORK_ZIP_SHA256_OVERRIDE=<sha256 of llama-$TAG-xcframework.zip>" >&2
    echo "or, for local development only, opt out of verification:" >&2
    echo "  LLAMA_CPP_ALLOW_UNVERIFIED=1" >&2
    exit 1
  fi
fi

URL="https://github.com/ggml-org/llama.cpp/releases/download/${TAG}/llama-${TAG}-xcframework.zip"

sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

if [ -d "$DEST" ] && [ -f "$MARKER" ]; then
  if [ -n "$EXPECTED" ] && [ "$(cat "$MARKER")" = "$TAG $EXPECTED" ]; then
    echo "llama.xcframework ($TAG) is already installed and verified"
    exit 0
  fi
  # Unverified override: any prior install of the same tag counts, since
  # there is no pinned checksum to compare against.
  if [ -z "$EXPECTED" ] && [ "$(cut -d' ' -f1 "$MARKER")" = "$TAG" ]; then
    echo "llama.xcframework ($TAG) is already installed (unverified override)"
    exit 0
  fi
fi

WORK="$(mktemp -d)"
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

echo "Downloading $URL"
curl --fail --location --silent --show-error \
  "$URL" --output "$WORK/llama-xcframework.zip"

ACTUAL="$(sha256 "$WORK/llama-xcframework.zip")"
if [ -n "$EXPECTED" ] && [ "$ACTUAL" != "$EXPECTED" ]; then
  echo "llama.xcframework checksum mismatch for $TAG" >&2
  echo "  expected: $EXPECTED" >&2
  echo "  actual:   $ACTUAL" >&2
  echo "If upstream re-published the release asset, re-pin via tool/update_deps.sh." >&2
  exit 1
fi

if command -v ditto >/dev/null 2>&1; then
  ditto -x -k "$WORK/llama-xcframework.zip" "$WORK/unpacked"
else
  unzip -q "$WORK/llama-xcframework.zip" -d "$WORK/unpacked"
fi

# Upstream zips nest the framework (currently under build-apple/); locate it
# instead of assuming a layout.
FW="$(find "$WORK/unpacked" -maxdepth 3 -type d -name llama.xcframework -print -quit)"
if [ -z "$FW" ]; then
  echo "llama.xcframework not found inside the release zip for $TAG" >&2
  exit 1
fi

mkdir -p "$FRAMEWORKS_DIR"
rm -rf "$DEST"
cp -R "$FW" "$DEST"
printf '%s %s\n' "$TAG" "$ACTUAL" > "$MARKER"
echo "Installed llama.xcframework ($TAG) at $DEST"
