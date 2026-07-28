#!/usr/bin/env bash
#
# Re-pins the llama.cpp release used for the native xcframework.
#
# Resolves the latest upstream release (or takes an explicit tag), downloads
# the xcframework zip, records tag + sha256 in tool/versions.env, checks the
# staging ABI (LlamaExtShim), and installs the framework locally.
#
# Usage:
#   tool/update_deps.sh              # pin to the latest upstream release
#   tool/update_deps.sh b10091      # pin to a specific tag
#   NO_INSTALL=1 tool/update_deps.sh # skip the local framework install (CI)
#
# An ABI-check failure does NOT abort the re-pin: the pin lands so the break
# is visible in review/CI, per the procedure in CLAUDE.md.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PACKAGE_DIR="$(dirname "$SCRIPT_DIR")"
VERSIONS="$SCRIPT_DIR/versions.env"

# shellcheck source=versions.env
source "$VERSIONS"

if [ -n "${1:-}" ]; then
  TAG="$1"
else
  # llama.cpp tags releases faster than its macOS job uploads assets, so the
  # newest release frequently has no xcframework zip yet (and the occasional
  # one never gets one — b10156 shipped without it). Resolving plain
  # `releases/latest` therefore 404s at random, which a daily cron would turn
  # into a recurring red run. Walk back from newest and take the first release
  # whose zip is actually downloadable.
  TAG=""
  for CANDIDATE in $(curl --fail --silent --show-error \
      "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=20" \
      | grep -o '"tag_name": *"[^"]*"' | sed 's/.*: *"\(.*\)"/\1/'); do
    if curl --fail --silent --head --location \
        "https://github.com/ggml-org/llama.cpp/releases/download/${CANDIDATE}/llama-${CANDIDATE}-xcframework.zip" \
        >/dev/null 2>&1; then
      TAG="$CANDIDATE"
      break
    fi
    echo "note: $CANDIDATE has no xcframework zip; trying the previous release" >&2
  done
  [ -n "$TAG" ] || {
    echo "error: no release in the last 20 has an xcframework zip" >&2
    exit 1
  }
fi

if [ "$TAG" = "$LLAMA_CPP_TAG" ]; then
  echo "llama.cpp pin is already $TAG; nothing to do."
  exit 0
fi

URL="https://github.com/ggml-org/llama.cpp/releases/download/${TAG}/llama-${TAG}-xcframework.zip"
WORK="$(mktemp -d)"
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

echo "Downloading $URL"
curl --fail --location --silent --show-error "$URL" \
  --output "$WORK/llama-xcframework.zip"

if command -v sha256sum >/dev/null 2>&1; then
  SHA="$(sha256sum "$WORK/llama-xcframework.zip" | awk '{print $1}')"
else
  SHA="$(shasum -a 256 "$WORK/llama-xcframework.zip" | awk '{print $1}')"
fi

sed -i.bak \
  -e "s/^LLAMA_CPP_TAG=.*/LLAMA_CPP_TAG=$TAG/" \
  -e "s/^LLAMA_XCFRAMEWORK_ZIP_SHA256=.*/LLAMA_XCFRAMEWORK_ZIP_SHA256=$SHA/" \
  "$VERSIONS"
rm -f "$VERSIONS.bak"
echo "Re-pinned: $LLAMA_CPP_TAG -> $TAG"
echo "  sha256: $SHA"

# Staging-ABI check; a failure is reported but does not revert the pin.
if ! "$SCRIPT_DIR/check_llama_ext_abi.sh" "$TAG"; then
  echo "WARNING: staging ABI check failed for $TAG — LlamaExtShim needs" >&2
  echo "updating before this pin can ship (see CLAUDE.md)." >&2
fi

if [ -z "${NO_INSTALL:-}" ]; then
  "$PACKAGE_DIR/scripts/fetch_llama_xcframework.sh"
  echo
  echo "Next: build the example app (link check) and run the tests:"
  echo "  (cd example && flutter build macos --debug) && flutter test"
fi
