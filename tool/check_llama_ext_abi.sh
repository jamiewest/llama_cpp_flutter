#!/usr/bin/env bash
#
# Verifies that llama.cpp's *staging* header src/llama-ext.h still declares
# the two functions LlamaExtShim.cpp re-declares and links by C++ mangling.
# Staging means upstream may change them at any time; when this fails, update
# together (see CLAUDE.md):
#   1. darwin/Classes/LlamaExtShim.{h,cpp}
#   2. the llama_ext_ call sites in darwin/Classes/LlamaSession.swift
#   3. the canonical signatures below
#
# Usage: check_llama_ext_abi.sh [<tag>]   (default: the pinned LLAMA_CPP_TAG)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=versions.env
source "$SCRIPT_DIR/versions.env"

TAG="${1:-$LLAMA_CPP_TAG}"
URL="https://raw.githubusercontent.com/ggml-org/llama.cpp/$TAG/src/llama-ext.h"

# Canonical signatures, whitespace-normalized. Must match the prototypes
# re-declared in darwin/Classes/LlamaExtShim.cpp.
CANONICAL=(
  'void llama_set_embeddings_nextn(struct llama_context * ctx, bool value, bool masked);'
  'float * llama_get_embeddings_nextn_ith(struct llama_context * ctx, int32_t i);'
)

header="$(curl --fail --location --silent --show-error "$URL")" || {
  echo "error: could not fetch $URL" >&2
  echo "(src/llama-ext.h may have been removed or moved at $TAG)" >&2
  exit 1
}

# Normalize: collapse whitespace, put one space around '*'.
normalize() {
  tr '\n' ' ' | sed -E 's/\*/ * /g; s/[[:space:]]+/ /g'
}

normalized="$(printf '%s' "$header" | normalize)"

status=0
for sig in "${CANONICAL[@]}"; do
  want="$(printf '%s' "$sig" | normalize)"
  if [[ "$normalized" != *"$want"* ]]; then
    echo "ABI MISMATCH at $TAG: missing declaration:" >&2
    echo "  $sig" >&2
    status=1
  fi
done

if [ "$status" -eq 0 ]; then
  echo "llama-ext ABI OK at $TAG (both staging declarations present)"
else
  echo "See the update procedure in CLAUDE.md (LlamaExtShim section)." >&2
fi
exit "$status"
