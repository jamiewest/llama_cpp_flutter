#!/usr/bin/env bash
#
# FALLBACK ONLY — the normal path is scripts/fetch_llama_xcframework.sh, which
# installs the prebuilt xcframework from ggml-org/llama.cpp GitHub releases.
# Use this script when you need a slice/patch upstream does not publish, or
# when upstream stops shipping the xcframework asset.
#
# Builds llama.cpp as an Apple xcframework (iOS + macOS + tvOS + visionOS,
# Metal embedded) and vendors the result into this plugin at
# darwin/Frameworks/llama.xcframework.
#
# Requirements: Xcode (xcodebuild) and CMake on PATH.
#
# Pin the upstream version by setting LLAMA_REF to a tag, branch, or commit:
#   LLAMA_REF=b6500 ./scripts/build_llama_xcframework.sh
# Defaults to the LLAMA_CPP_TAG pinned in tool/versions.env.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PKG_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(cd "$PKG_DIR/../.." && pwd -P)"

# shellcheck source=../../../tool/versions.env
source "$REPO_ROOT/tool/versions.env"
LLAMA_REF="${LLAMA_REF:-$LLAMA_CPP_TAG}"
DEST="$PKG_DIR/darwin/Frameworks"
WORK="$(mktemp -d)"
REPO="$WORK/llama.cpp"

cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

echo "==> Cloning ggml-org/llama.cpp @ $LLAMA_REF"
if ! git clone --depth 1 --branch "$LLAMA_REF" \
    https://github.com/ggml-org/llama.cpp "$REPO" 2>/dev/null; then
  # LLAMA_REF is a commit sha (not shallow-cloneable by --branch).
  git clone https://github.com/ggml-org/llama.cpp "$REPO"
  git -C "$REPO" checkout "$LLAMA_REF"
fi

# This app ships only iOS + macOS. Trim the tvOS/visionOS slices from the
# upstream build to roughly halve build time and disk usage. Set
# LLAMA_ALL_PLATFORMS=1 to build every slice instead.
if [ "${LLAMA_ALL_PLATFORMS:-0}" != "1" ]; then
  echo "==> Trimming tvOS/visionOS slices from build-xcframework.sh"
  python3 - "$REPO/build-xcframework.sh" <<'PY'
import re, sys

path = sys.argv[1]
lines = open(path).read().split("\n")

# The function definitions (which contain `case "visionos")` / `"tvos")`
# branches) must NOT be touched. Only trim the linear driver section, which
# begins at the first top-level cmake configure for the iOS simulator.
start = next(
    i for i, ln in enumerate(lines) if ln.startswith("cmake -B build-ios-sim")
)
preamble, driver = lines[:start], lines[start:]

# Group driver physical lines into logical lines (joined on trailing backslash).
logical, buf = [], []
for ln in driver:
    buf.append(ln)
    if ln.rstrip().endswith("\\"):
        continue
    logical.append("\n".join(buf))
    buf = []
if buf:
    logical.append("\n".join(buf))

bad = re.compile(r"visionos|tvos", re.IGNORECASE)
out = []
for text in logical:
    if "create-xcframework" in text.lower():
        # Keep the command, drop only tvOS/visionOS -framework/-debug-symbols
        # argument lines (each is its own backslash-continued physical line).
        out.append("\n".join(p for p in text.split("\n") if not bad.search(p)))
    elif bad.search(text):
        # Drop whole top-level cmake/setup/combine statements for these slices.
        continue
    else:
        out.append(text)

open(path, "w").write("\n".join(preamble + out))
PY
fi

# Bundle llama.cpp's multimodal library (libmtmd) so the framework can run
# vision models (Gemma, LLaVA, …). Upstream gained native mtmd bundling
# (LLAMA_BUILD_MTMD=ON, libmtmd.a in the libs array, mtmd headers copied)
# between b9640 and b9899 — when that flag is present the patch below must be
# skipped, or libmtmd.a is merged twice and the link fails on ~400 duplicate
# symbols. Older refs still need the patch: libmtmd.a (and the libcommon.a it
# depends on) must be merged into each slice's combined archive, and the mtmd
# headers exposed via the module map.
if grep -q "LLAMA_BUILD_MTMD" "$REPO/build-xcframework.sh"; then
  echo "==> Upstream build-xcframework.sh already bundles libmtmd; skipping patch"
else
echo "==> Patching build-xcframework.sh to bundle libmtmd"
python3 - "$REPO/build-xcframework.sh" <<'PY'
import sys

path = sys.argv[1]
lines = open(path).read().split("\n")

# Insert `new` lines immediately after the unique line containing `needle`,
# inheriting that line's indentation so we don't have to hand-match the
# upstream whitespace (which differs by nesting depth).
def insert_after(lines, needle, new):
    hits = [i for i, ln in enumerate(lines) if needle in ln]
    if len(hits) != 1:
        sys.exit(f"expected one match for {needle!r}, found {len(hits)}")
    idx = hits[0]
    indent = lines[idx][: len(lines[idx]) - len(lines[idx].lstrip())]
    return lines[: idx + 1] + [indent + n for n in new] + lines[idx + 1 :]

# 1. Merge libmtmd.a + the libcommon.a it depends on into the combined static
#    archive that becomes the framework binary. Without these the app link fails
#    on undefined `mtmd_*` symbols. Upstream's `combine_static_libraries` keeps a
#    hardcoded `libs` array and its build-tree layout shifts between releases
#    (e.g. b9528 moved away from the old `tools/mtmd/${release_dir}` path), so we
#    LOCATE the archives by name in this platform's build dir and append them to
#    `libs` right before the libtool combine, rather than assuming a path. The
#    echoes make the merge visible in the build log.
lines = insert_after(
    lines,
    "# match the target architecture. We suppress these warnings.",
    [
        "for _extra in libcommon.a libmtmd.a; do",
        '    _hit="$(find "${base_dir}/${build_dir}" -name "$_extra" -print -quit 2>/dev/null)"',
        '    if [ -n "$_hit" ]; then',
        '        echo "==> mtmd merge: adding $_hit"',
        '        libs+=("$_hit")',
        "    else",
        '        echo "==> mtmd merge: WARNING $_extra not found under ${build_dir}" >&2',
        "    fi",
        "done",
    ],
)

# 2. Copy the mtmd public headers into the framework (run from the repo root,
#    like the surrounding cp lines).
lines = insert_after(
    lines,
    "cp ggml/include/gguf.h",
    [
        "cp tools/mtmd/mtmd.h           ${header_path}",
        "cp tools/mtmd/mtmd-helper.h    ${header_path}",
    ],
)

# 3. Expose the mtmd headers through the framework module map. Two upstream
#    module-map styles exist: older revisions list each header explicitly
#    (`header "gguf.h"` ...), where we must add mtmd's lines; newer revisions
#    (>= ~b9528) use `umbrella "Headers"`, which auto-includes every header in
#    the Headers dir — so the mtmd headers copied in step 2 are already covered
#    and nothing needs inserting. Branch on which style is present.
if any('header "gguf.h"' in ln for ln in lines):
    lines = insert_after(
        lines,
        'header "gguf.h"',
        ['header "mtmd.h"', 'header "mtmd-helper.h"'],
    )
elif not any("umbrella" in ln for ln in lines):
    sys.exit(
        "module map has neither an explicit 'header \"gguf.h\"' line nor an "
        "umbrella header; upstream layout changed, update this patch"
    )

src = "\n".join(lines)

def replace_once(src, old, new):
    if src.count(old) != 1:
        sys.exit(f"expected one match for {old!r}, found {src.count(old)}")
    return src.replace(old, new)

# 4. Tools AND common are off upstream (build-xcframework.sh sets
#    LLAMA_BUILD_TOOLS=OFF and LLAMA_BUILD_COMMON=OFF as top-level vars feeding a
#    shared COMMON_CMAKE_ARGS array). The mtmd target lives under tools/ and links
#    common, so BOTH must be on or libmtmd.a/libcommon.a never build (the symptom:
#    the `find` in step 1 reports them missing and the app fails to link on
#    undefined `mtmd_*` symbols).
src = replace_once(src, "LLAMA_BUILD_TOOLS=OFF", "LLAMA_BUILD_TOOLS=ON")
src = replace_once(src, "LLAMA_BUILD_COMMON=OFF", "LLAMA_BUILD_COMMON=ON")

# 5. ...but restrict every slice's build to the `mtmd` target so we get just its
#    transitive deps (llama, ggml*, common, mtmd) and not the CLI tools/server,
#    which bloat the build and may not cross-compile to iOS/visionOS. The build
#    lines pass `-- -quiet` through to xcodebuild (preceded by `-j <n>`, so we
#    anchor on the passthrough, not the full line); inject the target before it.
n_build = src.count(" -- -quiet")
if n_build == 0:
    sys.exit("found no ' -- -quiet' build lines to scope to --target mtmd")
src = src.replace(" -- -quiet", " --target mtmd -- -quiet")

open(path, "w").write(src)
PY
fi

echo "==> Building xcframework (this can take 10-30 minutes)"
( cd "$REPO" && ./build-xcframework.sh )

echo "==> Vendoring llama.xcframework into $DEST"
mkdir -p "$DEST"
rm -rf "$DEST/llama.xcframework"
cp -R "$REPO/build-apple/llama.xcframework" "$DEST/llama.xcframework"

echo "==> Done. Slices:"
ls "$DEST/llama.xcframework"
