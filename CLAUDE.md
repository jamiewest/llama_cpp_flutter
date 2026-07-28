# llama_cpp_flutter

## How the xcframework gets installed

`scripts/fetch_llama_xcframework.sh` downloads the
pinned zip from ggml-org/llama.cpp releases, verifies the sha256, and unpacks
to `darwin/Frameworks/llama.xcframework` (gitignored). A marker file
(`.llama.xcframework.sha256`, storing `<tag> <sha>`) makes re-runs no-ops.

The podspec (`darwin/llama_cpp_flutter.podspec`) runs this script during podspec
evaluation, so plain `pod install` / `flutter build` works on a fresh checkout.
That hook is podspec-eval-time on purpose: CocoaPods does not run
`prepare_command` for `:path` (development) pods — which is how Flutter
integrates plugins — and `script_phase` runs after `vendored_frameworks`
resolution. Escape hatch: `LLAMA_CPP_FLUTTER_SKIP_FETCH=1`.

`scripts/build_llama_xcframework.sh` is a **fallback only** (source build with
mtmd patching for old refs); the normal path is the fetch script.

Future note (SPM): if the plugin migrates to Swift Package Manager, a
`binaryTarget(url:checksum:)` replaces this machinery natively — SPM uses its
own checksum format (`swift package compute-checksum`), so record that
alongside the zip sha at migration time.

## Staging ABI (LlamaExtShim)

`darwin/Classes/LlamaExtShim.{h,cpp}` re-declare two functions from llama.cpp's
*staging* header `src/llama-ext.h` (not shipped in the xcframework headers)
and link them by C++ mangling. They power the MTP speculative-decoding path in
`LlamaSession.swift`. Staging means upstream may change them at any time.

`tool/check_llama_ext_abi.sh [<tag>]` fetches `src/llama-ext.h` at the tag and
compares the two declarations against canonical signatures embedded in the
script. It runs in CI on every PR (against the pinned tag) and inside
`update_deps.sh` (against the candidate tag; a failure still opens the PR so
the break is visible, and CI keeps it red).

When it fails, update **together**:

1. the re-declared prototypes in `LlamaExtShim.cpp` (and, if the C surface
   changes, `LlamaExtShim.h`),
2. the call sites in `LlamaSession.swift` (search `llama_ext_`),
3. the canonical signature strings in `tool/check_llama_ext_abi.sh`.

The `build-macos` CI job is the link-time backstop: if the mangled symbols
drift, the example app fails to link.

## Upstream pins and the update PRs

All pins live in `tool/versions.env`: the llama.cpp release tag + xcframework
zip sha (native), and the wllama version + wasm sha/source (web).

The two backends update on **separate schedules**, because they have very
different risk profiles. llama.cpp tags 5-15 releases a day; a native re-pin
is cheap and reliable, while the wasm rebuild is a slow emscripten build that
breaks whenever llama.cpp's `common/` API drifts under wllama's glue. Keeping
them in one workflow meant a web-side compat break blocked routine native
bumps. Both workflows **validate before opening a PR**, so an open PR is one
that already passed its gates.

### `.github/workflows/update-llama-pin.yml` — native, daily

`tool/update_deps.sh` bumps the pin to the latest llama.cpp release
(downloads the zip, records the sha). "Latest" means the newest release whose
xcframework zip is actually downloadable, not `releases/latest`: llama.cpp
tags faster than its macOS job uploads assets, so the newest release often has
no zip yet, and some never get one (b10156). Resolving blindly would 404 at
random, which at a daily cadence becomes a recurring red run. The script walks
back from newest until a zip responds.

The PR opens only if both gates pass:

1. the staging-ABI tripwire at the new tag, and
2. the macOS link check (the example app builds against the new xcframework).

Dart analyze/test are not gates here — a native pin bump touches only
`tool/versions.env`, which the VM tests never exercise. Daily means "the pin
is never more than a day stale", not "every release is caught". A failed gate
fails the run and opens nothing.

### `.github/workflows/sync-wllama-wasm.yml` — web, weekly

Rebuilds `lib/assets/wasm/wllama.wasm` from `WLLAMA_SYNC_REPO@WLLAMA_SYNC_REF`
— the fork branch `jamiewest/wllama@flutter-sync`, cut from a wllama release
tag and carrying compat patches — with the `llama.cpp` submodule checked out
at the **currently pinned** tag. It does not bump `LLAMA_CPP_TAG` itself; it
catches the web side up to whatever the daily native workflow last landed.
The gates are `flutter analyze`, `flutter test`, and the browser-only suites.

If the build fails (glue drifted against newer llama.cpp), the run goes red
and **nothing is vendored** — the last known-good synced wasm stays in place.
This is deliberate: the earlier combined workflow refreshed from npm first and
overwrote with the synced build, so a failed build silently staged the *stock
npm* wasm, replacing a build that tracked the native pin with one that did
not. `tool/update_wllama.sh` is now only the manual/fallback path for when you
explicitly want the npm artifact.

The fix for a red sync build is another compat patch on the fork branch
(examples: `params_from_json_cmpl` → `server_schema::eval_llama_cmpl_schema`;
`use_mmap`/`use_mlock` moving out of `common_params`). When wllama publishes a
new release, rebase/recut `flutter-sync` from the new tag so the wasm stays
paired with the npm JS consumers load.

The wasm must stay paired with the `@wllama/wllama` JS version consumers
load (`WLLAMA_VERSION`); custom builds keep the pairing by building from
wllama's sources at exactly that tag, changing only the submodule. When a
PR changes the wasm, sanity-check a web build in a browser before merging —
the Dart suites load the runtime but do not prove inference is correct.
