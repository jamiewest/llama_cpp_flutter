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
