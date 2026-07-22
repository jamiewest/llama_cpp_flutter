# llama_flutter

On-device LLM inference for Flutter, backed by
[llama.cpp](https://github.com/ggml-org/llama.cpp) — local GGUF
`ChatClient` implementations for the
[`agents`](https://github.com/jamiewest/agents) framework, plus
memory-aware multi-agent orchestration over one loaded model.

Two entrypoints:

- `package:llama_flutter/llama_flutter.dart` — the cross-platform API most
  consumers want: neutral `LlamaRuntime`/`LlamaSession` abstraction,
  `ChatClient` adapter, chat formats, GGUF metadata reader, model
  downloader, and the orchestration layer (see `SPEC.md` and
  `doc/orchestration.md`). Native platforms run through the plugin bridge
  below; web runs through `@wllama/wllama` with the matching `wllama.wasm`
  shipped as a Flutter asset.
- `package:llama_flutter/bridge.dart` — the low-level iOS/macOS plugin
  bridge (kept separate because both layers define a `LlamaSession`):
  - **Native bridge:** [Pigeon](https://pub.dev/packages/pigeon) — a typed
    `@HostApi` for control and an `@EventChannelApi` token stream.
  - **Threading:** all native calls run from a dedicated Dart **worker
    isolate** (bound with `BackgroundIsolateBinaryMessenger`), so model
    loading and streaming never block the UI.
  - **Backend:** a vendored `llama.xcframework` (Metal-enabled).

## Setup: the vendored xcframework

The xcframework is large and is **not** committed. It is downloaded from the
official [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases)
automatically during `pod install` (the podspec runs the fetch script, which
is a no-op once the installed framework matches the pin in
`tool/versions.env`). Manual install / options:

```sh
./scripts/fetch_llama_xcframework.sh
# Try a different upstream release (checksum verification is skipped):
LLAMA_CPP_TAG_OVERRIDE=<tag> ./scripts/fetch_llama_xcframework.sh
# Skip the automatic fetch during pod install (offline/lint environments):
LLAMA_FLUTTER_SKIP_FETCH=1 pod install
```

This writes `darwin/Frameworks/llama.xcframework`. As a fallback, build from
source with `./scripts/build_llama_xcframework.sh` (requires Xcode + CMake;
`LLAMA_REF=<tag-or-commit>` to pin, `LLAMA_ALL_PLATFORMS=1` for every Apple
slice).

The `example/` app is a minimal harness that links the plugin; CI builds it on
macOS so every PR exercises compile + link against the pinned framework. Run a
real model through it with:

```sh
cd example
flutter test integration_test/model_smoke_test.dart -d macos \
  --dart-define=MODEL_PATH=/absolute/path/to/model.gguf
```

## Usage

```dart
final llama = LlamaFlutter();
final session = await llama.loadModel('/path/to/model.gguf');

await for (final token in session.generate('Hello, world!')) {
  stdout.write(token);
}

await session.dispose();
await llama.shutdown();
```

## Requirements & notes

- Deployment targets: **iOS 16.4 / macOS 13.3** (Metal build minimums).
- Models are loaded from a **runtime file path** — nothing is bundled. On
  sandboxed macOS the app needs `com.apple.security.files.user-selected.read-only`.
- The iOS **Simulator** has limited Metal support; pass `gpuLayers: 0` to
  `loadModel` to force CPU there.
- Regenerate the Pigeon bridge after editing `pigeons/messages.dart`:
  `dart run pigeon --input pigeons/messages.dart`.
