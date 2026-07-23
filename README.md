# llama_cpp_flutter

On-device LLM inference for Flutter, backed by
[llama.cpp](https://github.com/ggml-org/llama.cpp): load a local GGUF model
and stream generated text through one cross-platform API. Chat formats,
model downloading, KV-cache snapshots, multimodal input, and an adapter for
the [`agents`](https://github.com/jamiewest/agents) framework are layered on
top — use as much or as little as you need.

## Platform support

| Platform | Status    | Backend                                          |
| -------- | --------- | ------------------------------------------------ |
| iOS      | Supported | llama.cpp xcframework, Metal                     |
| macOS    | Supported | llama.cpp xcframework, Metal                     |
| Web      | Supported | [wllama](https://github.com/ngxson/wllama), Wasm |
| Android  | Not yet   | see [Roadmap](#roadmap)                          |
| Windows  | Not yet   | —                                                |
| Linux    | Not yet   | —                                                |

On unsupported platforms `createLlamaRuntime()` throws an
`UnsupportedError` naming the platform, rather than failing later with a
missing-plugin error.

## Quick start

`createLlamaRuntime()` returns the platform's [`LlamaRuntime`] — the
llama.cpp plugin on iOS/macOS, wllama on web. Describe the model with a
`ModelSpec`, load it, and stream:

```dart
import 'dart:io';

import 'package:llama_cpp_flutter/llama_cpp_flutter.dart';

Future<void> main() async {
  final runtime = createLlamaRuntime();

  final spec = ModelSpec(
    id: 'gemma-3-4b-it-q4km',
    displayName: 'Gemma 3 4B IT',
    modelUrl: huggingFaceModelUri(
      repo: 'unsloth/gemma-3-4b-it-GGUF',
      file: 'gemma-3-4b-it-Q4_K_M.gguf',
    ),
    contextSize: 4096,
    format: resolveChatFormat('gemma')!,
  );

  final session = await runtime.loadModel(
    spec,
    localPath: '/path/to/gemma-3-4b-it-Q4_K_M.gguf',
  );

  await for (final text in session.generate('Why is the sky blue?')) {
    stdout.write(text);
  }

  await session.dispose();
}
```

`generate` consumes a raw prompt string. For multi-turn conversations,
render the prompt through the spec's `ChatFormat` — or skip prompt handling
entirely and use the [`ChatClient` adapter](#using-with-the-agents-framework).

### Where the model file comes from

- **Native, already on device:** pass `localPath` as above (e.g. a
  user-picked file, or a path your app downloaded earlier).
- **Native, downloaded for you:** give the runtime an artifact cache
  directory and omit `localPath` — the spec's URLs (model, optional
  multimodal projector, optional draft model) are fetched into it first,
  skipping files already present:

  ```dart
  final runtime = createLlamaRuntime(
    artifactCacheDirectory: cacheDir.path,
  );
  final session = await runtime.loadModel(
    spec,
    onProgress: (progress) => debugPrint('load: $progress'),
  );
  ```

  For manual control over downloads (auth tokens, separate progress per
  artifact), use `ModelDownloader.downloadSpecArtifacts` and pass the
  resulting paths to `loadModel` yourself.

- **Web:** omit `localPath` and the runtime streams `spec.modelUrl`
  directly (Hugging Face `resolve` URLs send the CORS headers browsers
  need). `huggingFaceModelUri` builds those URLs on every platform.

### Web notes

The wllama Wasm binary ships as a Flutter asset — no extra setup. For
multi-threaded inference the page must be cross-origin isolated (served
with `Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp`); otherwise wllama silently
falls back to a single thread, which is slow enough to look like a hang for
multi-billion-parameter models. Check
`runtime.supportsMultiThreading` and surface a warning to users when it is
false.

## Chat formats

A `ChatFormat` renders conversation turns into a model family's wire format
and decodes its output (including streamed tool calls). Built-ins cover
Gemma, Llama 3, Mistral, Qwen, LFM2/LFM2.5, and generic ChatML; resolve one
by name with `resolveChatFormat`, register your own with
`registerChatFormat`.

You can also detect the right format from the model file itself —
`detectChatFormatNameForGguf` reads the GGUF's architecture and embedded
chat template:

```dart
final name = detectChatFormatNameForGguf(headerBytes);
final format = resolveChatFormat(name) ?? resolveChatFormat('chatml')!;
```

## Using with the `agents` framework

`createLlamaChatClient` wraps a loaded session in a `ChatClient`, handling
prompt rendering, sampling defaults, tool-call decoding, and multimodal
turns:

```dart
final client = createLlamaChatClient(
  spec: spec,
  sessionProvider: () async => session,
);

final response = await client.getResponse(
  messages: [ChatMessage.fromText(ChatRole.user, 'Hello!')],
);
print(response.text);
```

The client plugs into anything that accepts a `ChatClient` from
`package:extensions/ai.dart`, including agents built with the
[`agents`](https://github.com/jamiewest/agents) framework.

## Structured generation events

`generateEvents` is the typed alternative to the raw string stream: text
arrives as events, and a successful run always ends with a completed event
carrying the engine's stats — so the finish reason and token accounting
cannot be missed. Failures surface as stream errors.

```dart
await for (final event in session.generateEvents('Why is the sky blue?')) {
  switch (event) {
    case LlamaTextEvent(:final text):
      stdout.write(text);
    case LlamaCompletedEvent(:final stats):
      debugPrint('finished: ${stats?.finishReason}');
    case LlamaWarningEvent(:final message):
      debugPrint('warning: $message');
  }
}
```

## Concurrency and lifecycle

- A session runs **one generation at a time**. Calling `generate` while a
  run is in flight supersedes it: the engine cancels the current run and
  starts the newest call once cancellation completes. Prefer awaiting or
  cancelling the previous run explicitly.
- `session.cancel()` stops the in-flight run; the stream completes normally
  and the stats callback (when the engine reports one) carries
  `LlamaFinishReason.cancelled`. Cancelling the stream subscription has the
  same effect.
- `session.dispose()` is safe during generation — the stream closes, though
  a stats callback may not be delivered.
- `maxSequences > 1` gives a session independent KV-cache *sequences*
  (separate conversations sharing one loaded model), not parallel decode
  lanes; `sequenceId` selects which one a run extends.

See the `LlamaSession` API docs for the full contract, including KV-cache
persistence (`saveState`/`loadState`) and in-memory stashing
(`stashState`/`restoreStashedState`).

## Multi-agent orchestration

The orchestration layer runs several logical agents over one loaded model:
per-agent KV-cache swapping via sequences and stashes, GGUF-driven memory
estimation, and dynamic context budgeting. It is entirely optional and
lives in its own entrypoint — see `doc/orchestration.md` and `SPEC.md`:

```dart
import 'package:llama_cpp_flutter/orchestration.dart';
```

## Package layout

The main entrypoint stays deliberately small; deeper layers have their own
imports:

| Entrypoint            | Contents                                          |
| --------------------- | ------------------------------------------------- |
| `llama_cpp_flutter.dart` | Runtime, sessions, `ModelSpec`, downloads, format resolution, `ChatClient` adapter |
| `chat.dart`           | Concrete chat formats, templates, stream decoders, `LlamaChatClient`, prompt diagnostics |
| `gguf.dart`           | GGUF metadata reading, artifact cache naming      |
| `orchestration.dart`  | Multi-agent orchestration over one loaded model   |
| `bridge.dart`         | Low-level iOS/macOS plugin bridge                 |

## Setup: the vendored xcframework (iOS/macOS)

The xcframework is large and is **not** committed. It is downloaded from the
official [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases)
automatically during `pod install` (the podspec runs the fetch script, which
is a no-op once the installed framework matches the pin in
`tool/versions.env`). Manual install / options:

```sh
./scripts/fetch_llama_xcframework.sh
# Try a different upstream release; its zip checksum is required:
LLAMA_CPP_TAG_OVERRIDE=<tag> \
  LLAMA_XCFRAMEWORK_ZIP_SHA256_OVERRIDE=<sha256> \
  ./scripts/fetch_llama_xcframework.sh
# Development-only escape hatch when you don't have the checksum yet
# (prints a loud warning; never ship a build produced this way):
LLAMA_CPP_TAG_OVERRIDE=<tag> LLAMA_CPP_ALLOW_UNVERIFIED=1 \
  ./scripts/fetch_llama_xcframework.sh
# Skip the automatic fetch during pod install (offline/lint environments):
LLAMA_CPP_FLUTTER_SKIP_FETCH=1 pod install
```

Downloads are always verified against the sha256 pinned in
`tool/versions.env` (or the explicit override above) before unpacking.

Both backends track upstream automatically: a weekly workflow re-pins the
native xcframework to the latest llama.cpp release and rebuilds the wllama
wasm against that same tag, opening a PR gated by CI (ABI check + macOS
link). Manual equivalents: `tool/update_deps.sh` and
`tool/update_wllama.sh`.

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

## Advanced: the native bridge

`package:llama_cpp_flutter/bridge.dart` is the low-level iOS/macOS plugin
bridge underneath the neutral API. Most apps never need it; reach for it
only for Apple-specific control the neutral API doesn't expose. Its
session type is `LlamaBridgeSession`, so it coexists cleanly with the
neutral `LlamaSession`.

```dart
import 'package:llama_cpp_flutter/bridge.dart';

final llama = LlamaCppFlutter();
final session = await llama.loadModel('/path/to/model.gguf');

await for (final token in session.generate('Hello, world!')) {
  stdout.write(token);
}

await session.dispose();
await llama.shutdown();
```

How it works:

- **Native bridge:** [Pigeon](https://pub.dev/packages/pigeon) — a typed
  `@HostApi` for control and an `@EventChannelApi` token stream.
- **Threading:** all native calls run from a dedicated Dart **worker
  isolate** (bound with `BackgroundIsolateBinaryMessenger`), so model
  loading and streaming never block the UI.
- **Backend:** a vendored `llama.xcframework` (Metal-enabled).

## Roadmap

- **Android** is the most-requested missing platform and the next target
  for a native backend; there is no committed timeline yet. Windows and
  Linux are open beyond that. The neutral `LlamaRuntime` API is designed so
  new backends slot in without app-code changes.
- The API is pre-1.0 and may still see breaking changes; they are called
  out per release in the changelog.

## Requirements & notes

- SDK: **Dart ^3.11.5 / Flutter >=3.41.7** — the floor comes from the
  `agents` dependency, not this package's own code.
- Deployment targets: **iOS 16.4 / macOS 13.3** (Metal build minimums).
- Models are loaded from a **runtime file path** — nothing is bundled. On
  sandboxed macOS the app needs `com.apple.security.files.user-selected.read-only`.
- The iOS **Simulator** has limited Metal support; pass `gpuLayers: 0` to
  force CPU there.
- Model weights have their own licenses — shipping a model in your app
  means complying with that model's license, not this package's.
- Regenerate the Pigeon bridge after editing `pigeons/messages.dart`:
  `dart run pigeon --input pigeons/messages.dart`.

[`LlamaRuntime`]: https://pub.dev/documentation/llama_cpp_flutter/latest/
