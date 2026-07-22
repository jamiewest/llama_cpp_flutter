# Architecture

`llama_flutter` adapts local GGUF inference to the `ChatClient` contracts from
the Dart `agents` framework. It owns model metadata, prompt formatting, stream
decoding, diagnostics, and the runtime-neutral `LlamaRuntime` interface.

The runtime is selected with conditional exports:

- Flutter web uses the bundled wllama WASM asset and browser storage.
- iOS and macOS use the sibling `llama_flutter` plugin, whose Pigeon bridge
  sends inference work to a Dart worker isolate and a Swift llama.cpp session.

The web and native implementations must continue to satisfy the same runtime
and session interfaces. Application policy and UI belong in downstream apps,
not in either package in this repository.

## Development

Run commands from the repository root:

```sh
flutter pub get
flutter analyze packages/llama_flutter
flutter test packages/llama_flutter
flutter analyze packages/llama_flutter
```

Copy `pubspec_overrides.yaml.example` to `pubspec_overrides.yaml` to develop
against a sibling checkout of `jamiewest/agents`.

After changing `packages/llama_flutter/pigeons/messages.dart`, regenerate the
bridge from that package directory and commit both generated outputs:

```sh
dart run pigeon --input pigeons/messages.dart
```

Apple consumers obtain the ignored xcframework with
`packages/llama_flutter/scripts/fetch_llama_xcframework.sh`. Release artifacts
are built from the llama.cpp revision pinned in `build_llama_xcframework.sh`.
