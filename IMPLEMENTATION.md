# Architecture

`llama_cpp_flutter` adapts local GGUF inference to the `ChatClient` contracts from
`package:extensions/ai.dart`. It owns model metadata, prompt formatting, stream
decoding, diagnostics, and the runtime-neutral `LlamaRuntime` interface. Agent
frameworks sit on top of that `ChatClient`; none is depended on here.

The runtime is selected with conditional exports:

- Flutter web uses the bundled wllama WASM asset and browser storage.
- iOS and macOS use the sibling `llama_cpp_flutter` plugin, whose Pigeon bridge
  sends inference work to a Dart worker isolate and a Swift llama.cpp session.

The web and native implementations must continue to satisfy the same runtime
and session interfaces. Application policy and UI belong in downstream apps,
not in either package in this repository.

## Development

Run commands from the repository root:

```sh
flutter pub get
flutter analyze packages/llama_cpp_flutter
flutter test packages/llama_cpp_flutter
flutter analyze packages/llama_cpp_flutter
```

After changing `packages/llama_cpp_flutter/pigeons/messages.dart`, regenerate the
bridge from that package directory and commit both generated outputs:

```sh
dart run pigeon --input pigeons/messages.dart
```

Apple consumers obtain the ignored xcframework with
`packages/llama_cpp_flutter/scripts/fetch_llama_xcframework.sh`. Release artifacts
are built from the llama.cpp revision pinned in `build_llama_xcframework.sh`.
