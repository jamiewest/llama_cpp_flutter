# llama_flutter_example

Minimal harness that links `llama_flutter` against the vendored
`llama.xcframework`. CI builds it on macOS so every PR exercises compile +
link of the plugin's Swift/C++ against the pinned framework.

Run a real model end-to-end (loads a GGUF and streams a short generation):

```sh
flutter test integration_test/model_smoke_test.dart -d macos \
  --dart-define=MODEL_PATH=/absolute/path/to/model.gguf
```

Any small GGUF works, e.g.
[stories260K.gguf](https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf)
(~1 MB). The macOS app sandbox is disabled in this example's entitlements so
the test can read the model from an arbitrary path.
