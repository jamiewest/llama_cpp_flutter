# Changelog

## 0.1.0

Initial release.

- Cross-platform `LlamaRuntime` / `LlamaSession` API over
  [llama.cpp](https://github.com/ggml-org/llama.cpp):
  - **iOS / macOS** via a vendored Metal-enabled `llama.xcframework`
    (downloaded from official llama.cpp releases during `pod install`) with a
    typed Pigeon bridge. All native calls run on a dedicated worker isolate so
    model loading and token streaming never block the UI.
  - **Web** via `@wllama/wllama`, with the matching `wllama.wasm` shipped as a
    Flutter asset.
- `ChatClient` implementations for the
  [`agents`](https://github.com/jamiewest/agents) framework, backed by local
  GGUF models.
- Chat format layer with auto-detection from GGUF metadata: ChatML, Gemma,
  LFM2, Llama 3, Mistral, and Qwen templates, including streaming tool-call
  decoding (Hermes-style and model-specific markers).
- GGUF utilities: metadata reader, split-model (`-00001-of-0000N`) support,
  and a resumable Hugging Face model downloader.
- Memory-aware multi-agent orchestration over a single loaded model: memory
  estimation and budget planning, runtime memory monitoring, session snapshot
  store, and artifact staging.
- MTP speculative decoding on iOS/macOS via the `LlamaExtShim` staging ABI,
  with a CI ABI-drift check.
- Prompt inspector diagnostics for examining rendered prompts.
