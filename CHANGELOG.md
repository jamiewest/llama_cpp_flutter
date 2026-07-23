# Changelog

## 0.3.1

- Upstream: llama.cpp `b10069` → `b10091` on **both** backends. The
  native xcframework re-pins to the `b10091` release, and the vendored
  wasm is now built from wllama 3.5.1 sources with its llama.cpp
  submodule at that same tag (previously the stock npm build, whose
  embedded llama.cpp dated from June 2026). Keep loading
  `@wllama/wllama` 3.5.1 on the page — the JS pairing is unchanged.
  Verified in-browser: model download, load, and streamed generation.
- New maintenance pipeline: `tool/update_deps.sh` (native re-pin with
  ABI check), `tool/update_wllama.sh` (wasm refresh with recorded
  provenance), `tool/check_llama_ext_abi.sh` (staging-ABI tripwire), a CI
  workflow (analyze/test, ABI check, macOS link build), and a weekly
  workflow that re-pins both backends to the same llama.cpp release and
  opens a PR — including rebuilding the wllama wasm against the pinned
  tag.

## 0.3.0

- New: `generateEvents()` on `LlamaSession` — a typed
  `Stream<LlamaGenerationEvent>` (`LlamaTextEvent` /
  `LlamaCompletedEvent` / `LlamaWarningEvent`) over the same generation
  contract as `generate()`. A successful run always ends with a completed
  event carrying the engine's stats, so finish reason and token accounting
  cannot be missed. Non-breaking: implemented as an extension over the
  existing string stream.
- Fixed: the native session now reports
  `capabilities.canSetImageTokenBudget` as `false` for sessions loaded
  without a multimodal projector (previously always `true`, and calling
  `setImageTokenBudget` on a text-only session could only fail natively).

## 0.2.0

Pre-1.0 API cleanup release. Pinned upstream: llama.cpp `b10069`
(xcframework, sha256-verified); wllama wasm vendored as a Flutter asset.

### Breaking changes

- The bridge session type is renamed `LlamaSession` → `LlamaBridgeSession`
  (`bridge.dart`); the neutral `LlamaSession` in the main entrypoint is
  unchanged.
- `LlamaChatTurn` now carries a typed role (`LlamaChatRole`, including
  `tool`) and ordered content parts (`LlamaContentPart`:
  `LlamaTextPart` / `LlamaImagePart` / `LlamaAudioPart`) instead of a role
  string with separate `text`/`images`/`audio` fields. The old fields
  remain as read-only getters derived from `parts`; construct turns with
  `parts:` or `LlamaChatTurn.text(...)`. Interleaved text/media ordering
  is now preserved end to end (the web runtime sends parts in author
  order).
- The main entrypoint exports far less. Concrete chat formats, templates,
  and stream decoders moved to `package:llama_cpp_flutter/chat.dart`; GGUF
  metadata reading and artifact cache naming to
  `package:llama_cpp_flutter/gguf.dart`; the orchestration layer to
  `package:llama_cpp_flutter/orchestration.dart`. `resolveChatFormat`,
  `detectChatFormatNameForGguf`, `createLlamaChatClient`, `ModelSpec`,
  `ModelDownloader`, and the runtime API stay in the main entrypoint.
- Overriding the pinned xcframework tag
  (`LLAMA_CPP_TAG_OVERRIDE`) now requires the matching
  `LLAMA_XCFRAMEWORK_ZIP_SHA256_OVERRIDE`, or an explicit
  `LLAMA_CPP_ALLOW_UNVERIFIED=1` opt-out (development only); previously
  verification was silently skipped.

### Changed

- SDK floor lowered from Dart `^3.12.0` / Flutter `>=3.44.0` to Dart
  `^3.11.5` / Flutter `>=3.41.7` (the floor now comes from the `agents`
  dependency, not this package).
- `createLlamaRuntime()` gains `artifactCacheDirectory` for
  download-on-load, and throws a clear `UnsupportedError` on IO platforms
  without a native backend (Android, Windows, Linux) instead of failing
  later with a missing-plugin error.
- `LlamaSession` documents its concurrency and lifecycle contract (one
  generation at a time; overlapping `generate` supersedes; `cancel` and
  dispose-during-generation semantics).
- Core data classes (`ModelSpec`, `SamplingDefaults`,
  `LlamaGenerationStats`, `LlamaSessionCapabilities`, `ImageTiling`,
  `AgentSnapshot`) gain value equality and `toString`;
  `LlamaSessionCapabilities` and `ModelSpec` assert their invariants.

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
