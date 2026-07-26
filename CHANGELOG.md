# Changelog

## 0.6.2

- **Fixed: on the web, prompt budgeting used the requested context size even
  when the runtime had silently capped it.** llama.cpp's server slot (which
  wllama wraps) caps its context at the model's training context
  (`n_ctx_train`), so loading e.g. a 2048-token-trained model with
  `contextSize: 8192` really allocates 2048 tokens. The session still budgeted
  against 8192: the fail-fast guard passed prompts that could not fit, which
  then died in the wasm with `request (N tokens) exceeds the available context
  size (2048 tokens)` — an error naming a window the caller never chose — and
  `max_tokens` clamping targeted the wrong window too. The web session now
  reads `n_ctx_train` from wllama's loaded-context info and budgets against
  `min(contextSize, n_ctx_train)`, so oversized prompts fail fast with the
  actionable message and output clamping matches the real allocation.

## 0.6.1

- **Fixed: a failed model download or import on the web reported
  `TypeError: Cannot close a ERRORED writable stream` instead of the real
  cause.** When a write into OPFS rejected — quota exhausted, the origin's
  storage evicted, the connection dropped mid-transfer — the recovery path
  called `close()` on the writable, which is invalid once the stream has
  errored. The `TypeError` that raised was thrown from inside the `catch`
  block, so it displaced the `ArtifactStorageException` that would have named
  the actual failure. The stream is now discarded with `abort()`, which is
  valid in that state, and the happy-path `close()` is guarded so a late sink
  error surfaces as an `ArtifactStorageException` too. Because the `TypeError`
  also escaped before the cleanup that followed it, a failed import left its
  partial file and in-flight marker behind in OPFS rather than removing them;
  that cleanup now runs.

- **Fixed: loading a GGUF of 2 GiB or more on the web failed with
  `Unsupported operation: Uint64 accessor not supported by dart2js`.**
  Such a file cannot be staged in wasm32, so it goes through the
  client-side splitter — which read and wrote its 64-bit header fields
  with `ByteData.get/setUint64`. Those accessors throw under dart2js,
  the one platform the splitter runs on, so the path failed for every
  oversized model. The 64-bit fields are now composed from their two
  32-bit halves. GGUF header tests run under `--platform chrome` to keep
  the JS path covered.

## 0.6.0

**Breaking: `agents` is no longer a dependency.** This package now depends
only on `extensions`, so the agent framework layered on top — if any — is
entirely the host app's choice. The practical win is the SDK floor: **Dart
`^3.11.5` → `^3.9.0`, Flutter `>=3.41.7` → `>=3.35.0`**, since the old floor
came from `agents` rather than from anything in this package. `extensions`
remains a direct dependency and the `ChatClient` API is unchanged.

Two removals, both pre-1.0 and both easy to restore in your own code:

- **`AgentHandle.agent` is gone.** It built a `ChatClientAgent` from the
  handle's profile. Build it yourself from `AgentHandle.chatClient` —
  the profile is still carried verbatim on `AgentHandle.profile`
  (`id`, `name`, `description`, `instructions`, `tools`), and
  `chatClient` still returns one cached instance per handle, so an agent
  built from it keeps a single KV-cache identity:

  ```dart
  final agent = ChatClientAgent(
    handle.chatClient,
    options: ChatClientAgentOptions()
      ..id = handle.profile.id
      ..name = handle.profile.name ?? handle.profile.id
      ..description = handle.profile.description
      ..chatOptions = ChatOptions(
        instructions: handle.profile.instructions,
        tools: handle.profile.tools,
      ),
  );
  ```

- **`messagesWithRuntimeContext` is gone**, and `messagesWithInstructions`
  no longer calls it — it now only materializes `ChatOptions.instructions`
  as the leading system message. This is a **silent** behavior change for
  anyone who called `messagesWithInstructions` directly: messages are no
  longer reordered.

  What it did: text-only messages attributed to an `AIContextProvider` were
  pulled out and re-inserted as one `Runtime context:` turn immediately
  before the latest user message. Detecting them required the `agents`
  attribution API, so the behavior cannot live here anymore. It is worth
  re-implementing in whatever agent layer replaces it, because the reasoning
  is non-obvious: such context must not be merged into the system
  instructions, since that text is the head of the rendered prompt and any
  per-turn change to it invalidates the whole llama.cpp KV-cache prefix and
  forces a full re-prefill every turn. Placing it after the stable history
  keeps the prefix reusable while the model still reads it right before the
  request. See `messagesWithRuntimeContext` in 0.5.0 for the implementation.

`AgentProfile.tools` keeps its `List<AITool>?` type (`AITool` comes from
`extensions`); it is configuration only, and nothing in this package
executes it.

## 0.5.0

- New `ArtifactStore`, exported from `package:llama_cpp_flutter`: app-managed
  storage for model artifacts, with one API on every platform — a directory
  the app names on iOS/macOS, the origin-private file system on web. Build
  one with `createArtifactStore`, then `fetch` a URL or `importFile` /
  `importStream` a local file into it, and `resolve` the stored keys into the
  `localPath` / `localMmprojPath` / `localDraftPath` arguments
  `loadModel` takes. `list`, `lookup`, `totalSizeBytes`, and `delete` are
  what a model-library UI needs to show users what is on their device and
  take it back off.

  This is the alternative to an engine's opaque cache. Previously a host app
  could either hand `loadModel` a path it managed entirely itself (native
  only) or let the runtime download to somewhere it could not enumerate —
  on web, into wllama's internal URL cache.

  Downloads resume where they stopped and never report a partial file as
  complete: native transfers finalize through `ModelDownloader`'s `.part`
  rename, and web transfers carry an in-flight marker alongside the data.
  `importFile` copies by default — a macOS document picker hands back the
  user's real path — and moves only when the caller opts in with
  `moveSource`, renaming instead of duplicating a multi-gigabyte GGUF when
  the source is on the same volume. Storage failures surface as
  `ArtifactStorageException`, with `isQuotaExceeded` set when the browser is
  out of room.

  On web, `resolve` returns opaque handles rather than blob URLs, and the
  runtime dereferences them straight to their stored files. A model past the
  ~2 GiB wasm32 per-file limit is split into stageable parts from disk, as
  the runtime's own oversized-model path already did — so managed storage
  handles large models transparently instead of reading gigabytes back
  through the network stack. Speculative-decoding draft models stay
  native-only.

- The web runtime's oversized-model cache now writes through the same OPFS
  directory and names as `ArtifactStore`, so a model it downloaded on its
  own is listable and deletable through managed storage. Artifacts cached by
  earlier versions are adopted in place, not re-downloaded.

- `package:llama_cpp_flutter/gguf.dart` adds `importedArtifactFileName` and
  `artifactDisplayName` alongside `stableArtifactFileName`.

- Example: the hidden model cache is now a user-managed model library built
  on `ArtifactStore`. Models are added from the catalog, from a download
  URL, or from GGUF files on the device — each with an optional vision
  projector and, on native, a draft model — and the library shows what is
  stored, what it costs, and which model is loaded. Entries are editable
  (save, or save and reload), deletion is confirmed and removes only files
  no other entry references, and the catalog is persisted and reconciled
  with storage at startup. Image attachments are enabled only while the
  loaded model has a projector.

## 0.4.0

- New `TokenSmoother` stream transformer and the `Stream<String>.smoothed()`
  extension, exported from `package:llama_cpp_flutter/chat.dart`. Token
  streams arrive in bursts; the smoother re-paces them into a steady,
  typewriter-style grapheme stream for display.

  The release rate tracks the measured arrival rate rather than draining the
  backlog on a fixed schedule — it estimates how fast text is arriving,
  releases at that rate while holding about `window` in reserve, and uses the
  backlog only as a correction, with the rate low-pass filtered over
  `smoothing` so speed changes ramp instead of jump. The rate is fractional
  and carried across frames, so it can emit slower than one grapheme per
  frame and match a slow model instead of outrunning it. Against a 5 tok/s
  source the worst gap between graphemes is 64 ms (median 48 ms), versus
  152 ms (median 16 ms) for fixed-window draining, which empties its buffer
  in four frames and then stalls.

  Grapheme clusters (emoji ZWJ sequences) are never split, an `atomic`
  predicate keeps in-band markers from rendering half-formed, and the tail
  drains within `window` of the source ending. Opt-in by design — generation
  is not smoothed automatically, so headless and batch callers are
  unaffected. Apply it where the text is rendered:
  `session.generate(prompt).smoothed()`.
- Adds a direct dependency on `package:characters`.

## 0.3.2

- Example: the web demo now deploys to GitHub Pages on every push to
  `main`, and vendors `coi-serviceworker` to inject COOP/COEP headers on
  hosts that can't set them — keeping wllama multi-threaded on Pages.
  No library changes.

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
