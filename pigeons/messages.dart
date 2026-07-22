import 'package:pigeon/pigeon.dart';

@ConfigurePigeon(
  PigeonOptions(
    dartOut: 'lib/src/messages.g.dart',
    dartOptions: DartOptions(),
    swiftOut: 'darwin/Classes/Messages.g.swift',
    swiftOptions: SwiftOptions(errorClassName: 'LlamaError'),
    dartPackageName: 'llama_flutter',
  ),
)
/// Parameters for an optional drafter (assistant) model that enables
/// speculative decoding.
///
/// The drafter proposes tokens cheaply and the main model verifies them, so
/// the output distribution is identical to decoding with the main model
/// alone — just faster when the drafter's guesses are accepted. Any GGUF
/// whose vocabulary matches the main model works (e.g. a Gemma 4 MTP
/// `*-assistant` drafter); this is not tied to one architecture.
class DraftModelOptions {
  DraftModelOptions({
    required this.modelPath,
    required this.gpuLayers,
    required this.maxDraftTokens,
  });

  /// Absolute filesystem path to the drafter `.gguf`.
  String modelPath;

  /// Number of drafter layers to offload to the GPU (`n_gpu_layers`).
  ///
  /// `0` forces CPU-only inference (e.g. the iOS Simulator).
  int gpuLayers;

  /// Maximum tokens drafted per verification step.
  int maxDraftTokens;
}

/// Parameters for loading a GGUF model into a native llama.cpp session.
class ModelLoadRequest {
  ModelLoadRequest({
    required this.modelPath,
    required this.contextSize,
    required this.gpuLayers,
    this.mmprojPath,
    this.imageTokenBudget,
    this.draftModel,
  });

  /// Absolute filesystem path to the `.gguf` model.
  String modelPath;

  /// Context window size (`n_ctx`).
  int contextSize;

  /// Number of layers to offload to the GPU (`n_gpu_layers`).
  ///
  /// `0` forces CPU-only inference (e.g. the iOS Simulator).
  int gpuLayers;

  /// Absolute path to the multimodal projector (`mmproj`) `.gguf`.
  ///
  /// When non-null the session also stands up an `mtmd` context, enabling
  /// image input via [GenerationRequest.images]. Null keeps the session
  /// text-only.
  String? mmprojPath;

  /// Vision-encoder token budget per image (mtmd `image_min_tokens` and
  /// `image_max_tokens`, both set to this value).
  ///
  /// Only meaningful with [mmprojPath], for vision models with dynamic
  /// resolution. Gemma supports 70, 140, 280, 560, or 1120; higher budgets
  /// trade prefill time for fidelity (1120 suits OCR / small text). Null
  /// keeps the model's metadata default. The physical batch is floored at
  /// this value so the encoder's non-causal image chunk fits one ubatch.
  int? imageTokenBudget;

  /// Optional drafter model enabling speculative decoding.
  ///
  /// Null disables speculation; the session then behaves exactly as a
  /// single-model session.
  DraftModelOptions? draftModel;

  /// Number of independent KV-cache sequences the context supports
  /// (`n_seq_max`). Null or `1` keeps the classic single-sequence session.
  ///
  /// With N sequences, [contextSize] is the TOTAL KV budget and each
  /// sequence gets `contextSize / N` positions. Generations target a
  /// sequence via [GenerationRequest.sequenceId]. Speculative decoding
  /// (draft models) only runs on sequence 0.
  int? maxSequences;
}

/// Parameters for a single generation run against a loaded session.
class GenerationRequest {
  GenerationRequest({
    required this.sessionId,
    required this.prompt,
    required this.maxTokens,
    required this.temperature,
    this.topK,
    this.topP,
    this.seed,
    required this.stopSequences,
    this.images,
  });

  /// Session returned by [LlamaHostApi.loadModel].
  int sessionId;

  /// The fully formatted prompt to feed the model.
  String prompt;

  /// Hard cap on the number of tokens to generate.
  int maxTokens;

  /// Sampling temperature; `0` is greedy.
  double temperature;

  /// Top-k cutoff applied before temperature sampling. Null or `<= 0`
  /// disables the top-k stage.
  int? topK;

  /// Nucleus (top-p) cutoff applied before temperature sampling. Null or
  /// `>= 1.0` disables the top-p stage.
  double? topP;

  /// Sampler seed for reproducible generation. Null draws a random seed.
  int? seed;

  /// Strings that, when produced in the decoded output, halt generation.
  ///
  /// The matched stop sequence is removed from the streamed text and is not
  /// emitted. Matching runs against the decoded text (with special tokens
  /// rendered), so control markers such as Gemma's `<turn|>` work directly.
  /// Empty disables stop-sequence handling.
  List<String> stopSequences;

  /// Encoded image bytes (PNG/JPEG) to feed alongside [prompt].
  ///
  /// Each entry corresponds, in order, to one media marker in [prompt]. Only
  /// honoured when the session was loaded with a
  /// [ModelLoadRequest.mmprojPath]; otherwise ignored. Null or empty runs a
  /// text-only generation.
  List<Uint8List>? images;

  /// KV-cache sequence this generation runs in. Null means sequence 0.
  ///
  /// Only meaningful when the session was loaded with
  /// [ModelLoadRequest.maxSequences] > 1; each sequence keeps its own
  /// prompt-prefix cache. Speculative decoding is bypassed for non-zero
  /// sequences.
  int? sequenceId;
}

/// Why a generation run stopped.
enum FinishReason {
  /// The model produced an end-of-generation token.
  eogToken,

  /// The run reached [GenerationRequest.maxTokens].
  maxTokens,

  /// A [GenerationRequest.stopSequences] entry matched.
  stopSequence,

  /// The context window filled before the run ended naturally.
  contextFull,

  /// The run was cancelled via [LlamaHostApi.cancelGeneration].
  cancelled,

  /// The native runtime ended the run early after an unrecoverable internal
  /// condition (e.g. a failed speculative rollback). Text streamed before
  /// this point is valid.
  aborted,
}

/// Token accounting and timing for one generation run, delivered on the
/// final [TokenEvent].
class GenerationStats {
  GenerationStats({
    required this.promptTokenCount,
    required this.cachedTokenCount,
    required this.generatedTokenCount,
    required this.finishReason,
    required this.prefillMicroseconds,
    required this.decodeMicroseconds,
    this.draftedTokenCount,
    this.acceptedTokenCount,
  });

  /// Prompt tokens fed to the model (including the reused prefix).
  int promptTokenCount;

  /// Prompt tokens served from the reused KV-cache prefix.
  int cachedTokenCount;

  /// Tokens generated.
  int generatedTokenCount;

  /// Why the run stopped.
  FinishReason finishReason;

  /// Wall-clock time spent ingesting the prompt, in microseconds.
  int prefillMicroseconds;

  /// Wall-clock time spent in the decode loop, in microseconds.
  int decodeMicroseconds;

  /// Tokens proposed by the draft model across the run; null when the
  /// session runs without speculative decoding.
  int? draftedTokenCount;

  /// Drafted tokens the main model accepted; null when the session runs
  /// without speculative decoding.
  int? acceptedTokenCount;
}

/// A single streamed generation event.
///
/// Emitted repeatedly with [text] set while tokens are produced, then once
/// more with [done] true. On failure [error] is set and [done] is true.
class TokenEvent {
  TokenEvent({
    required this.sessionId,
    this.text,
    required this.done,
    this.error,
    this.stats,
  });

  int sessionId;
  String? text;
  bool done;
  String? error;

  /// Accounting for the completed run, set on the `done` event (absent when
  /// [error] is set).
  GenerationStats? stats;
}

/// Control surface implemented natively (Swift) and called from Dart.
@HostApi()
abstract class LlamaHostApi {
  /// Loads the model and returns an opaque session id.
  @async
  int loadModel(ModelLoadRequest request);

  /// Starts generation; tokens arrive on the [LlamaTokenStream] event channel.
  void startGeneration(GenerationRequest request);

  /// Requests cancellation of an in-flight generation for [sessionId].
  void cancelGeneration(int sessionId);

  /// Saves one sequence's KV-cache state (and its token ledger) to [path].
  ///
  /// [sequenceId] null means sequence 0. Returns the number of tokens
  /// covered by the snapshot: positive on success, `0` when the sequence
  /// holds no reusable cache (no file is written). Runs on the session's
  /// serial queue, so it cannot interleave with generation.
  @async
  int saveSessionState(int sessionId, String path, int? sequenceId);

  /// Restores KV-cache state previously written by [saveSessionState] into
  /// [sequenceId] (null = sequence 0).
  ///
  /// Replaces that sequence's current cache contents. Returns the number of
  /// tokens restored. Fails (throws) when the file is missing, corrupt, or
  /// was written by an incompatible model — the sequence is left with an
  /// empty cache in that case, so the next generation simply prefills from
  /// scratch.
  @async
  int loadSessionState(int sessionId, String path, int? sequenceId);

  /// Measures the byte size of one sequence's current state (KV cache plus
  /// bookkeeping) without writing it anywhere.
  ///
  /// This is the exact size [saveSessionState] would write for the current
  /// cache, so orchestrating callers can budget memory or disk before
  /// snapshotting. Runs on the session's serial queue, so it cannot
  /// interleave with generation.
  @async
  int getSessionStateSize(int sessionId, int? sequenceId);

  /// Copies one sequence's KV state into an in-memory, native-side stash
  /// under [key], without touching the sequence itself.
  ///
  /// The blob never crosses the platform channel — restoring later is a
  /// RAM-to-RAM copy, far faster than the file path. The stash lives and
  /// dies with the session (disposed with it). An existing entry under
  /// [key] is replaced. Returns zero counts when the sequence holds no
  /// reusable cache (nothing is stashed).
  @async
  StashResult stashSessionState(int sessionId, int sequenceId, String key);

  /// Restores stashed state saved under [key] into [sequenceId], replacing
  /// that sequence's current cache. The stash entry is kept (drop it
  /// explicitly with [dropStashedState]).
  ///
  /// Returns the number of tokens restored; fails (throws) when [key] has
  /// no entry or the copy fails, leaving the sequence's cache empty.
  @async
  int restoreStashedState(int sessionId, int sequenceId, String key);

  /// Removes the stash entry under [key], freeing its memory. Returns the
  /// number of bytes freed (`0` when no entry existed).
  @async
  int dropStashedState(int sessionId, String key);

  /// Erases one sequence's KV cache and prompt ledger. Queued behind any
  /// in-flight generation like every other session operation.
  @async
  void clearSequence(int sessionId, int sequenceId);

  /// Changes the per-image vision token budget for subsequent image turns
  /// (see [ModelLoadRequest.imageTokenBudget]); null restores the model's
  /// metadata default.
  ///
  /// The mtmd context is freed so the next image turn re-creates it with
  /// the new budget — a projector reload from disk, while the LLM context
  /// and KV caches stay untouched. Fails when the budget exceeds the
  /// micro-batch size fixed at load (`n_ubatch`); raising it beyond that
  /// requires reloading the model.
  @async
  void setImageTokenBudget(int sessionId, int? imageTokenBudget);

  /// Frees the model/context associated with [sessionId].
  void disposeSession(int sessionId);
}

/// Result of [LlamaHostApi.stashSessionState].
class StashResult {
  StashResult({required this.tokens, required this.bytes});

  /// Tokens the stashed state covers (`0` = nothing was stashed).
  int tokens;

  /// Size of the stashed blob in bytes (counts against process memory
  /// until dropped or the session is disposed).
  int bytes;
}

/// Token stream delivered from native to Dart via an `EventChannel`.
///
/// A single stream multiplexes every session's tokens; each [TokenEvent]
/// carries its [TokenEvent.sessionId] so listeners can filter per session.
@EventChannelApi()
abstract class LlamaTokenStream {
  TokenEvent streamTokens();
}
