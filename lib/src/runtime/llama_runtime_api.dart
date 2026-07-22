import 'dart:typed_data';

import '../models/model_spec.dart';

/// Browser asset URL for the wllama WebAssembly runtime.
const String llamaWasmAssetPath =
    'assets/packages/llama_flutter/lib/assets/wasm/wllama.wasm';

/// Reports model load/download progress as a value from 0 to 1.
typedef LlamaLoadProgress = void Function(double progress);

/// One conversation turn in a structured, engine-neutral shape.
///
/// [LlamaSession.generate] receives the fully rendered prompt string, which
/// is all a token-in/token-out engine needs. Engines whose multimodal path is
/// message-level (the web runtime drives wllama's chat-completion API for
/// image turns) additionally need the conversation as discrete turns; this
/// type carries that view alongside the rendered prompt.
class LlamaChatTurn {
  /// Creates a turn with [role], its concatenated [text], and any [images] or
  /// [audio].
  const LlamaChatTurn({
    required this.role,
    required this.text,
    this.images = const <Uint8List>[],
    this.audio = const <Uint8List>[],
  });

  /// The chat role: `'system'`, `'user'`, or `'assistant'`.
  final String role;

  /// The turn's text content.
  final String text;

  /// Raw encoded image bytes attached to this turn.
  ///
  /// Kept separate from [audio] because the message-level multimodal path
  /// (wllama's chat-completion API) labels each content part by kind
  /// (`{type: 'image'}` vs `{type: 'audio'}`).
  final List<Uint8List> images;

  /// Raw encoded audio bytes attached to this turn (e.g. WAV), for models with
  /// an audio-capable projector (Gemma 4). See [images] for why kinds are split.
  final List<Uint8List> audio;
}

/// Why a generation run stopped, engine-neutral.
enum LlamaFinishReason {
  /// The model produced an end-of-generation token.
  eogToken,

  /// The run reached its `maxTokens` cap.
  maxTokens,

  /// A stop sequence matched.
  stopSequence,

  /// The context window filled before the run ended naturally.
  contextFull,

  /// The run was cancelled.
  cancelled,

  /// The engine ended the run early after an unrecoverable internal
  /// condition; text streamed before this point is valid.
  aborted,
}

/// Token accounting for one completed generation run, engine-neutral.
///
/// The token counts are always present. The remaining fields depend on what
/// the engine observes: the native runtime reports all of them; the web
/// runtime reports [finishReason] (inferred) but no timings or draft counts.
class LlamaGenerationStats {
  /// Creates a [LlamaGenerationStats].
  const LlamaGenerationStats({
    required this.promptTokenCount,
    required this.cachedTokenCount,
    required this.generatedTokenCount,
    this.finishReason,
    this.prefillDuration,
    this.decodeDuration,
    this.draftedTokenCount,
    this.acceptedTokenCount,
  });

  /// Prompt tokens fed to the model (including any reused cache prefix).
  final int promptTokenCount;

  /// Prompt tokens served from a reused KV-cache prefix.
  final int cachedTokenCount;

  /// Tokens generated.
  final int generatedTokenCount;

  /// Why the run stopped; null when the engine does not report it.
  final LlamaFinishReason? finishReason;

  /// Wall-clock time spent ingesting the prompt; null when the engine does
  /// not measure it.
  final Duration? prefillDuration;

  /// Wall-clock time spent generating tokens; null when the engine does not
  /// measure it.
  final Duration? decodeDuration;

  /// Tokens proposed by a draft model; null when the run did not use
  /// speculative decoding.
  final int? draftedTokenCount;

  /// Drafted tokens the main model accepted; null when the run did not use
  /// speculative decoding.
  final int? acceptedTokenCount;

  /// Prompt-ingestion speed over the tokens that were actually decoded
  /// (prompt minus cached), in tokens per second; null when
  /// [prefillDuration] is unknown.
  double? get prefillTokensPerSecond =>
      _rate(promptTokenCount - cachedTokenCount, prefillDuration);

  /// Generation speed in tokens per second; null when [decodeDuration] is
  /// unknown.
  double? get decodeTokensPerSecond =>
      _rate(generatedTokenCount, decodeDuration);

  static double? _rate(int tokens, Duration? elapsed) {
    if (elapsed == null) return null;
    final seconds = elapsed.inMicroseconds / Duration.microsecondsPerSecond;
    return seconds > 0 ? tokens / seconds : 0;
  }
}

/// Invoked once per generation with the run's token accounting.
typedef LlamaStatsCallback = void Function(LlamaGenerationStats stats);

/// What a [LlamaSession] implementation can actually do, beyond generating.
///
/// Callers that orchestrate sessions (e.g. swapping per-agent KV state)
/// check this instead of special-casing platforms.
class LlamaSessionCapabilities {
  /// Creates a capability set.
  const LlamaSessionCapabilities({
    required this.canPersistState,
    required this.reportsStateSize,
    this.canStashState = false,
    this.canSetImageTokenBudget = false,
    this.maxSequences = 1,
  });

  /// Whether [LlamaSession.saveState] writes a restorable snapshot.
  ///
  /// False means [LlamaSession.saveState] always returns `0` and
  /// [LlamaSession.loadState] throws [UnsupportedError].
  final bool canPersistState;

  /// Whether [LlamaSession.stateSizeBytes] returns a real measurement.
  ///
  /// False means [LlamaSession.stateSizeBytes] always returns `0`.
  final bool reportsStateSize;

  /// Whether [LlamaSession.stashState] keeps a restorable in-memory copy.
  ///
  /// False means [LlamaSession.stashState] always returns zero counts and
  /// [LlamaSession.restoreStashedState] throws [UnsupportedError].
  final bool canStashState;

  /// Whether [LlamaSession.setImageTokenBudget] changes the vision
  /// encoder's per-image token budget at runtime.
  ///
  /// False means [LlamaSession.setImageTokenBudget] throws
  /// [UnsupportedError].
  final bool canSetImageTokenBudget;

  /// Number of independent KV-cache sequences this session supports.
  ///
  /// `1` means the classic single-conversation session; sequence arguments
  /// other than `0` are then invalid.
  final int maxSequences;
}

/// Outcome of [LlamaSession.stashState]: what the in-memory copy covers.
///
/// [bytes] counts against process memory until the stash entry is dropped
/// or the session is disposed.
typedef LlamaStashResult = ({int tokens, int bytes});

/// A loaded llama-family model session.
///
/// Implementations must yield text with stop sequences removed and terminate at
/// the first stop sequence.
abstract interface class LlamaSession {
  /// Generates text for [prompt], yielding decoded pieces as they arrive.
  ///
  /// [media] carries encoded media bytes (image or audio) referenced by the
  /// markers embedded in [prompt], in marker order; mtmd auto-detects each
  /// blob's kind. Used by runtimes that consume the rendered [prompt] directly.
  ///
  /// [turns] is an optional structured view of the same conversation, used by
  /// runtimes whose multimodal path is message-level; runtimes that consume
  /// the rendered [prompt] directly ignore it.
  ///
  /// [onStats] is invoked once with the run's token accounting when the
  /// engine reports it; engines that cannot count tokens may never call it.
  Stream<String> generate(
    String prompt, {
    int maxTokens = 256,
    double temperature = 0.8,
    int? topK,
    double? topP,
    int? seed,
    List<String> stopSequences = const <String>[],
    List<Uint8List>? media,
    List<LlamaChatTurn>? turns,
    int sequenceId = 0,
    LlamaStatsCallback? onStats,
  });

  /// Requests cancellation of any in-flight generation.
  Future<void> cancel();

  /// What this session can do beyond generating (state persistence, state
  /// size reporting).
  LlamaSessionCapabilities get capabilities;

  /// Saves one sequence's KV-cache state to [path].
  ///
  /// Returns the number of tokens the snapshot covers — positive on
  /// success, `0` when nothing is cached (no file is written) or when
  /// [capabilities] reports `canPersistState` false.
  Future<int> saveState(String path, {int sequenceId = 0});

  /// Restores KV-cache state previously written by [saveState] into
  /// [sequenceId], replacing that sequence's current cache.
  ///
  /// Returns the number of tokens restored. Throws when the file is
  /// missing, corrupt, or from an incompatible model — the sequence is
  /// left with an empty cache, so the next generation prefills from
  /// scratch. Throws [UnsupportedError] when [capabilities] reports
  /// `canPersistState` false.
  Future<int> loadState(String path, {int sequenceId = 0});

  /// Measures the byte size of one sequence's current state (KV cache plus
  /// bookkeeping) without writing it anywhere.
  ///
  /// Returns `0` when [capabilities] reports `reportsStateSize` false.
  Future<int> stateSizeBytes({int sequenceId = 0});

  /// Copies one sequence's KV state into an engine-side, in-memory stash
  /// under [key] — the fast swap path (no disk, no serialization).
  ///
  /// Returns the stash's token and byte counts; `(tokens: 0, bytes: 0)`
  /// when the sequence held nothing reusable or [capabilities] reports
  /// `canStashState` false. The stash lives and dies with the session.
  Future<LlamaStashResult> stashState(String key, {int sequenceId = 0});

  /// Restores stashed state saved under [key] into [sequenceId], replacing
  /// that sequence's cache. The entry is kept until [dropStashedState].
  ///
  /// Returns the tokens restored. Throws when [key] has no entry or the
  /// copy fails (the sequence is then left empty), and
  /// [UnsupportedError] when [capabilities] reports `canStashState` false.
  Future<int> restoreStashedState(String key, {int sequenceId = 0});

  /// Drops the stash entry under [key]; returns the bytes freed (`0` when
  /// no entry existed or stashing is unsupported).
  Future<int> dropStashedState(String key);

  /// Erases one sequence's KV cache and prompt ledger. A no-op on engines
  /// without sequence support.
  Future<void> clearSequence(int sequenceId);

  /// Changes the vision encoder's per-image token budget for subsequent
  /// image turns; null restores the model's metadata default.
  ///
  /// Cheap relative to a model reload — the engine rebuilds only its
  /// multimodal projector state on the next image turn, keeping the KV
  /// caches — but bounded by the batch size fixed at load: raising the
  /// budget beyond it throws, and needs a reload with
  /// `ModelSpec.imageTokenBudget` instead. Throws [UnsupportedError] when
  /// [capabilities] reports `canSetImageTokenBudget` false.
  Future<void> setImageTokenBudget(int? imageTokenBudget);

  /// Releases resources held by this session.
  Future<void> dispose();
}

/// Cross-platform loader for local llama-family model sessions.
abstract interface class LlamaRuntime {
  /// Whether the engine can run inference on multiple threads.
  ///
  /// Native runtimes always can. The web runtime can only when the page is
  /// cross-origin isolated (served with `Cross-Origin-Opener-Policy` and
  /// `Cross-Origin-Embedder-Policy` headers), because wasm threads need
  /// `SharedArrayBuffer`. Without isolation, wllama silently falls back to a
  /// single thread and generation becomes slow enough to look like a hang for
  /// multi-billion-parameter models — surface this to the user.
  bool get supportsMultiThreading;

  /// Loads [spec], optionally using already-resolved local artifacts.
  ///
  /// [localPath], [localMmprojPath], and [localDraftPath] are filesystem
  /// paths on native platforms and blob object URLs on the web. When absent,
  /// runtimes fall back to the spec's artifact URLs where supported.
  Future<LlamaSession> loadModel(
    ModelSpec spec, {
    String? localPath,
    String? localMmprojPath,
    String? localDraftPath,
    LlamaLoadProgress? onProgress,
  });
}
