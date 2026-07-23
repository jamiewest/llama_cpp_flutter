import 'dart:async';
import 'dart:typed_data';

import '../models/model_spec.dart';

/// Browser asset URL for the wllama WebAssembly runtime.
const String llamaWasmAssetPath =
    'assets/packages/llama_cpp_flutter/lib/assets/wasm/wllama.wasm';

/// Reports model load/download progress as a value from 0 to 1.
typedef LlamaLoadProgress = void Function(double progress);

/// The author of a [LlamaChatTurn].
enum LlamaChatRole {
  /// Instructions framing the conversation.
  system,

  /// The human (or calling application) side of the conversation.
  user,

  /// The model's own prior output.
  assistant,

  /// The result of a tool invocation the assistant requested.
  ///
  /// Engines whose wire format has no tool role (wllama's chat-completion
  /// API) fold tool turns into user turns.
  tool,
}

/// One piece of a turn's content, in the order the author arranged it.
///
/// Turns hold an ordered `List<LlamaContentPart>` so interleavings like
/// "compare [image 1] with [image 2]" survive intact rather than being
/// split into separate text/media buckets.
sealed class LlamaContentPart {
  const LlamaContentPart();
}

/// A run of text within a turn.
final class LlamaTextPart extends LlamaContentPart {
  /// Creates a text part.
  const LlamaTextPart(this.text);

  /// The text content.
  final String text;

  @override
  String toString() => 'LlamaTextPart(${text.length} chars)';
}

/// An encoded image (PNG/JPEG/...) within a turn.
final class LlamaImagePart extends LlamaContentPart {
  /// Creates an image part from raw encoded bytes.
  const LlamaImagePart(this.bytes, {this.mimeType});

  /// The encoded image bytes.
  final Uint8List bytes;

  /// The declared media type (e.g. `image/png`), when known. Engines
  /// generally sniff the actual format from [bytes].
  final String? mimeType;

  @override
  String toString() => 'LlamaImagePart(${bytes.length} bytes)';
}

/// An encoded audio clip (e.g. WAV) within a turn, for models with an
/// audio-capable projector (Gemma 4).
final class LlamaAudioPart extends LlamaContentPart {
  /// Creates an audio part from raw encoded bytes.
  const LlamaAudioPart(this.bytes, {this.mimeType});

  /// The encoded audio bytes.
  final Uint8List bytes;

  /// The declared media type (e.g. `audio/wav`), when known. Engines
  /// generally sniff the actual format from [bytes].
  final String? mimeType;

  @override
  String toString() => 'LlamaAudioPart(${bytes.length} bytes)';
}

/// One conversation turn in a structured, engine-neutral shape.
///
/// [LlamaSession.generate] receives the fully rendered prompt string, which
/// is all a token-in/token-out engine needs. Engines whose multimodal path is
/// message-level (the web runtime drives wllama's chat-completion API for
/// image turns) additionally need the conversation as discrete turns; this
/// type carries that view alongside the rendered prompt.
class LlamaChatTurn {
  /// Creates a turn with [role] and its ordered content [parts].
  const LlamaChatTurn({required this.role, required this.parts});

  /// Creates a text-only turn.
  LlamaChatTurn.text(this.role, String text)
    : parts = <LlamaContentPart>[LlamaTextPart(text)];

  /// The turn's author.
  final LlamaChatRole role;

  /// The turn's content, in author order.
  final List<LlamaContentPart> parts;

  /// The turn's text: every [LlamaTextPart] concatenated in order.
  String get text =>
      parts.whereType<LlamaTextPart>().map((part) => part.text).join();

  /// The bytes of every [LlamaImagePart], in order.
  List<Uint8List> get images => <Uint8List>[
    for (final part in parts)
      if (part is LlamaImagePart) part.bytes,
  ];

  /// The bytes of every [LlamaAudioPart], in order.
  List<Uint8List> get audio => <Uint8List>[
    for (final part in parts)
      if (part is LlamaAudioPart) part.bytes,
  ];

  // Deliberately no ==/hashCode: turns carry raw media buffers, and
  // byte-content equality over them is both expensive and rarely what a
  // caller comparing turns means.
  @override
  String toString() =>
      'LlamaChatTurn(role: ${role.name}, text: ${text.length} chars, '
      'images: ${images.length}, audio: ${audio.length})';
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

  @override
  bool operator ==(Object other) =>
      other is LlamaGenerationStats &&
      other.promptTokenCount == promptTokenCount &&
      other.cachedTokenCount == cachedTokenCount &&
      other.generatedTokenCount == generatedTokenCount &&
      other.finishReason == finishReason &&
      other.prefillDuration == prefillDuration &&
      other.decodeDuration == decodeDuration &&
      other.draftedTokenCount == draftedTokenCount &&
      other.acceptedTokenCount == acceptedTokenCount;

  @override
  int get hashCode => Object.hash(
    promptTokenCount,
    cachedTokenCount,
    generatedTokenCount,
    finishReason,
    prefillDuration,
    decodeDuration,
    draftedTokenCount,
    acceptedTokenCount,
  );

  @override
  String toString() =>
      'LlamaGenerationStats(prompt: $promptTokenCount, '
      'cached: $cachedTokenCount, generated: $generatedTokenCount, '
      'finishReason: ${finishReason?.name}, '
      'prefill: $prefillDuration, decode: $decodeDuration, '
      'drafted: $draftedTokenCount, accepted: $acceptedTokenCount)';
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
  }) : assert(maxSequences > 0, 'maxSequences must be at least 1');

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

  @override
  bool operator ==(Object other) =>
      other is LlamaSessionCapabilities &&
      other.canPersistState == canPersistState &&
      other.reportsStateSize == reportsStateSize &&
      other.canStashState == canStashState &&
      other.canSetImageTokenBudget == canSetImageTokenBudget &&
      other.maxSequences == maxSequences;

  @override
  int get hashCode => Object.hash(
    canPersistState,
    reportsStateSize,
    canStashState,
    canSetImageTokenBudget,
    maxSequences,
  );

  @override
  String toString() =>
      'LlamaSessionCapabilities(canPersistState: $canPersistState, '
      'reportsStateSize: $reportsStateSize, canStashState: $canStashState, '
      'canSetImageTokenBudget: $canSetImageTokenBudget, '
      'maxSequences: $maxSequences)';
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
///
/// ## Concurrency contract
///
/// A session runs at most one generation at a time, regardless of
/// [LlamaSessionCapabilities.maxSequences] — sequences are independent
/// KV caches sharing one loaded model, not parallel decode lanes. Calling
/// [generate] while a run is in flight supersedes it: the engine cancels
/// the current run and starts the newest call once cancellation completes
/// (an intermediate superseded call's stream closes without emitting
/// text). Callers that care about the superseded run's output should
/// [cancel] and drain its stream before starting the next run.
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
  ///
  /// [sequenceId] selects which KV-cache sequence the run extends and must
  /// be less than [LlamaSessionCapabilities.maxSequences]. Only one run
  /// executes at a time even with multiple sequences; see the class docs
  /// for how an overlapping call is handled. Cancelling the returned
  /// stream's subscription cancels the run, same as [cancel].
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

  /// Requests cancellation of the in-flight generation, if any.
  ///
  /// The generation stream then completes promptly without an error, and
  /// when the engine reports stats the run's
  /// [LlamaGenerationStats.finishReason] is
  /// [LlamaFinishReason.cancelled]. Text already streamed remains valid.
  /// A no-op when nothing is generating.
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
  ///
  /// Safe to call during generation: the in-flight run is abandoned and its
  /// stream closes, though a stats callback may not be delivered. The
  /// session, its streams, and any stashed state must not be used
  /// afterwards.
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

/// One event in a structured generation stream — see
/// [LlamaSessionEvents.generateEvents].
///
/// The hierarchy is sealed, so exhaustive switches compile today; note
/// that [LlamaWarningEvent] is reserved and nothing emits it yet.
sealed class LlamaGenerationEvent {
  const LlamaGenerationEvent();
}

/// A decoded piece of generated text.
final class LlamaTextEvent extends LlamaGenerationEvent {
  /// Creates a text event.
  const LlamaTextEvent(this.text);

  /// The decoded text piece, with stop sequences already removed.
  final String text;

  @override
  String toString() => 'LlamaTextEvent(${text.length} chars)';
}

/// The run ended; always the final event of a successful stream.
final class LlamaCompletedEvent extends LlamaGenerationEvent {
  /// Creates a completed event.
  const LlamaCompletedEvent(this.stats);

  /// The run's token accounting, or null when the engine reported none.
  ///
  /// When present, `stats.finishReason` says why the run stopped.
  final LlamaGenerationStats? stats;

  @override
  String toString() => 'LlamaCompletedEvent($stats)';
}

/// A non-fatal condition worth surfacing to the caller.
///
/// Reserved for future engines: nothing emits it today, but it is part of
/// the sealed vocabulary so adding an emitter later is not a breaking
/// change for exhaustive switches.
final class LlamaWarningEvent extends LlamaGenerationEvent {
  /// Creates a warning event.
  const LlamaWarningEvent(this.message);

  /// Human-readable description of the condition.
  final String message;

  @override
  String toString() => 'LlamaWarningEvent($message)';
}

/// Structured event streaming over [LlamaSession.generate].
extension LlamaSessionEvents on LlamaSession {
  /// Generates text for [prompt] as a stream of typed events instead of
  /// raw strings.
  ///
  /// Text arrives as [LlamaTextEvent]s. A successful run always ends with
  /// a [LlamaCompletedEvent] carrying the engine's stats (null when the
  /// engine reports none), so completion metadata cannot be missed — the
  /// events equivalent of [LlamaSession.generate]'s `onStats` callback. A
  /// failed run surfaces the failure as a stream error instead of a
  /// completed event.
  ///
  /// Everything else matches [LlamaSession.generate]: same parameters,
  /// same one-generation-at-a-time contract, and cancelling the
  /// subscription cancels the run.
  Stream<LlamaGenerationEvent> generateEvents(
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
  }) {
    LlamaGenerationStats? stats;
    var errored = false;
    StreamSubscription<String>? subscription;
    late final StreamController<LlamaGenerationEvent> controller;
    controller = StreamController<LlamaGenerationEvent>(
      onListen: () {
        subscription = generate(
          prompt,
          maxTokens: maxTokens,
          temperature: temperature,
          topK: topK,
          topP: topP,
          seed: seed,
          stopSequences: stopSequences,
          media: media,
          turns: turns,
          sequenceId: sequenceId,
          // Engines report stats before closing the token stream, so the
          // capture is complete by the time onDone runs.
          onStats: (s) => stats = s,
        ).listen(
          (text) => controller.add(LlamaTextEvent(text)),
          onError: (Object error, StackTrace stackTrace) {
            errored = true;
            controller.addError(error, stackTrace);
          },
          onDone: () {
            if (!errored) controller.add(LlamaCompletedEvent(stats));
            controller.close();
          },
        );
      },
      onPause: () => subscription?.pause(),
      onResume: () => subscription?.resume(),
      onCancel: () => subscription?.cancel(),
    );
    return controller.stream;
  }
}
