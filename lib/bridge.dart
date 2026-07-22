/// The low-level iOS/macOS plugin bridge, backed by llama.cpp.
///
/// All native work runs through a Pigeon bridge that is driven from a dedicated
/// worker isolate, so model loading and token streaming never block the UI.
///
/// Most consumers want the neutral, cross-platform API in
/// `package:llama_cpp_flutter/llama_cpp_flutter.dart` instead; this entrypoint is the
/// raw bridge it wraps (kept separate because both layers define a
/// `LlamaSession` type).
library;

import 'dart:typed_data';

import 'package:extensions/logging.dart';

import 'src/llama_isolate.dart';
import 'src/messages.g.dart';

export 'src/llama_isolate.dart'
    show LlamaException, LlamaFinishReason, LlamaGenerationStats;

/// A loaded model session, returned by [LlamaCppFlutter.loadModel].
class LlamaSession {
  LlamaSession._(this._isolate, this.id, this.maxSequences);

  final LlamaIsolate _isolate;

  /// Native session identifier.
  final int id;

  /// Number of independent KV-cache sequences this session was loaded
  /// with (`n_seq_max`). `1` is the classic single-conversation session.
  final int maxSequences;

  /// Generates text for [prompt], yielding decoded tokens as they arrive.
  ///
  /// Generation halts when any string in [stopSequences] appears in the
  /// output; the matched sequence is stripped from the stream. Cancel by
  /// cancelling the returned stream's subscription.
  ///
  /// Pass encoded image bytes (PNG/JPEG) in [images] to feed a vision model;
  /// each entry corresponds, in order, to one media marker in [prompt]. Images
  /// require the session to have been loaded with a `mmprojPath`.
  ///
  /// [topK] and [topP] add the corresponding sampler stages ahead of
  /// temperature sampling; null leaves each stage out. [seed] makes sampling
  /// reproducible; null draws a random seed.
  ///
  /// [onStats] is invoked once when generation completes, with the run's
  /// prompt/cached/generated token counts, the reason it stopped, and its
  /// prefill/decode timings.
  Stream<String> generate(
    String prompt, {
    int maxTokens = 256,
    double temperature = 0.8,
    int? topK,
    double? topP,
    int? seed,
    List<String> stopSequences = const <String>[],
    List<Uint8List>? images,
    int sequenceId = 0,
    void Function(LlamaGenerationStats)? onStats,
  }) {
    return _isolate.generate(
      GenerationRequest(
        sessionId: id,
        prompt: prompt,
        maxTokens: maxTokens,
        temperature: temperature,
        topK: topK,
        topP: topP,
        seed: seed,
        stopSequences: stopSequences,
        images: images,
        sequenceId: sequenceId,
      ),
      onStats: onStats,
    );
  }

  /// Requests cancellation of any in-flight generation for this session.
  Future<void> cancel() => _isolate.cancel(id);

  /// Saves the session's KV-cache state to [path].
  ///
  /// Resolves to the number of tokens the snapshot covers — positive on
  /// success, `0` when nothing is cached (no file is written). Restoring the
  /// file later with [loadState] lets the next [generate] reuse the saved
  /// prompt prefix instead of re-prefilling it (the prompt cache otherwise
  /// only survives while the same conversation stays the session's most
  /// recent one). Throws [LlamaException] on failure.
  Future<int> saveState(String path, {int sequenceId = 0}) =>
      _isolate.sessionState(id, path, save: true, sequenceId: sequenceId);

  /// Restores KV-cache state previously written by [saveState] into
  /// [sequenceId], replacing that sequence's current cache.
  ///
  /// Resolves to the number of tokens restored. Throws [LlamaException] when
  /// the file is missing, corrupt, or from an incompatible model; the
  /// sequence is left with an empty cache in that case, so the next
  /// generation prefills from scratch.
  Future<int> loadState(String path, {int sequenceId = 0}) =>
      _isolate.sessionState(id, path, save: false, sequenceId: sequenceId);

  /// Measures the byte size of one sequence's current state (KV cache plus
  /// bookkeeping) without writing it anywhere.
  ///
  /// This is the exact size [saveState] would write for the current cache,
  /// so callers can budget memory or disk before snapshotting.
  Future<int> stateSizeBytes({int sequenceId = 0}) =>
      _isolate.stateSize(id, sequenceId: sequenceId);

  /// Copies one sequence's KV state into a native, in-memory stash under
  /// [key] — the fast path for swapping conversations: no disk write and
  /// no bytes crossing the platform channel.
  ///
  /// Resolves to the stash's token and byte counts; `(0, 0)` when the
  /// sequence held nothing reusable (nothing is stashed). The stash lives
  /// and dies with the session. An existing entry under [key] is replaced.
  Future<({int tokens, int bytes})> stashState(
    String key, {
    int sequenceId = 0,
  }) => _isolate.stashState(id, sequenceId, key);

  /// Restores stashed state saved under [key] into [sequenceId], replacing
  /// that sequence's cache. The stash entry is kept until
  /// [dropStashedState].
  ///
  /// Resolves to the tokens restored; throws [LlamaException] when [key]
  /// has no entry or the copy fails (the sequence is then left empty).
  Future<int> restoreStashedState(String key, {int sequenceId = 0}) =>
      _isolate.restoreStashedState(id, sequenceId, key);

  /// Drops the stash entry under [key]; resolves to the bytes freed (`0`
  /// when no entry existed).
  Future<int> dropStashedState(String key) =>
      _isolate.dropStashedState(id, key);

  /// Erases one sequence's KV cache and prompt ledger.
  Future<void> clearSequence(int sequenceId) =>
      _isolate.clearSequence(id, sequenceId);

  /// Changes the per-image vision token budget for subsequent image turns
  /// (see [LlamaCppFlutter.loadModel]'s `imageTokenBudget`); null restores the
  /// model's metadata default.
  ///
  /// Cheap relative to a model reload: the next image turn re-creates the
  /// mtmd context (a projector reload from disk) while the LLM context and
  /// every KV cache stay untouched. Throws [LlamaException] when the budget
  /// exceeds the micro-batch size fixed at load — raising it beyond that
  /// requires reloading the model with the larger `imageTokenBudget`.
  Future<void> setImageTokenBudget(int? imageTokenBudget) =>
      _isolate.setImageTokenBudget(id, imageTokenBudget);

  /// Frees the native model and context for this session.
  Future<void> dispose() => _isolate.disposeSession(id);
}

/// Entry point for llama.cpp inference.
///
/// Create one instance, [loadModel] one or more GGUF files, then [generate]
/// against the returned [LlamaSession]. Call [shutdown] when done to tear down
/// the worker isolate.
class LlamaCppFlutter {
  /// Creates the entry point. [loggerFactory] supplies loggers for model
  /// loading and generation diagnostics; null logs nothing.
  LlamaCppFlutter({LoggerFactory? loggerFactory})
    : _isolate = LlamaIsolate(loggerFactory: loggerFactory),
      _logger = (loggerFactory ?? NullLoggerFactory.instance).createLogger(
        'LlamaCppFlutter',
      );

  final LlamaIsolate _isolate;
  final Logger _logger;

  /// Loads a GGUF model from an absolute filesystem [path].
  ///
  /// [contextSize] is the token context window. [gpuLayers] controls Metal
  /// offload; pass `0` to force CPU (e.g. the iOS Simulator).
  ///
  /// Pass [mmprojPath] (an absolute path to a multimodal projector `.gguf`) to
  /// enable image input via [LlamaSession.generate]'s `images` argument.
  /// [imageTokenBudget] then sets the vision encoder's per-image token budget
  /// (mtmd `image_min_tokens`/`image_max_tokens`) for vision models with
  /// dynamic resolution — Gemma supports 70, 140, 280, 560, or 1120, where
  /// higher budgets trade prefill time for fidelity (1120 suits OCR and small
  /// text). Null keeps the model's metadata default.
  ///
  /// Pass [draftModelPath] (an absolute path to a drafter `.gguf` whose
  /// vocabulary matches the main model, e.g. a Gemma 4 MTP assistant) to
  /// enable speculative decoding: the drafter proposes up to [maxDraftTokens]
  /// tokens per step and the main model verifies them, leaving output
  /// identical but faster. [draftGpuLayers] controls the drafter's Metal
  /// offload independently of [gpuLayers]. Null keeps the session
  /// single-model. The drafter is best-effort: if it fails to load or its
  /// vocabulary is incompatible, the reason is logged natively and the
  /// session loads without speculation rather than failing.
  Future<LlamaSession> loadModel(
    String path, {
    int contextSize = 4096,
    int gpuLayers = 999,
    String? mmprojPath,
    int? imageTokenBudget,
    String? draftModelPath,
    int draftGpuLayers = 999,
    int maxDraftTokens = 3,
    int maxSequences = 1,
  }) async {
    // Idempotent and safe under concurrent calls; see LlamaIsolate.start.
    await _isolate.start();
    _logger.logInformation(
      'Loading model $path (contextSize: $contextSize, '
      'gpuLayers: $gpuLayers, maxSequences: $maxSequences'
      '${mmprojPath == null ? '' : ', mmproj: $mmprojPath'}'
      '${draftModelPath == null ? '' : ', draft: $draftModelPath'}).',
    );
    final id = await _isolate.loadModel(
      ModelLoadRequest(
        modelPath: path,
        contextSize: contextSize,
        gpuLayers: gpuLayers,
        mmprojPath: mmprojPath,
        imageTokenBudget: imageTokenBudget,
        draftModel: draftModelPath == null
            ? null
            : DraftModelOptions(
                modelPath: draftModelPath,
                gpuLayers: draftGpuLayers,
                maxDraftTokens: maxDraftTokens,
              ),
        maxSequences: maxSequences,
      ),
    );
    return LlamaSession._(_isolate, id, maxSequences);
  }

  /// Tears down the worker isolate. The instance is unusable afterwards.
  Future<void> shutdown() => _isolate.dispose();
}
