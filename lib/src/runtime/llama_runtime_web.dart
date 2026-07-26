// ignore_for_file: avoid_web_libraries_in_flutter, use_null_aware_elements

import 'dart:async';
import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:extensions/logging.dart';
import 'package:web/web.dart' as web;

import '../models/model_spec.dart';
import 'artifact_store_api.dart';
import 'gguf_split.dart';
import 'llama_runtime_api.dart';
import 'opfs_artifacts.dart';
import 'stop_sequence_filter.dart';

const int _maxWebModelBytes = 0x7fffffff;

/// Creates the wllama runtime for Flutter web.
///
/// [artifactCacheDirectory] is ignored: the web runtime streams models
/// straight from their URLs into browser-managed storage.
LlamaRuntime createLlamaRuntime({
  String? artifactCacheDirectory,
  LoggerFactory? loggerFactory,
}) => WebLlamaRuntime(loggerFactory: loggerFactory);

/// Loads GGUF models through a page-provided wllama JavaScript constructor.
///
/// The page must expose `globalThis.Wllama` from `@wllama/wllama` before a
/// model is loaded. The package provides the wasm asset paths expected by that
/// constructor.
final class WebLlamaRuntime implements LlamaRuntime {
  /// Creates the runtime. [loggerFactory] supplies the model-staging logger;
  /// null logs nothing.
  WebLlamaRuntime({LoggerFactory? loggerFactory})
    : _logger = (loggerFactory ?? NullLoggerFactory.instance).createLogger(
        'WebLlamaRuntime',
      );

  final Logger _logger;

  @override
  bool get supportsMultiThreading => web.window.crossOriginIsolated;

  @override
  Future<LlamaSession> loadModel(
    ModelSpec spec, {
    String? localPath,
    String? localMmprojPath,
    String? localDraftPath,
    LlamaLoadProgress? onProgress,
  }) async {
    final selectedDraftPath = localDraftPath?.trim();
    if (spec.draftUrl != null ||
        (selectedDraftPath != null && selectedDraftPath.isNotEmpty)) {
      throw UnsupportedError(
        'Speculative-decoding draft (MTP) models are not supported by the '
        'web local llama runtime: wllama cannot stage a second GGUF for '
        'spec_draft_model. Remove the draft artifact or run the native app.',
      );
    }

    final constructor = web.window.getProperty<JSFunction?>('Wllama'.toJS);
    if (constructor == null) {
      throw StateError(
        'Wllama is not available on globalThis. Load @wllama/wllama before '
        'creating a local llama web session.',
      );
    }

    // wllama 3.x ships one unified wasm (threading is chosen at runtime) and
    // reads only the 'default' key from this map; the per-thread-count paths
    // of wllama 1.x/2.x are gone.
    final instance = constructor.callAsConstructor<JSObject>(
      <String, String>{'default': llamaWasmAssetPath}.jsify(),
    );

    final selectedLocalPath = localPath?.trim();
    if (selectedLocalPath != null && selectedLocalPath.isNotEmpty) {
      _logger.logInformation(
        'Loading model "${spec.id}" from local blob URL '
        '(contextSize: ${spec.contextSize}).',
      );
      await _loadFromLocalBlobs(
        instance,
        spec,
        modelUrl: selectedLocalPath,
        mmprojUrl: localMmprojPath?.trim(),
      );
      return _WebLlamaSession(instance, _effectiveContextSize(instance, spec));
    }

    // A single file of 2 GiB or more cannot be staged in wasm32, so it
    // is downloaded once into OPFS and split client-side into the same
    // multi-file layout `llama-gguf-split` would produce. Smaller
    // models keep wllama's own URL download cache.
    final contentLength = await _fetchContentLength(spec.modelUrl);
    if (contentLength != null && contentLength > _maxWebModelBytes) {
      _logger.logInformation(
        'Loading oversized model "${spec.id}" ($contentLength bytes) via '
        'OPFS client-side split (contextSize: ${spec.contextSize}).',
      );
      await _loadOversizedFromUrl(instance, spec, contentLength, onProgress);
    } else {
      _logger.logInformation(
        'Loading model "${spec.id}" from ${spec.modelUrl} '
        '(contextSize: ${spec.contextSize}).',
      );
      await _loadFromUrl(instance, spec, onProgress);
    }
    return _WebLlamaSession(instance, _effectiveContextSize(instance, spec));
  }

  /// The context window the loaded model actually has.
  ///
  /// llama.cpp's server slot (which wllama wraps) silently caps its context
  /// at the model's training context, so a spec asking for more than
  /// `n_ctx_train` gets `n_ctx_train`. Budgeting against the requested size
  /// would let over-long prompts through the fail-fast guard only to die in
  /// the wasm with a confusing error naming a window the caller never chose.
  static int _effectiveContextSize(JSObject instance, ModelSpec spec) {
    try {
      final info = instance.callMethod<JSObject?>(
        'getLoadedContextInfo'.toJS,
      );
      final trainContext = info
          ?.getProperty<JSNumber?>('n_ctx_train'.toJS)
          ?.toDartInt;
      if (trainContext != null && trainContext > 0) {
        return math.min(spec.contextSize, trainContext);
      }
    } on Object {
      // Older wllama builds without getLoadedContextInfo: keep the spec.
    }
    return spec.contextSize;
  }

  /// Stages already-resolved local artifacts (splitting the model if
  /// oversized) and loads them.
  static Future<void> _loadFromLocalBlobs(
    JSObject instance,
    ModelSpec spec, {
    required String modelUrl,
    String? mmprojUrl,
  }) async {
    final blobs = <web.Blob>[
      ...await _asLoadableParts(await _resolveBlob(modelUrl)),
    ];
    if (mmprojUrl != null && mmprojUrl.isNotEmpty) {
      // wllama identifies the projector blob by its GGUF metadata
      // (general.architecture == "clip"), so order does not matter.
      blobs.add(await _resolveBlob(mmprojUrl));
    }
    await _loadFromBlobs(instance, blobs, spec);
  }

  /// Resolves one `loadModel` path argument to a blob.
  ///
  /// A managed-storage handle (see `ArtifactStore.resolve`) resolves
  /// straight to its OPFS file: the bytes stay on disk, and an oversized
  /// model is sliced into stageable parts from there. Anything else is a
  /// URL — a blob URL from a file input, say — and is fetched.
  static Future<web.Blob> _resolveBlob(String path) async {
    final key = artifactKeyOf(path);
    if (key == null) return _fetchBlob(path);
    final directory = await requireArtifactDirectory();
    final file = await artifactFile(directory, key);
    if (file == null) {
      throw ArtifactStorageException(
        'The artifact "$key" is no longer in managed storage.',
      );
    }
    return file;
  }

  /// Downloads a model too large for wllama's URL cache into OPFS, splits
  /// it client-side, and loads the parts.
  static Future<void> _loadOversizedFromUrl(
    JSObject instance,
    ModelSpec spec,
    int contentLength,
    LlamaLoadProgress? onProgress,
  ) async {
    final model = await _cachedLargeModel(
      spec.modelUrl,
      contentLength,
      onProgress,
    );
    final blobs = <web.Blob>[
      ...await _asLoadableParts(model),
      if (spec.mmprojUrl != null) await _fetchBlob(spec.mmprojUrl.toString()),
    ];
    await _loadFromBlobs(instance, blobs, spec);
  }

  /// Loads directly from the spec's URLs through wllama's own download cache.
  static Future<void> _loadFromUrl(
    JSObject instance,
    ModelSpec spec,
    LlamaLoadProgress? onProgress,
  ) async {
    await instance.callMethodVarArgs<JSPromise<JSAny?>>(
      'loadModelFromUrl'.toJS,
      [
        <String, Object?>{
          'url': spec.modelUrl.toString(),
          if (spec.mmprojUrl != null) 'mmprojUrl': spec.mmprojUrl.toString(),
        }.jsify(),
        <String, Object?>{
          'n_ctx': spec.contextSize,
          'n_gpu_layers': spec.gpuLayers,
          'useCache': true,
          if (onProgress != null)
            'progressCallback': _createProgressCallback(onProgress),
        }.jsify(),
      ],
    ).toDart;
  }

  static Future<void> _loadFromBlobs(
    JSObject instance,
    List<web.Blob> blobs,
    ModelSpec spec,
  ) async {
    await instance.callMethodVarArgs<JSPromise<JSAny?>>('loadModel'.toJS, [
      blobs.toJS,
      <String, Object?>{
        'n_ctx': spec.contextSize,
        'n_gpu_layers': spec.gpuLayers,
      }.jsify(),
    ]).toDart;
  }

  /// Returns [model] as the blob list wllama can stage: the blob itself
  /// when it fits the wasm32 per-file limit, otherwise GGUF splits
  /// composed of a rewritten header plus zero-copy slices of [model].
  static Future<List<web.Blob>> _asLoadableParts(web.Blob model) async {
    if (model.size <= _maxWebModelBytes) return [model];

    // The header (metadata plus tensor infos) must be parsed whole; its
    // size is unknown up front, so read a growing prefix. Tokenizer
    // metadata dominates and rarely passes a few tens of megabytes.
    var prefixLength = 8 * 1024 * 1024;
    const maxPrefixLength = 1024 * 1024 * 1024;
    while (true) {
      prefixLength = math.min(prefixLength, model.size);
      final prefix = (await model.slice(0, prefixLength).arrayBuffer().toDart)
          .toDart
          .asUint8List();
      final result = planGgufSplit(
        headerPrefix: prefix,
        totalBytes: model.size,
      );
      switch (result) {
        case GgufSplitPlan(:final parts):
          return [
            for (final part in parts)
              web.Blob(
                [
                  part.headerBytes.toJS,
                  model.slice(part.dataStart, part.dataEnd),
                ].toJS,
              ),
          ];
        case GgufSplitNeedsLargerPrefix():
          if (prefixLength >= model.size || prefixLength >= maxPrefixLength) {
            throw UnsupportedError(
              'The GGUF header could not be parsed for client-side '
              'splitting. Use a pre-split GGUF (a file ending in '
              '-00001-of-00002.gguf) or run the native app.',
            );
          }
          prefixLength *= 8;
        case GgufSplitUnsupported(:final reason):
          throw UnsupportedError(
            'This ${_formatBytes(model.size)} GGUF cannot be loaded on '
            'web: $reason Use a pre-split GGUF (a file ending in '
            '-00001-of-00002.gguf) or run the native app.',
          );
      }
    }
  }

  /// Downloads an oversized model into managed OPFS storage (reusing a
  /// previous copy, and resuming an interrupted one) and returns it as a
  /// disk-backed [web.Blob].
  ///
  /// The copy lands where `ArtifactStore` looks, so a model the runtime
  /// downloaded on its own is listable and deletable through managed
  /// storage. Falls back to an in-memory fetch when OPFS is unavailable —
  /// the model then loads but is re-downloaded next session.
  static Future<web.Blob> _cachedLargeModel(
    Uri modelUrl,
    int contentLength,
    LlamaLoadProgress? onProgress,
  ) async {
    final directory = await openArtifactDirectory();
    if (directory == null) {
      final response = await _fetchOk(modelUrl.toString());
      return response.blob().toDart;
    }
    return ensureArtifactFromUrl(
      directory,
      modelUrl,
      contentLength: contentLength,
      onProgress: onProgress,
    );
  }

  static Future<web.Response> _fetchOk(String url) async {
    final response = await web.window
        .callMethodVarArgs<JSPromise<web.Response>>('fetch'.toJS, [url.toJS])
        .toDart;
    if (!response.ok) {
      throw StateError(
        'The model download failed with HTTP ${response.status}.',
      );
    }
    return response;
  }

  static Future<web.Blob> _fetchBlob(String objectUrl) async {
    final response = await _fetchOk(objectUrl);
    return response.blob().toDart;
  }

  static Future<int?> _fetchContentLength(Uri modelUrl) async {
    try {
      final response = await web.window.callMethodVarArgs<JSPromise<JSObject>>(
        'fetch'.toJS,
        [
          modelUrl.toString().toJS,
          <String, Object?>{'method': 'HEAD'}.jsify(),
        ],
      ).toDart;
      final headers = response.getProperty<JSObject?>('headers'.toJS);
      final contentLength = headers
          ?.callMethod<JSString?>('get'.toJS, 'content-length'.toJS)
          ?.toDart;
      return contentLength == null ? null : int.tryParse(contentLength);
    } on UnsupportedError {
      rethrow;
    } catch (_) {
      return null;
    }
  }

  static String _formatBytes(int bytes) {
    final gibibytes = bytes / (1024 * 1024 * 1024);
    return '${gibibytes.toStringAsFixed(2)} GiB';
  }

  static JSFunction _createProgressCallback(
    LlamaLoadProgress onProgress,
  ) => ((JSObject progress) {
    final loaded = progress.getProperty<JSNumber?>('loaded'.toJS)?.toDartDouble;
    final total = progress.getProperty<JSNumber?>('total'.toJS)?.toDartDouble;
    if (loaded == null || total == null || total <= 0) return;
    onProgress((loaded / total).clamp(0, 1).toDouble());
  }).toJS;
}

final class _WebLlamaSession implements LlamaSession {
  _WebLlamaSession(this._wllama, this._contextSize);

  final JSObject _wllama;

  /// The `n_ctx` the model was loaded with. Used to fail fast when a prompt
  /// cannot fit (wllama's prefill hangs rather than erroring) and to clamp
  /// `max_tokens` so prompt + output stays within the window.
  final int _contextSize;

  /// Pessimistic characters-per-token ratio for the token estimate below.
  /// Real tokenization varies, but ~3 chars/token over-counts for English
  /// prose, which is what we want for a conservative "does it fit" guard.
  static const double _charsPerToken = 3;

  /// Aborts the in-flight run, set by [generate] for [cancel] to invoke.
  void Function()? _abortCurrentRun;

  @override
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
  }) {
    if (sequenceId != 0) {
      throw ArgumentError.value(
        sequenceId,
        'sequenceId',
        'wllama has no KV-sequence support; only sequence 0 exists.',
      );
    }
    final isMediaTurn = media != null && media.isNotEmpty;
    if (isMediaTurn) {
      if (turns == null) {
        throw UnsupportedError(
          'Media input on the web local llama runtime requires structured '
          'chat turns.',
        );
      }
      final hasImage = turns.any((turn) => turn.images.isNotEmpty);
      final hasAudio = turns.any((turn) => turn.audio.isNotEmpty);
      if (hasImage && !_supportsModality('image')) {
        throw StateError(
          'The loaded local llama model has no vision projector. Configure a '
          'projector (mmproj) GGUF for this model to send images.',
        );
      }
      if (hasAudio && !_supportsModality('audio')) {
        throw StateError(
          'The loaded local llama model has no audio projector. Configure an '
          'audio-capable projector (mmproj) GGUF for this model to send audio.',
        );
      }
    }

    final controller = StreamController<String>();

    // A prompt longer than the context window makes wllama's prefill hang
    // indefinitely instead of erroring. Estimate prompt tokens pessimistically
    // and fail fast with an actionable message; also clamp `max_tokens` so the
    // prompt plus the requested output stays within `n_ctx`. Skip for media
    // turns, whose token cost is dominated by mtmd image/audio tokens we can't
    // size from the text here.
    var effectiveMaxTokens = maxTokens;
    if (!isMediaTurn) {
      final estimatedPromptTokens = (prompt.length / _charsPerToken).ceil();
      if (estimatedPromptTokens >= _contextSize) {
        controller
          ..addError(
            StateError(
              'The prompt (~$estimatedPromptTokens tokens) does not fit the '
              'model context ($_contextSize tokens). Increase the model '
              'context size or shorten the prompt (e.g. fewer tools).',
            ),
          )
          ..close();
        return StopSequenceFilter(stopSequences).bind(controller.stream);
      }
      final room = _contextSize - estimatedPromptTokens;
      effectiveMaxTokens = maxTokens.clamp(1, room);
    }

    // wllama has no server-side stop sequences and keeps generating to
    // `max_tokens` unless aborted, so every way this run can end early — a
    // stop-sequence match in the Dart filter, cancel(), or the listener
    // unsubscribing — must abort the underlying completion. `deliberateAbort`
    // marks the resulting promise rejection as a normal end, not an error.
    var deliberateAbort = false;
    var stopMatched = false;
    final abortController = _createAbortController();
    void abortRun() {
      deliberateAbort = true;
      abortController?.callMethod<JSAny?>('abort'.toJS);
    }

    _abortCurrentRun = abortRun;
    final abortSignal = abortController?.getProperty<JSAny?>('signal'.toJS);

    final options =
        <String, Object?>{
              if (isMediaTurn)
                'messages': _chatMessages(turns!)
              else
                'prompt': prompt,
              'stream': true,
              'max_tokens': effectiveMaxTokens,
              'temp': temperature,
              if (topK != null) 'top_k': topK,
              if (topP != null) 'top_p': topP,
              if (seed != null) 'seed': seed,
              if (abortSignal != null) 'abortSignal': abortSignal,
            }.jsify()
            as JSObject;
    // Token accounting: wllama invokes onData once per generated token, and
    // its tokenizer sizes the prompt. The count runs concurrently with the
    // generation and is best-effort — a tokenize failure only skips stats,
    // never the generation. Media turns approximate: the count covers the
    // rendered text prompt, not mtmd image/audio tokens.
    var generatedCount = 0;
    final promptTokenCount = onStats == null
        ? Future<int?>.value()
        : _countPromptTokens(prompt);

    options['onData'] = ((JSObject chunk) {
      generatedCount++;
      final text = isMediaTurn ? _chatChunkText(chunk) : _chunkText(chunk);
      if (text.isNotEmpty && !controller.isClosed) {
        controller.add(text);
      }
    }).toJS;

    final requestedMaxTokens = effectiveMaxTokens;
    var settled = false;
    // Closes the token controller and then reports stats (best-effort).
    // Invoked exactly once, from whichever of the promise's resolution or a
    // deliberate abort happens first. The close is awaited before the finish
    // reason is read so the stop filter has consumed every queued token —
    // a stop sequence in the final pieces still counts.
    Future<void> settle() async {
      if (settled) return;
      settled = true;
      // Snapshot before the close: draining the stream ends the consumer,
      // whose exit abort (see _abortOnExit) must not read as a cancellation.
      final wasCancelled = deliberateAbort;
      if (!controller.isClosed) await controller.close();
      if (onStats != null) {
        final promptTokens = await promptTokenCount;
        if (promptTokens != null) {
          final LlamaFinishReason reason;
          if (stopMatched) {
            reason = LlamaFinishReason.stopSequence;
          } else if (wasCancelled) {
            reason = LlamaFinishReason.cancelled;
          } else if (generatedCount >= requestedMaxTokens) {
            reason = LlamaFinishReason.maxTokens;
          } else {
            reason = LlamaFinishReason.eogToken;
          }
          onStats(
            LlamaGenerationStats(
              promptTokenCount: promptTokens,
              cachedTokenCount: 0,
              generatedTokenCount: generatedCount,
              finishReason: reason,
            ),
          );
        }
      }
    }

    final method = isMediaTurn ? 'createChatCompletion' : 'createCompletion';
    unawaited(
      _wllama
          .callMethod<JSPromise<JSAny?>>(method.toJS, options)
          .toDart
          .then((_) => settle())
          .catchError((Object error, StackTrace stackTrace) async {
            // A rejection caused by our own abort() is a normal early end
            // (stop sequence, cancel, or unsubscribe), not a failure.
            if (deliberateAbort) {
              await settle();
              return;
            }
            settled = true;
            if (!controller.isClosed) {
              controller
                ..addError(error, stackTrace)
                ..close();
            }
          }),
    );

    final filtered = StopSequenceFilter(
      stopSequences,
      onStop: () => stopMatched = true,
    ).bind(controller.stream);
    return _abortOnExit(filtered, abortRun);
  }

  /// Yields [source], guaranteeing [abort] runs when the consumer is done
  /// with the stream — a stop sequence matched, the listener cancelled, or
  /// the stream ended naturally (where the abort is a no-op because the
  /// completion promise has already settled).
  static Stream<String> _abortOnExit(
    Stream<String> source,
    void Function() abort,
  ) async* {
    try {
      yield* source;
    } finally {
      abort();
    }
  }

  /// Sizes [prompt] with wllama's tokenizer; null when tokenization fails.
  Future<int?> _countPromptTokens(String prompt) async {
    try {
      final result = await _wllama
          .callMethod<JSPromise<JSAny?>>('tokenize'.toJS, prompt.toJS)
          .toDart;
      final tokens = result as JSObject?;
      return tokens?.getProperty<JSNumber?>('length'.toJS)?.toDartInt;
    } on Object {
      return null;
    }
  }

  /// Whether wllama reports the loaded model accepts [modality]
  /// (`'image'` or `'audio'`) — i.e. an appropriate mmproj is loaded.
  bool _supportsModality(String modality) =>
      _wllama
          .callMethod<JSBoolean?>('supportInputModality'.toJS, modality.toJS)
          ?.toDart ??
      false;

  /// Converts [turns] to wllama's OAI-style chat messages. Turns with media
  /// carry `{type: 'image'|'audio', data: <bytes>}` parts in author order
  /// (wllama accepts any typed array where its types say `ArrayBuffer`);
  /// text-only turns use plain string content. Tool turns fold into user
  /// turns — wllama's chat-completion API has no tool role.
  static List<Map<String, Object?>> _chatMessages(List<LlamaChatTurn> turns) =>
      <Map<String, Object?>>[
        for (final turn in turns)
          <String, Object?>{
            'role': switch (turn.role) {
              LlamaChatRole.system => 'system',
              LlamaChatRole.assistant => 'assistant',
              LlamaChatRole.user || LlamaChatRole.tool => 'user',
            },
            'content': turn.images.isEmpty && turn.audio.isEmpty
                ? turn.text
                : <Map<String, Object?>>[
                    for (final part in turn.parts)
                      ...switch (part) {
                        LlamaImagePart(:final bytes) => [
                          <String, Object?>{'type': 'image', 'data': bytes},
                        ],
                        LlamaAudioPart(:final bytes) => [
                          <String, Object?>{'type': 'audio', 'data': bytes},
                        ],
                        LlamaTextPart(:final text) => [
                          if (text.isNotEmpty)
                            <String, Object?>{'type': 'text', 'text': text},
                        ],
                      },
                  ],
          },
      ];

  @override
  Future<void> cancel() async {
    _abortCurrentRun?.call();
  }

  @override
  LlamaSessionCapabilities get capabilities => const LlamaSessionCapabilities(
    canPersistState: false,
    reportsStateSize: false,
  );

  @override
  Future<int> saveState(String path, {int sequenceId = 0}) async => 0;

  @override
  Future<int> loadState(String path, {int sequenceId = 0}) async {
    throw UnsupportedError(
      'wllama exposes no KV-cache state restore; the web local llama '
      'runtime re-prefills instead.',
    );
  }

  @override
  Future<int> stateSizeBytes({int sequenceId = 0}) async => 0;

  @override
  Future<LlamaStashResult> stashState(String key, {int sequenceId = 0}) async =>
      (tokens: 0, bytes: 0);

  @override
  Future<int> restoreStashedState(String key, {int sequenceId = 0}) async {
    throw UnsupportedError(
      'wllama exposes no KV-cache state restore; the web local llama '
      'runtime re-prefills instead.',
    );
  }

  @override
  Future<int> dropStashedState(String key) async => 0;

  @override
  Future<void> clearSequence(int sequenceId) async {}

  @override
  Future<void> setImageTokenBudget(int? imageTokenBudget) async {
    throw UnsupportedError(
      'wllama exposes no image token budget; the web local llama runtime '
      'uses the model\'s metadata default.',
    );
  }

  @override
  Future<void> dispose() async {
    await cancel();
    await _wllama.callMethod<JSPromise<JSAny?>>('exit'.toJS).toDart;
  }

  static JSObject? _createAbortController() {
    final constructor = web.window.getProperty<JSFunction?>(
      'AbortController'.toJS,
    );
    if (constructor == null) return null;
    return constructor.callAsConstructor<JSObject>();
  }

  static String _chunkText(JSObject chunk) {
    final first = _firstChoice(chunk);
    if (first == null) return '';
    return first.getProperty<JSString?>('text'.toJS)?.toDart ?? '';
  }

  static String _chatChunkText(JSObject chunk) {
    final first = _firstChoice(chunk);
    if (first == null) return '';
    final delta = first.getProperty<JSObject?>('delta'.toJS);
    if (delta == null) return '';
    return delta.getProperty<JSString?>('content'.toJS)?.toDart ?? '';
  }

  static JSObject? _firstChoice(JSObject chunk) {
    final choices = chunk.getProperty<JSObject?>('choices'.toJS);
    return choices?.getProperty<JSObject?>('0'.toJS);
  }
}
