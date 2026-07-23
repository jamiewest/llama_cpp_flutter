import 'dart:io';
import 'dart:typed_data';

import 'package:extensions/logging.dart';
import 'package:llama_cpp_flutter/bridge.dart' as native;

import '../models/model_spec.dart';
import 'llama_runtime_api.dart';
import 'model_downloader_io.dart';

/// Creates the native llama.cpp runtime for iOS and macOS.
///
/// With [artifactCacheDirectory] set, `loadModel` calls that omit
/// `localPath` download the spec's artifact URLs into that directory
/// first (see [NativeLlamaRuntime.new]).
///
/// Throws [UnsupportedError] on IO platforms without a native backend
/// (Android, Windows, Linux) — the plugin currently registers
/// implementations for iOS and macOS only.
LlamaRuntime createLlamaRuntime({
  String? artifactCacheDirectory,
  LoggerFactory? loggerFactory,
}) {
  if (!Platform.isIOS && !Platform.isMacOS) {
    throw UnsupportedError(
      'llama_cpp_flutter supports native inference on iOS and macOS only '
      '(current platform: ${Platform.operatingSystem}). Web is supported '
      'through wllama; Android/Windows/Linux backends do not exist yet.',
    );
  }
  return NativeLlamaRuntime(
    artifactCacheDirectory: artifactCacheDirectory,
    loggerFactory: loggerFactory,
  );
}

/// Loads GGUF models through the `llama_cpp_flutter` plugin.
final class NativeLlamaRuntime implements LlamaRuntime {
  /// Creates the runtime.
  ///
  /// With [artifactCacheDirectory] set, `loadModel` calls that omit
  /// `localPath` download the spec's artifact URLs (e.g. Hugging Face
  /// `resolve` URLs) into that directory first — the same
  /// load-straight-from-the-spec experience the web runtime gives.
  /// [downloader] overrides how those downloads run (auth token, fakes in
  /// tests).
  NativeLlamaRuntime({
    native.LlamaCppFlutter? llama,
    ModelDownloader? downloader,
    String? artifactCacheDirectory,
    LoggerFactory? loggerFactory,
  }) : _llama = llama ?? native.LlamaCppFlutter(loggerFactory: loggerFactory),
       _downloader = downloader,
       _artifactCacheDirectory = artifactCacheDirectory,
       _loggerFactory = loggerFactory,
       _logger = (loggerFactory ?? NullLoggerFactory.instance).createLogger(
         'NativeLlamaRuntime',
       );

  final native.LlamaCppFlutter _llama;
  final ModelDownloader? _downloader;
  final String? _artifactCacheDirectory;
  final LoggerFactory? _loggerFactory;
  final Logger _logger;

  @override
  bool get supportsMultiThreading => true;

  @override
  Future<LlamaSession> loadModel(
    ModelSpec spec, {
    String? localPath,
    String? localMmprojPath,
    String? localDraftPath,
    LlamaLoadProgress? onProgress,
  }) async {
    var modelPath = localPath;
    var mmprojPath = localMmprojPath;
    var draftPath = localDraftPath;
    if (modelPath == null || modelPath.isEmpty) {
      final cacheDirectory = _artifactCacheDirectory;
      if (cacheDirectory == null) {
        throw ArgumentError.value(
          localPath,
          'localPath',
          'Native llama runtime requires a downloaded model path (or '
              'construct NativeLlamaRuntime with an artifactCacheDirectory '
              'to download the spec\'s URLs automatically).',
        );
      }
      _logger.logInformation(
        'Downloading model artifacts for "${spec.id}" into $cacheDirectory.',
      );
      final artifacts =
          await (_downloader ?? ModelDownloader(loggerFactory: _loggerFactory))
          .downloadSpecArtifacts(
            spec,
            directory: cacheDirectory,
            onProgress: onProgress,
          );
      modelPath = artifacts.modelPath;
      mmprojPath ??= artifacts.mmprojPath;
      draftPath ??= artifacts.draftPath;
    }

    final session = await _llama.loadModel(
      modelPath,
      contextSize: spec.contextSize,
      gpuLayers: spec.gpuLayers,
      mmprojPath: mmprojPath,
      imageTokenBudget: spec.imageTokenBudget,
      draftModelPath: draftPath,
      draftGpuLayers: spec.draftGpuLayers,
      maxDraftTokens: spec.maxDraftTokens,
      maxSequences: spec.maxSequences,
    );
    return _NativeLlamaSession(
      session,
      hasVisionProjector: mmprojPath != null && mmprojPath.isNotEmpty,
    );
  }
}

final class _NativeLlamaSession implements LlamaSession {
  _NativeLlamaSession(this._inner, {required bool hasVisionProjector})
    : _hasVisionProjector = hasVisionProjector;

  final native.LlamaBridgeSession _inner;

  /// Whether this session was loaded with a multimodal projector; without
  /// one there is no vision encoder whose token budget could change.
  final bool _hasVisionProjector;

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
  }) => _inner.generate(
    prompt,
    maxTokens: maxTokens,
    temperature: temperature,
    topK: topK,
    topP: topP,
    seed: seed,
    stopSequences: stopSequences,
    // The native FFI `images` parameter accepts any mtmd media bytes; the
    // Swift side auto-detects image vs audio from each blob's magic bytes.
    images: media,
    sequenceId: sequenceId,
    onStats: onStats == null
        ? null
        : (stats) => onStats(
            LlamaGenerationStats(
              promptTokenCount: stats.promptTokenCount,
              cachedTokenCount: stats.cachedTokenCount,
              generatedTokenCount: stats.generatedTokenCount,
              finishReason: _finishReason(stats.finishReason),
              prefillDuration: stats.prefillDuration,
              decodeDuration: stats.decodeDuration,
              draftedTokenCount: stats.draftedTokenCount,
              acceptedTokenCount: stats.acceptedTokenCount,
            ),
          ),
  );

  @override
  Future<void> cancel() => _inner.cancel();

  @override
  LlamaSessionCapabilities get capabilities => LlamaSessionCapabilities(
    canPersistState: true,
    reportsStateSize: true,
    canStashState: true,
    canSetImageTokenBudget: _hasVisionProjector,
    maxSequences: _inner.maxSequences,
  );

  @override
  Future<int> saveState(String path, {int sequenceId = 0}) =>
      _inner.saveState(path, sequenceId: sequenceId);

  @override
  Future<int> loadState(String path, {int sequenceId = 0}) =>
      _inner.loadState(path, sequenceId: sequenceId);

  @override
  Future<int> stateSizeBytes({int sequenceId = 0}) =>
      _inner.stateSizeBytes(sequenceId: sequenceId);

  @override
  Future<LlamaStashResult> stashState(String key, {int sequenceId = 0}) =>
      _inner.stashState(key, sequenceId: sequenceId);

  @override
  Future<int> restoreStashedState(String key, {int sequenceId = 0}) =>
      _inner.restoreStashedState(key, sequenceId: sequenceId);

  @override
  Future<int> dropStashedState(String key) => _inner.dropStashedState(key);

  @override
  Future<void> clearSequence(int sequenceId) =>
      _inner.clearSequence(sequenceId);

  @override
  Future<void> setImageTokenBudget(int? imageTokenBudget) =>
      _inner.setImageTokenBudget(imageTokenBudget);

  @override
  Future<void> dispose() => _inner.dispose();
}

/// Maps the plugin's finish reason onto the engine-neutral enum.
LlamaFinishReason _finishReason(native.LlamaFinishReason reason) =>
    switch (reason) {
      native.LlamaFinishReason.eogToken => LlamaFinishReason.eogToken,
      native.LlamaFinishReason.maxTokens => LlamaFinishReason.maxTokens,
      native.LlamaFinishReason.stopSequence => LlamaFinishReason.stopSequence,
      native.LlamaFinishReason.contextFull => LlamaFinishReason.contextFull,
      native.LlamaFinishReason.cancelled => LlamaFinishReason.cancelled,
      native.LlamaFinishReason.aborted => LlamaFinishReason.aborted,
    };
