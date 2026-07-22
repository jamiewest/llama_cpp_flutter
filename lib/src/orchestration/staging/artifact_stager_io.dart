import 'dart:io';

import 'package:llama_flutter/llama_flutter.dart';

import 'staged_artifacts.dart';

/// Stages [spec]'s artifacts locally and reads a memory estimate from the
/// model's GGUF header.
///
/// Explicit local paths win; anything missing is downloaded into
/// `<cacheDirectory>/models` (resumable, URL-keyed — see
/// `ModelDownloader`). Estimation failures are swallowed: a model that
/// loads but cannot be estimated still runs, just without dynamic
/// budgeting.
Future<StagedArtifacts> stageModelArtifacts(
  ModelSpec spec, {
  required String cacheDirectory,
  String? localPath,
  String? localMmprojPath,
  String? localDraftPath,
  ModelDownloader? downloader,
  LlamaLoadProgress? onProgress,
}) async {
  var modelPath = localPath;
  var mmprojPath = localMmprojPath;
  var draftPath = localDraftPath;
  if (modelPath == null || modelPath.isEmpty) {
    final artifacts = await (downloader ?? ModelDownloader())
        .downloadSpecArtifacts(
          spec,
          directory: '$cacheDirectory/models',
          onProgress: onProgress,
        );
    modelPath = artifacts.modelPath;
    mmprojPath ??= artifacts.mmprojPath;
    draftPath ??= artifacts.draftPath;
  }

  return StagedArtifacts(
    modelPath: modelPath,
    mmprojPath: mmprojPath,
    draftPath: draftPath,
    estimate: await _estimateFromFile(modelPath, draftPath),
  );
}

Future<ModelMemoryEstimate?> _estimateFromFile(
  String modelPath,
  String? draftPath,
) async {
  try {
    final archResult = await readGgufMetadataFile(
      modelPath,
      keys: {ggufArchitectureKey},
    );
    if (archResult is! GgufMetadata) return null;
    final architecture = archResult.values[ggufArchitectureKey];
    if (architecture == null) return null;

    final result = await readGgufMetadataFile(
      modelPath,
      keys: ggufMemoryMetadataKeys(architecture),
    );
    if (result is! GgufMetadata) return null;

    final draftSize = draftPath == null || draftPath.isEmpty
        ? 0
        : await File(draftPath).length();
    return estimateModelMemory(
      architecture: architecture,
      metadata: result.numericValues,
      modelFileSizeBytes: await File(modelPath).length(),
      draftFileSizeBytes: draftSize,
    );
  } on Object {
    return null;
  }
}
