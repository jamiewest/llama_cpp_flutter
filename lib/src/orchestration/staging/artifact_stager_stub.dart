import 'package:llama_cpp_flutter/llama_cpp_flutter.dart';

import 'staged_artifacts.dart';

/// Web stub: the web runtime streams and caches the spec's URLs itself,
/// so staging just forwards whatever local blob URLs the caller supplied.
/// No estimate is produced; the orchestrator falls back to fixed budgets.
Future<StagedArtifacts> stageModelArtifacts(
  ModelSpec spec, {
  required String cacheDirectory,
  String? localPath,
  String? localMmprojPath,
  String? localDraftPath,
  ModelDownloader? downloader,
  LlamaLoadProgress? onProgress,
}) async => StagedArtifacts(
  modelPath: localPath,
  mmprojPath: localMmprojPath,
  draftPath: localDraftPath,
);
