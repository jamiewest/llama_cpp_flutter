import '../memory/memory_estimator.dart';

/// Result of staging a model's artifacts ahead of a runtime load.
class StagedArtifacts {
  /// Creates the result.
  const StagedArtifacts({
    this.modelPath,
    this.mmprojPath,
    this.draftPath,
    this.estimate,
  });

  /// Local model path to hand to `LlamaRuntime.loadModel`, or null when
  /// the runtime streams the spec's URLs itself (web).
  final String? modelPath;

  /// Local projector path, when staged.
  final String? mmprojPath;

  /// Local draft-model path, when staged.
  final String? draftPath;

  /// Memory estimate read from the staged GGUF header, or null when the
  /// header was unreadable or nothing was staged locally.
  final ModelMemoryEstimate? estimate;
}
