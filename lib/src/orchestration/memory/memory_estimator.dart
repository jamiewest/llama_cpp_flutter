/// Model memory estimation from GGUF metadata.
///
/// The estimate keys off the hyperparameters that actually cost memory —
/// weights (file size), KV cache per token, and compute buffers. Sampling
/// parameters (temperature, top-k, top-p) cost nothing and are ignored.
library;

/// Baseline compute/graph overhead independent of model size.
const int _baseOverheadBytes = 64 * 1024 * 1024;

/// Per-embedding-dimension compute-buffer cost, assuming the plugin's
/// 2048-token micro-batch cap. Deliberately conservative; calibrate against
/// `LlamaSession.stateSizeBytes` and real footprints over time.
const int _computeOverheadPerEmbedding = 2048 * 8;

/// GGUF metadata keys the estimator wants for [architecture], suitable for
/// `readGgufMetadata` / `readGgufMetadataFile`.
Set<String> ggufMemoryMetadataKeys(String architecture) => {
  '$architecture.block_count',
  '$architecture.embedding_length',
  '$architecture.attention.head_count',
  '$architecture.attention.head_count_kv',
  '$architecture.attention.key_length',
  '$architecture.attention.value_length',
  '$architecture.context_length',
};

/// What loading a model costs, split into the parts that scale differently.
class ModelMemoryEstimate {
  /// Creates an estimate.
  const ModelMemoryEstimate({
    required this.weightsBytes,
    required this.kvBytesPerToken,
    required this.fixedOverheadBytes,
    this.trainedContextTokens,
  });

  /// Weight cost — the GGUF file size(s). The file is mmapped, but with
  /// full GPU offload on unified memory it is effectively resident.
  final int weightsBytes;

  /// KV-cache cost per context token (f16 K+V across all layers).
  ///
  /// Conservative for models with interleaved sliding-window layers
  /// (Gemma 3+ style): those layers cap out below `n_ctx`, so the true
  /// cost is lower than this linear estimate.
  final int kvBytesPerToken;

  /// Compute-graph and output-buffer overhead, independent of context.
  final int fixedOverheadBytes;

  /// The context length the model was trained for, when the GGUF reports
  /// it. Contexts beyond it degrade quality, so planners clamp to it.
  final int? trainedContextTokens;

  /// Total bytes to run with a [contextTokens]-sized window.
  int bytesForContext(int contextTokens) =>
      weightsBytes + fixedOverheadBytes + kvBytesPerToken * contextTokens;

  /// The largest context that fits in [budgetBytes], or `0` when even an
  /// empty context does not fit.
  int maxContextForBudget(int budgetBytes) {
    if (kvBytesPerToken <= 0) return 0;
    final kvBudget = budgetBytes - weightsBytes - fixedOverheadBytes;
    return kvBudget <= 0 ? 0 : kvBudget ~/ kvBytesPerToken;
  }
}

/// Builds a [ModelMemoryEstimate] from GGUF numeric [metadata] (keyed as
/// `<architecture>.<suffix>`) and artifact file sizes.
///
/// Returns null when the metadata lacks the required hyperparameters
/// (block count, embedding length, head count) — callers then fall back to
/// loading with the spec's context size and skipping dynamic budgeting.
ModelMemoryEstimate? estimateModelMemory({
  required String architecture,
  required Map<String, int> metadata,
  required int modelFileSizeBytes,
  int draftFileSizeBytes = 0,
}) {
  int? value(String suffix) => metadata['$architecture.$suffix'];

  final layerCount = value('block_count');
  final embeddingLength = value('embedding_length');
  final headCount = value('attention.head_count');
  if (layerCount == null || layerCount <= 0) return null;
  if (embeddingLength == null || embeddingLength <= 0) return null;
  if (headCount == null || headCount <= 0) return null;

  // Missing KV head count (e.g. stored per-layer as an array) falls back
  // to full multi-head attention — an over-estimate, which is the safe
  // direction for a budget.
  final kvHeadCount = value('attention.head_count_kv') ?? headCount;
  final headDimension = embeddingLength ~/ headCount;
  final keyLength = value('attention.key_length') ?? headDimension;
  final valueLength = value('attention.value_length') ?? headDimension;

  const kvBytesPerElement = 2;
  final kvBytesPerToken =
      layerCount * kvHeadCount * (keyLength + valueLength) * kvBytesPerElement;

  return ModelMemoryEstimate(
    weightsBytes: modelFileSizeBytes + draftFileSizeBytes,
    kvBytesPerToken: kvBytesPerToken,
    fixedOverheadBytes:
        _baseOverheadBytes + embeddingLength * _computeOverheadPerEmbedding,
    trainedContextTokens: value('context_length'),
  );
}
