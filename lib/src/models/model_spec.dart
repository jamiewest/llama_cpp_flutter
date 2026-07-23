/// Declarative description of an on-device model: where its artifacts live,
/// how to configure the engine, and which chat format it speaks.
library;

import '../llm/chat_format.dart';
import '../llm/image_tiling.dart';

/// A model's preferred sampling parameters, used when the per-request
/// `ChatOptions` doesn't override them.
class SamplingDefaults {
  const SamplingDefaults({
    this.maxTokens = 512,
    this.temperature = 1.0,
    this.topK,
    this.topP,
    this.seed,
  });

  /// Max tokens per generated turn.
  final int maxTokens;

  /// Sampling temperature.
  final double temperature;

  /// Top-k cutoff, or null for the engine default.
  final int? topK;

  /// Nucleus-sampling cutoff, or null for the engine default.
  final double? topP;

  /// Sampler seed for reproducible generation, or null for random.
  final int? seed;

  @override
  bool operator ==(Object other) =>
      other is SamplingDefaults &&
      other.maxTokens == maxTokens &&
      other.temperature == temperature &&
      other.topK == topK &&
      other.topP == topP &&
      other.seed == seed;

  @override
  int get hashCode => Object.hash(maxTokens, temperature, topK, topP, seed);

  @override
  String toString() =>
      'SamplingDefaults(maxTokens: $maxTokens, temperature: $temperature, '
      'topK: $topK, topP: $topP, seed: $seed)';
}

/// Everything the app needs to provision and run one model: artifact URLs,
/// engine configuration, the [ChatFormat] it speaks, and sampling defaults.
///
/// Specs are plain data declared in `model_registry.dart`; services
/// (download, load, chat client construction) consume whichever spec is
/// registered, so switching models is a registry change, not a code hunt.
class ModelSpec {
  ModelSpec({
    required this.id,
    required this.displayName,
    required this.modelUrl,
    this.mmprojUrl,
    this.imageTokenBudget,
    this.imageTiling,
    this.draftUrl,
    required this.contextSize,
    this.maxSequences = 1,
    this.gpuLayers = 999,
    this.draftGpuLayers = 999,
    this.maxDraftTokens = 3,
    required this.format,
    this.sampling = const SamplingDefaults(),
    this.engineId = 'llama',
    this.filterTools = true,
    this.enableThinking = true,
  }) : assert(contextSize > 0, 'contextSize must be positive'),
       assert(maxSequences > 0, 'maxSequences must be at least 1'),
       assert(maxDraftTokens > 0, 'maxDraftTokens must be at least 1');

  /// Stable identifier, e.g. `'gemma-4-e4b-it-q4km'`.
  final String id;

  /// Human-readable name for UI surfaces.
  final String displayName;

  /// Where the model weights are downloaded from.
  final Uri modelUrl;

  /// The multimodal projector pairing with [modelUrl] (mtmd vision), or null
  /// for a text-only model.
  final Uri? mmprojUrl;

  /// Vision-encoder token budget per image, or null for the model's
  /// metadata default.
  ///
  /// Only meaningful with [mmprojUrl], for vision models with dynamic
  /// resolution. Gemma supports 70, 140, 280, 560, or 1120: 70–140 suits
  /// classification/captioning, 280–560 charts and UI, 1120 OCR and small
  /// text. Higher budgets cost proportionally more prefill time per image.
  /// Native (mtmd) only — the web runtime has no equivalent knob and
  /// ignores it.
  final int? imageTokenBudget;

  /// How to split attached images into overlapping crops before prompt
  /// rendering, or null to pass images through whole.
  ///
  /// Tiling multiplies the effective resolution the vision encoder sees —
  /// each crop gets its own [imageTokenBudget] — at the cost of one
  /// prefill per crop. Pair with a high budget for dense-document OCR.
  final ImageTiling? imageTiling;

  /// The speculative-decoding drafter pairing with [modelUrl], or null to
  /// decode single-model.
  final Uri? draftUrl;

  /// Context window the engine allocates at load time.
  ///
  /// With [maxSequences] > 1 this is the TOTAL KV budget; each sequence
  /// gets `contextSize / maxSequences` positions.
  final int contextSize;

  /// Number of independent KV-cache sequences the engine allocates
  /// (llama.cpp `n_seq_max`). Engines without sequence support ignore it.
  ///
  /// Defaults to 1 (the classic single-conversation session). Speculative
  /// decoding is unavailable when > 1.
  final int maxSequences;

  /// GPU offload hint (llama.cpp layer count; 999 = everything on Metal).
  /// Engines that don't split layers ignore it.
  final int gpuLayers;

  /// GPU offload hint for the speculative-decoding drafter; only meaningful
  /// when [draftUrl] (or a locally selected draft artifact) is configured.
  final int draftGpuLayers;

  /// Upper bound on tokens the drafter proposes per speculation step.
  ///
  /// Defaults to 3, matching upstream llama.cpp's speculative n_max: at
  /// sampling temperatures the per-token acceptance rate makes longer draft
  /// chains net-negative (every rejected token costs a wasted draft decode
  /// plus a wasted verification slot).
  final int maxDraftTokens;

  /// The model family's prompt wire format.
  final ChatFormat format;

  /// Sampling parameters used when `ChatOptions` doesn't override them.
  final SamplingDefaults sampling;

  /// Whether per-turn keyword tool selection applies to this model's prompts.
  ///
  /// `true` (default) keeps `ToolSelectionContextProvider` in the agent's
  /// context-provider chain, shrinking the per-turn tool list to keyword-matched
  /// groups — a prefill/KV-cache win for capable models. Set `false` for a
  /// less-capable model where a keyword miss (a needed tool dropped from the
  /// prompt) hurts more than a longer tool list: the provider is omitted and the
  /// full registry reaches every prompt.
  final bool filterTools;

  /// Whether to inject the family's reasoning channel (Gemma's `<|think|>`
  /// marker) ahead of each answer.
  ///
  /// `true` (default) lets a capable model reason before responding. Set
  /// `false` for an edge model (e.g. E2B) where the thinking pass burns scarce
  /// decode tokens before any user-visible output for little quality gain. The
  /// flag is a no-op for formats whose `ChatFormat.supportsThinking` is false.
  final bool enableThinking;

  /// Returns a copy with the given fields replaced.
  ///
  /// Nullable fields ([mmprojUrl], [imageTokenBudget], [imageTiling],
  /// [draftUrl]) keep their current value; a copy cannot clear them. Exists mainly so an orchestrator can
  /// derive a spec with a memory-budgeted [contextSize] from a registry
  /// spec without hand-copying every field.
  ModelSpec copyWith({
    String? id,
    String? displayName,
    Uri? modelUrl,
    Uri? mmprojUrl,
    int? imageTokenBudget,
    ImageTiling? imageTiling,
    Uri? draftUrl,
    int? contextSize,
    int? maxSequences,
    int? gpuLayers,
    int? draftGpuLayers,
    int? maxDraftTokens,
    ChatFormat? format,
    SamplingDefaults? sampling,
    String? engineId,
    bool? filterTools,
    bool? enableThinking,
  }) => ModelSpec(
    id: id ?? this.id,
    displayName: displayName ?? this.displayName,
    modelUrl: modelUrl ?? this.modelUrl,
    mmprojUrl: mmprojUrl ?? this.mmprojUrl,
    imageTokenBudget: imageTokenBudget ?? this.imageTokenBudget,
    imageTiling: imageTiling ?? this.imageTiling,
    draftUrl: draftUrl ?? this.draftUrl,
    contextSize: contextSize ?? this.contextSize,
    maxSequences: maxSequences ?? this.maxSequences,
    gpuLayers: gpuLayers ?? this.gpuLayers,
    draftGpuLayers: draftGpuLayers ?? this.draftGpuLayers,
    maxDraftTokens: maxDraftTokens ?? this.maxDraftTokens,
    format: format ?? this.format,
    sampling: sampling ?? this.sampling,
    engineId: engineId ?? this.engineId,
    filterTools: filterTools ?? this.filterTools,
    enableThinking: enableThinking ?? this.enableThinking,
  );

  /// Which inference engine runs this spec. Today only `'llama'` exists.
  ///
  /// When a second engine lands (LiteRT-LM, Apple Foundation Models), this
  /// becomes the dispatch key for an `InferenceEngine` abstraction —
  /// `canRun(spec)` / `createClient(spec)` — yielding a `ChatClient` per
  /// spec. A token-in/token-out engine reuses [format]; a message-native
  /// engine (Apple FM) implements `ChatClient` directly and ignores it.
  /// Deliberately not built yet: with one engine there is nothing to
  /// dispatch.
  final String engineId;

  @override
  bool operator ==(Object other) =>
      other is ModelSpec &&
      other.id == id &&
      other.displayName == displayName &&
      other.modelUrl == modelUrl &&
      other.mmprojUrl == mmprojUrl &&
      other.imageTokenBudget == imageTokenBudget &&
      other.imageTiling == imageTiling &&
      other.draftUrl == draftUrl &&
      other.contextSize == contextSize &&
      other.maxSequences == maxSequences &&
      other.gpuLayers == gpuLayers &&
      other.draftGpuLayers == draftGpuLayers &&
      other.maxDraftTokens == maxDraftTokens &&
      other.format == format &&
      other.sampling == sampling &&
      other.engineId == engineId &&
      other.filterTools == filterTools &&
      other.enableThinking == enableThinking;

  @override
  int get hashCode => Object.hash(
    id,
    displayName,
    modelUrl,
    mmprojUrl,
    imageTokenBudget,
    imageTiling,
    draftUrl,
    contextSize,
    maxSequences,
    gpuLayers,
    draftGpuLayers,
    maxDraftTokens,
    format,
    sampling,
    engineId,
    filterTools,
    enableThinking,
  );

  @override
  String toString() =>
      'ModelSpec(id: $id, displayName: $displayName, '
      'contextSize: $contextSize, maxSequences: $maxSequences, '
      'format: ${format.runtimeType}, engineId: $engineId)';
}
