import 'package:llama_cpp_flutter/llama_cpp_flutter.dart';

/// An identifier for a new library entry, unique within one library.
String newEntryId() => DateTime.now().microsecondsSinceEpoch.toRadixString(36);

/// One artifact slot of a [ModelEntry]: where the file came from, and where
/// it lives in managed storage once it has been transferred there.
class ArtifactRef {
  /// Creates a reference to an artifact.
  const ArtifactRef({
    required this.fileName,
    this.url,
    this.key,
    this.sizeBytes = 0,
  });

  /// Reads a reference previously written by [toJson].
  factory ArtifactRef.fromJson(Map<String, Object?> json) => ArtifactRef(
    fileName: json['fileName'] as String? ?? 'model.gguf',
    url: json['url'] == null ? null : Uri.parse(json['url']! as String),
    key: json['key'] as String?,
    sizeBytes: json['sizeBytes'] as int? ?? 0,
  );

  /// The artifact's file name, for display.
  final String fileName;

  /// Where the artifact is downloaded from, or null when it was imported
  /// from a local file and has no remote source.
  final Uri? url;

  /// The managed-storage key, or null while the artifact is not stored.
  final String? key;

  /// Bytes the stored artifact occupies; 0 until it is stored.
  final int sizeBytes;

  /// Whether the artifact is in managed storage and ready to load.
  bool get isStored => key != null;

  /// Whether the artifact can never be transferred again on its own: it was
  /// imported from a local file, and that copy is gone.
  bool get isLost => key == null && url == null;

  /// Returns a copy with the given fields replaced.
  ArtifactRef copyWith({
    String? fileName,
    Uri? url,
    String? key,
    int? sizeBytes,
    bool clearKey = false,
  }) => ArtifactRef(
    fileName: fileName ?? this.fileName,
    url: url ?? this.url,
    key: clearKey ? null : key ?? this.key,
    sizeBytes: clearKey ? 0 : sizeBytes ?? this.sizeBytes,
  );

  /// This reference as JSON.
  Map<String, Object?> toJson() => <String, Object?>{
    'fileName': fileName,
    if (url != null) 'url': url.toString(),
    if (key != null) 'key': key,
    'sizeBytes': sizeBytes,
  };
}

/// A model in the user's library: its artifacts and how to run them.
///
/// Entries are immutable; the editor screen builds a modified copy through
/// [copyWith] and hands it back to `AppModel` to persist.
class ModelEntry {
  /// Creates a library entry.
  const ModelEntry({
    required this.id,
    required this.name,
    required this.main,
    this.projector,
    this.draft,
    this.formatName = 'chatml',
    this.contextSize = 4096,
    this.gpuOffload = true,
    this.enableThinking = true,
  });

  /// Reads an entry previously written by [toJson].
  factory ModelEntry.fromJson(Map<String, Object?> json) => ModelEntry(
    id: json['id']! as String,
    name: json['name'] as String? ?? 'Model',
    main: ArtifactRef.fromJson(json['main']! as Map<String, Object?>),
    projector: json['projector'] == null
        ? null
        : ArtifactRef.fromJson(json['projector']! as Map<String, Object?>),
    draft: json['draft'] == null
        ? null
        : ArtifactRef.fromJson(json['draft']! as Map<String, Object?>),
    formatName: json['formatName'] as String? ?? 'chatml',
    contextSize: json['contextSize'] as int? ?? 4096,
    gpuOffload: json['gpuOffload'] as bool? ?? true,
    enableThinking: json['enableThinking'] as bool? ?? true,
  );

  /// Stable identifier, unique within the library.
  final String id;

  /// User-editable display name.
  final String name;

  /// The GGUF weights. Every entry has one.
  final ArtifactRef main;

  /// The multimodal projector (mmproj) that gives this model vision, or
  /// null for a text-only entry.
  final ArtifactRef? projector;

  /// The speculative-decoding draft model, or null to decode single-model.
  /// Native only — the web runtime cannot stage a second GGUF.
  final ArtifactRef? draft;

  /// Name of the chat format this model speaks.
  final String formatName;

  /// Context window to allocate at load.
  final int contextSize;

  /// Whether to offload layers to the GPU.
  final bool gpuOffload;

  /// Whether to let the model use its reasoning channel, where it has one.
  final bool enableThinking;

  /// Every artifact this entry references, in slot order.
  List<ArtifactRef> get artifacts => <ArtifactRef>[main, ?projector, ?draft];

  /// Whether every artifact this platform will actually load is in managed
  /// storage, so the entry can load without transferring anything first.
  ///
  /// [includeDraft] is false on the web, which never loads a draft model —
  /// an entry that carries one is still ready there once its weights and
  /// projector are stored.
  bool isReady({bool includeDraft = true}) =>
      _relevant(includeDraft: includeDraft).every((a) => a.isStored);

  /// Whether an imported artifact this platform needs went missing from
  /// managed storage; the entry cannot load until it is re-imported.
  bool hasLostArtifact({bool includeDraft = true}) =>
      _relevant(includeDraft: includeDraft).any((a) => a.isLost);

  List<ArtifactRef> _relevant({required bool includeDraft}) => <ArtifactRef>[
    main,
    ?projector,
    if (includeDraft) ?draft,
  ];

  /// Bytes this entry's stored artifacts occupy.
  int get storedSizeBytes =>
      artifacts.fold(0, (total, artifact) => total + artifact.sizeBytes);

  /// Returns a copy with the given fields replaced.
  ///
  /// [clearProjector] and [clearDraft] remove those artifacts, which
  /// passing null cannot express.
  ModelEntry copyWith({
    String? name,
    ArtifactRef? main,
    ArtifactRef? projector,
    ArtifactRef? draft,
    String? formatName,
    int? contextSize,
    bool? gpuOffload,
    bool? enableThinking,
    bool clearProjector = false,
    bool clearDraft = false,
  }) => ModelEntry(
    id: id,
    name: name ?? this.name,
    main: main ?? this.main,
    projector: clearProjector ? null : projector ?? this.projector,
    draft: clearDraft ? null : draft ?? this.draft,
    formatName: formatName ?? this.formatName,
    contextSize: contextSize ?? this.contextSize,
    gpuOffload: gpuOffload ?? this.gpuOffload,
    enableThinking: enableThinking ?? this.enableThinking,
  );

  /// The runtime spec for this entry.
  ///
  /// Artifacts always reach `loadModel` as managed-storage paths, so the
  /// spec's URLs are never fetched; an imported artifact with no remote
  /// source gets a placeholder standing in for the required `modelUrl`.
  /// [includeDraft] is false on the web, whose runtime rejects a spec that
  /// declares a draft model at all.
  ModelSpec toSpec({bool includeDraft = true}) => ModelSpec(
    id: id,
    displayName: name,
    modelUrl: main.url ?? Uri.parse('managed:${main.key ?? main.fileName}'),
    mmprojUrl: projector?.url,
    draftUrl: includeDraft ? draft?.url : null,
    contextSize: contextSize,
    gpuLayers: gpuOffload ? 999 : 0,
    format: resolveChatFormat(formatName) ?? resolveChatFormat('chatml')!,
    enableThinking: enableThinking,
  );

  /// This entry as JSON.
  Map<String, Object?> toJson() => <String, Object?>{
    'id': id,
    'name': name,
    'main': main.toJson(),
    if (projector case final projector?) 'projector': projector.toJson(),
    if (draft case final draft?) 'draft': draft.toJson(),
    'formatName': formatName,
    'contextSize': contextSize,
    'gpuOffload': gpuOffload,
    'enableThinking': enableThinking,
  };
}
