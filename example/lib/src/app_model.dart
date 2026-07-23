import 'package:agents/agents.dart';
import 'package:file_selector/file_selector.dart' show XFile;
import 'package:flutter/foundation.dart';
import 'package:llama_cpp_flutter/llama_cpp_flutter.dart';
import 'package:path_provider/path_provider.dart';

import 'demo_tools.dart';
import 'library/model_catalog.dart';
import 'library/model_entry.dart';
import 'llama_agent_provider.dart';

/// What the agent screen edits. Applied immediately; the transcript is kept.
class AgentConfig {
  String name = 'Assistant';
  String instructions =
      'You are a helpful assistant running fully on-device. '
      'Keep answers concise.';
  double temperature = 0.7;
  int maxTokens = 1024;
  int? topK;
  double? topP;
  bool enableTools = true;
}

/// What the chat screen shows.
enum ModelStatus { empty, loading, ready, error }

/// How one library entry stands right now.
enum EntryState {
  /// Its artifacts are not in managed storage yet.
  notDownloaded,

  /// It is downloading or importing right now.
  transferring,

  /// Its artifacts are stored and it is ready to load.
  stored,

  /// It is the loaded model.
  loaded,

  /// Its last transfer or load failed.
  failed,

  /// An imported artifact is gone from managed storage; re-import it.
  incomplete,
}

/// Owns the model library, managed storage, the loaded session, and the
/// agent built on top of it.
///
/// The runtime, artifact store, and catalog storage are injectable so tests
/// can drive the library without a real engine, filesystem, or browser.
class AppModel extends ChangeNotifier {
  /// Creates the app model. Call [initialize] before use.
  AppModel({
    LlamaRuntime? runtime,
    ArtifactStore? store,
    CatalogStorage? catalogStorage,
  }) : _runtime = runtime,
       _store = store,
       _catalog = ModelCatalog(
         catalogStorage ?? const PreferencesCatalogStorage(),
       );

  final AgentConfig agentConfig = AgentConfig();

  late final LlamaAgentProvider provider = LlamaAgentProvider(
    agentResolver: () => _agent,
  );

  LlamaRuntime? _runtime;
  ArtifactStore? _store;
  final ModelCatalog _catalog;

  LlamaSession? _session;
  AIAgent? _agent;

  final List<ModelEntry> _entries = <ModelEntry>[];
  String? _activeEntryId;
  String? _lastLoadedId;
  String? _transferringEntryId;
  final Map<String, String> _entryErrors = <String, String>{};

  /// Whether [initialize] has finished.
  bool isInitialized = false;

  ModelStatus status = ModelStatus.empty;
  double loadProgress = 0;
  String? errorMessage;

  /// Bytes managed storage occupies, refreshed as artifacts come and go.
  int storageUsedBytes = 0;

  /// Bumped per successful load so the chat view resets with the new model.
  int modelGeneration = 0;

  /// Every model in the library, in the order it was added.
  List<ModelEntry> get library => List<ModelEntry>.unmodifiable(_entries);

  /// The entry backing the loaded session, or null when nothing is loaded.
  ModelEntry? get activeEntry => _entryOrNull(_activeEntryId);

  /// The loaded model's name, for the chat screen's title.
  String? get loadedModelName =>
      status == ModelStatus.ready ? activeEntry?.name : null;

  /// Whether the loaded model can accept images — it has a projector.
  bool get supportsImages =>
      status == ModelStatus.ready && activeEntry?.projector != null;

  /// Whether speculative decoding is available at all; the web runtime
  /// cannot stage a draft model alongside the main one.
  static bool get supportsDraftModels => !kIsWeb;

  /// False on web when the page is not cross-origin isolated, in which case
  /// wllama runs single-threaded (slow enough to look like a hang).
  bool get multiThreaded => _runtime?.supportsMultiThreading ?? true;

  /// Loads the saved library and reconciles it with managed storage.
  Future<void> initialize() async {
    if (isInitialized) return;
    final store = await _ensureStore();
    final snapshot = await _catalog.load();
    _entries
      ..clear()
      ..addAll(snapshot.entries);
    _activeEntryId = snapshot.activeId;

    // Artifacts can disappear from under the catalog — a browser evicting
    // origin-private storage, someone clearing app data. Drop keys that no
    // longer resolve so the entry offers to fetch them again.
    var changed = false;
    for (var index = 0; index < _entries.length; index++) {
      final reconciled = await _reconcile(store, _entries[index]);
      if (reconciled != null) {
        _entries[index] = reconciled;
        changed = true;
      }
    }
    if (changed) await _save();

    storageUsedBytes = await store.totalSizeBytes();
    isInitialized = true;
    notifyListeners();
  }

  /// How [entry] stands right now.
  EntryState stateOf(ModelEntry entry) {
    if (_transferringEntryId == entry.id) return EntryState.transferring;
    if (_entryErrors.containsKey(entry.id)) return EntryState.failed;
    if (entry.hasLostArtifact(includeDraft: supportsDraftModels)) {
      return EntryState.incomplete;
    }
    if (_activeEntryId == entry.id && status == ModelStatus.ready) {
      return EntryState.loaded;
    }
    return entry.isReady(includeDraft: supportsDraftModels)
        ? EntryState.stored
        : EntryState.notDownloaded;
  }

  /// Why [entry]'s last transfer or load failed, or null.
  String? errorFor(ModelEntry entry) => _entryErrors[entry.id];

  /// Adds [entry] to the library without downloading anything.
  Future<void> addEntry(ModelEntry entry) async {
    _entries.add(entry);
    await _save();
    notifyListeners();
  }

  /// Replaces the stored copy of [entry], optionally reloading the session
  /// when it is the loaded model.
  ///
  /// An edit that swaps one artifact for another leaves the old file behind;
  /// it is deleted here unless another entry still references it.
  Future<void> saveEntry(ModelEntry entry, {bool reload = false}) async {
    final index = _entries.indexWhere((other) => other.id == entry.id);
    if (index < 0) return;
    final replaced = <String>[
      for (final artifact in _entries[index].artifacts) ?artifact.key,
    ];
    _entries[index] = entry;
    _entryErrors.remove(entry.id);
    await _save();
    notifyListeners();
    await _deleteUnreferenced(replaced);
    if (reload) await loadEntry(entry.id);
  }

  /// Deletes any of [keys] the library no longer references.
  ///
  /// Importing a file puts it in managed storage immediately, so an import
  /// the user then abandons — by backing out of the editor, or by picking a
  /// different file — would otherwise occupy space nothing can reclaim.
  Future<void> discardUnreferencedArtifacts(Iterable<String> keys) =>
      _deleteUnreferenced(keys);

  /// Permanently deletes [id]: unloads it when it is live, forgets it, and
  /// deletes every artifact no other entry still references.
  Future<void> deleteEntry(String id) async {
    final entry = _entryOrNull(id);
    if (entry == null) return;
    if (_activeEntryId == id) await _unload();
    _entries.removeWhere((other) => other.id == id);
    _entryErrors.remove(id);
    await _save();
    notifyListeners();
    await _deleteUnreferenced(<String>[
      for (final artifact in entry.artifacts) ?artifact.key,
    ]);
  }

  /// Deletes every key in [candidates] that no entry references any more.
  ///
  /// Reference counts are derived from the catalog rather than stored, so
  /// they cannot drift out of sync with it.
  Future<void> _deleteUnreferenced(Iterable<String> candidates) async {
    final unique = candidates.toSet();
    if (unique.isEmpty) return;
    final referenced = <String>{
      for (final entry in _entries)
        for (final artifact in entry.artifacts) ?artifact.key,
    };
    final store = await _ensureStore();
    for (final key in unique.difference(referenced)) {
      await store.delete(key);
    }
    storageUsedBytes = await store.totalSizeBytes();
    notifyListeners();
  }

  /// Retries the most recent load, after a failure.
  Future<void> retryLastLoad() async {
    if (_lastLoadedId case final id?) await loadEntry(id);
  }

  /// Downloads whatever [id] is missing, then loads it into the session.
  Future<void> loadEntry(String id) async {
    if (status == ModelStatus.loading) return;
    var entry = _entryOrNull(id);
    if (entry == null) return;
    _lastLoadedId = id;

    status = ModelStatus.loading;
    loadProgress = 0;
    errorMessage = null;
    _entryErrors.remove(id);
    _transferringEntryId = id;
    notifyListeners();

    try {
      final store = await _ensureStore();
      entry = await _transferArtifacts(store, entry);
      _transferringEntryId = null;
      notifyListeners();

      // Free the old model before allocating the new one: two multi-gigabyte
      // models will not fit on a phone at once.
      final previous = _session;
      _session = null;
      _agent = null;
      await previous?.dispose();

      final paths = await store.resolve(
        modelKey: entry.main.key!,
        mmprojKey: entry.projector?.key,
        draftKey: supportsDraftModels ? entry.draft?.key : null,
      );
      final session = await (await _ensureRuntime()).loadModel(
        entry.toSpec(includeDraft: supportsDraftModels),
        localPath: paths.modelPath,
        localMmprojPath: paths.mmprojPath,
        localDraftPath: paths.draftPath,
        onProgress: (progress) {
          loadProgress = progress;
          notifyListeners();
        },
      );
      _session = session;
      _activeEntryId = entry.id;
      _rebuildAgent(entry);
      provider.clear();
      modelGeneration++;
      status = ModelStatus.ready;
      await _save();
    } catch (error) {
      errorMessage = '$error';
      _entryErrors[id] = '$error';
      status = ModelStatus.error;
    }
    _transferringEntryId = null;
    storageUsedBytes = await (await _ensureStore()).totalSizeBytes();
    notifyListeners();
  }

  /// Imports [file] into managed storage and returns a reference to it.
  ///
  /// Copies rather than moves: the file belongs to the user, and on macOS
  /// the picker hands back its real location.
  Future<ArtifactRef> importArtifact(
    XFile file, {
    LlamaLoadProgress? onProgress,
  }) async {
    final store = await _ensureStore();
    final artifact = kIsWeb
        ? await store.importStream(
            file.openRead(),
            fileName: file.name,
            totalBytes: await file.length(),
            onProgress: onProgress,
          )
        : await store.importFile(file.path, onProgress: onProgress);
    storageUsedBytes = await store.totalSizeBytes();
    notifyListeners();
    return ArtifactRef(
      fileName: artifact.fileName,
      key: artifact.key,
      sizeBytes: artifact.sizeBytes,
    );
  }

  /// Applies agent-screen changes to the live agent; the transcript is kept.
  void applyAgentConfig() {
    final entry = activeEntry;
    if (_session != null && entry != null) _rebuildAgent(entry);
    notifyListeners();
  }

  /// Fetches every artifact of [entry] that is not stored yet, recording
  /// each one's key as it lands so a later run does not refetch it.
  Future<ModelEntry> _transferArtifacts(
    ArtifactStore store,
    ModelEntry entry,
  ) async {
    var updated = entry;
    // Only the weights report progress; projectors and drafters are small
    // enough that a second progress bar would be noise.
    updated = updated.copyWith(
      main: await _transfer(
        store,
        updated.main,
        'weights',
        onProgress: (progress) {
          loadProgress = progress;
          notifyListeners();
        },
      ),
    );
    if (updated.projector case final projector?) {
      updated = updated.copyWith(
        projector: await _transfer(store, projector, 'projector'),
      );
    }
    if (supportsDraftModels && updated.draft != null) {
      updated = updated.copyWith(
        draft: await _transfer(store, updated.draft!, 'draft model'),
      );
    }
    if (updated.main.key != entry.main.key ||
        updated.projector?.key != entry.projector?.key ||
        updated.draft?.key != entry.draft?.key) {
      final index = _entries.indexWhere((other) => other.id == entry.id);
      if (index >= 0) _entries[index] = updated;
      await _save();
    }
    return updated;
  }

  Future<ArtifactRef> _transfer(
    ArtifactStore store,
    ArtifactRef artifact,
    String role, {
    LlamaLoadProgress? onProgress,
  }) async {
    if (artifact.isStored) return artifact;
    final url = artifact.url;
    if (url == null) {
      throw StateError(
        'The $role file "${artifact.fileName}" is no longer in storage. '
        'Import it again from the model\'s settings.',
      );
    }
    final stored = await store.fetch(url, onProgress: onProgress);
    return artifact.copyWith(key: stored.key, sizeBytes: stored.sizeBytes);
  }

  /// [entry] with any artifact that vanished from [store] unstored again,
  /// or null when nothing changed.
  Future<ModelEntry?> _reconcile(ArtifactStore store, ModelEntry entry) async {
    Future<ArtifactRef?> check(ArtifactRef? artifact) async {
      if (artifact?.key case final key?) {
        if (await store.lookup(key) == null) {
          return artifact!.copyWith(clearKey: true);
        }
      }
      return null;
    }

    final main = await check(entry.main);
    final projector = await check(entry.projector);
    final draft = await check(entry.draft);
    if (main == null && projector == null && draft == null) return null;
    return entry.copyWith(
      main: main ?? entry.main,
      projector: projector ?? entry.projector,
      draft: draft ?? entry.draft,
    );
  }

  Future<void> _unload() async {
    final previous = _session;
    _session = null;
    _agent = null;
    _activeEntryId = null;
    status = ModelStatus.empty;
    loadProgress = 0;
    await previous?.dispose();
    provider.clear();
    modelGeneration++;
  }

  void _rebuildAgent(ModelEntry entry) {
    final config = agentConfig;
    final chatClient = createLlamaChatClient(
      spec: entry.toSpec(includeDraft: supportsDraftModels),
      sessionProvider: () async => _session!,
      samplingOverride: SamplingDefaults(
        maxTokens: config.maxTokens,
        temperature: config.temperature,
        topK: config.topK,
        topP: config.topP,
      ),
      isThinkingEnabled: () => entry.enableThinking,
    );
    _agent = ChatClientAgent.withSettings(
      chatClient,
      name: config.name,
      instructions: config.instructions.trim().isEmpty
          ? null
          : config.instructions.trim(),
      tools: config.enableTools ? buildDemoTools() : null,
    );
  }

  ModelEntry? _entryOrNull(String? id) {
    if (id == null) return null;
    for (final entry in _entries) {
      if (entry.id == id) return entry;
    }
    return null;
  }

  Future<void> _save() => _catalog.save(
    _entries,
    status == ModelStatus.ready ? _activeEntryId : null,
  );

  Future<LlamaRuntime> _ensureRuntime() async =>
      _runtime ??= createLlamaRuntime();

  Future<ArtifactStore> _ensureStore() async {
    if (_store case final store?) return store;
    String? directory;
    if (!kIsWeb) {
      final support = await getApplicationSupportDirectory();
      directory = '${support.path}/models';
    }
    return _store = createArtifactStore(directory: directory);
  }

  @override
  void dispose() {
    _session?.dispose();
    provider.dispose();
    super.dispose();
  }
}
