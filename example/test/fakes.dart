import 'dart:async';
import 'dart:typed_data';

import 'package:llama_cpp_flutter/gguf.dart';
import 'package:llama_cpp_flutter/llama_cpp_flutter.dart';

/// Managed storage backed by a map, so library tests need no filesystem,
/// browser, or network.
final class FakeArtifactStore implements ArtifactStore {
  final Map<String, ManagedArtifact> artifacts = <String, ManagedArtifact>{};

  /// URLs passed to [fetch], in order — including repeats, so a test can
  /// prove a stored artifact is not downloaded twice.
  final List<Uri> fetched = <Uri>[];

  /// Keys passed to [delete], in order.
  final List<String> deleted = <String>[];

  /// Bytes reported for anything fetched or imported.
  int artifactSize = 100;

  int _importCounter = 0;

  @override
  Future<List<ManagedArtifact>> list() async => artifacts.values.toList();

  @override
  Future<ManagedArtifact?> lookup(String key) async => artifacts[key];

  @override
  Future<int> totalSizeBytes() async => artifacts.values.fold<int>(
    0,
    (total, artifact) => total + artifact.sizeBytes,
  );

  @override
  Future<ManagedArtifact> fetch(
    Uri url, {
    LlamaLoadProgress? onProgress,
  }) async {
    fetched.add(url);
    final key = stableArtifactFileName(url);
    onProgress?.call(1);
    return artifacts[key] ??= ManagedArtifact(
      key: key,
      fileName: artifactDisplayName(key),
      sizeBytes: artifactSize,
    );
  }

  @override
  Future<ManagedArtifact> importStream(
    Stream<List<int>> bytes, {
    required String fileName,
    int? totalBytes,
    LlamaLoadProgress? onProgress,
  }) async {
    var received = 0;
    await for (final chunk in bytes) {
      received += chunk.length;
    }
    onProgress?.call(1);
    return _store(fileName, received == 0 ? artifactSize : received);
  }

  @override
  Future<ManagedArtifact> importFile(
    String path, {
    bool moveSource = false,
    LlamaLoadProgress? onProgress,
  }) async {
    onProgress?.call(1);
    return _store(path.split('/').last, artifactSize);
  }

  @override
  Future<ManagedArtifactPaths> resolve({
    required String modelKey,
    String? mmprojKey,
    String? draftKey,
  }) async => (
    modelPath: _resolveOne(modelKey),
    mmprojPath: mmprojKey == null ? null : _resolveOne(mmprojKey),
    draftPath: draftKey == null ? null : _resolveOne(draftKey),
  );

  @override
  Future<void> delete(String key) async {
    deleted.add(key);
    artifacts.remove(key);
  }

  ManagedArtifact _store(String fileName, int sizeBytes) {
    final key = importedArtifactFileName(
      fileName,
      nonce: '${_importCounter++}',
    );
    return artifacts[key] = ManagedArtifact(
      key: key,
      fileName: fileName,
      sizeBytes: sizeBytes,
    );
  }

  String _resolveOne(String key) {
    if (!artifacts.containsKey(key)) {
      throw ArtifactStorageException('"$key" is not in managed storage.');
    }
    return '/managed/$key';
  }
}

/// A runtime that records what it was asked to load.
final class FakeLlamaRuntime implements LlamaRuntime {
  /// Every `loadModel` call, in order.
  final List<({ModelSpec spec, String? model, String? mmproj, String? draft})>
  loads = [];

  /// Set to make the next load fail.
  Object? failWith;

  /// Sessions handed out, so a test can assert one was disposed.
  final List<FakeLlamaSession> sessions = <FakeLlamaSession>[];

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
    loads.add((
      spec: spec,
      model: localPath,
      mmproj: localMmprojPath,
      draft: localDraftPath,
    ));
    if (failWith case final error?) {
      failWith = null;
      throw error;
    }
    onProgress?.call(1);
    final session = FakeLlamaSession();
    sessions.add(session);
    return session;
  }
}

/// A session that does nothing but remember that it was disposed.
final class FakeLlamaSession implements LlamaSession {
  bool disposed = false;

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
  }) => const Stream<String>.empty();

  @override
  Future<void> cancel() async {}

  @override
  LlamaSessionCapabilities get capabilities => const LlamaSessionCapabilities(
    canPersistState: false,
    reportsStateSize: false,
  );

  @override
  Future<int> saveState(String path, {int sequenceId = 0}) async => 0;

  @override
  Future<int> loadState(String path, {int sequenceId = 0}) async => 0;

  @override
  Future<int> stateSizeBytes({int sequenceId = 0}) async => 0;

  @override
  Future<LlamaStashResult> stashState(String key, {int sequenceId = 0}) async =>
      (tokens: 0, bytes: 0);

  @override
  Future<int> restoreStashedState(String key, {int sequenceId = 0}) async => 0;

  @override
  Future<int> dropStashedState(String key) async => 0;

  @override
  Future<void> clearSequence(int sequenceId) async {}

  @override
  Future<void> setImageTokenBudget(int? imageTokenBudget) async {}

  @override
  Future<void> dispose() async => disposed = true;
}
