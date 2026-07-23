/// OPFS-backed managed artifact storage for Flutter web.
///
/// Artifacts land in the origin-private file system, in the same directory
/// and under the same names the web runtime uses for its own downloads, so
/// the two see one shared set of files.
///
/// [ArtifactStore.resolve] returns opaque handles rather than blob URLs.
/// That matters for large models: the runtime resolves a handle straight to
/// the disk-backed OPFS file and splits it into stageable parts, where a
/// blob URL would have to be fetched back through the network stack first —
/// materializing gigabytes the origin-private file system exists to keep on
/// disk.
library;

import 'package:extensions/logging.dart';
import 'package:web/web.dart' as web;

import 'artifact_cache_name.dart';
import 'artifact_store_api.dart';
import 'llama_runtime_api.dart';
import 'opfs_artifacts.dart';

/// Creates OPFS-backed managed artifact storage.
///
/// [directory] is ignored: the browser decides where origin-private storage
/// lives. [authToken] is sent as a `Bearer` token on artifact downloads, for
/// gated Hugging Face repositories.
ArtifactStore createArtifactStore({
  String? directory,
  String? authToken,
  LoggerFactory? loggerFactory,
}) => OpfsArtifactStore(authToken: authToken, loggerFactory: loggerFactory);

/// Managed artifact storage backed by the origin-private file system.
final class OpfsArtifactStore implements ArtifactStore {
  /// Creates the store. Nothing is written until an artifact is fetched or
  /// imported.
  OpfsArtifactStore({String? authToken, LoggerFactory? loggerFactory})
    : _authToken = authToken,
      _logger = (loggerFactory ?? NullLoggerFactory.instance).createLogger(
        'OpfsArtifactStore',
      );

  final String? _authToken;
  final Logger _logger;

  /// Guards import naming: two imports in the same microsecond would
  /// otherwise derive the same key.
  int _importCounter = 0;

  @override
  Future<List<ManagedArtifact>> list() async {
    final directory = await openArtifactDirectory();
    if (directory == null) return const <ManagedArtifact>[];
    final names = await artifactEntryNames(directory);
    final entries = names.toSet();
    final artifacts = <ManagedArtifact>[];
    for (final name in names) {
      // An in-flight marker, as opposed to an artifact whose own name ends
      // in `.part`, only exists beside the artifact it marks.
      if (name.endsWith('.part') &&
          entries.contains(name.substring(0, name.length - '.part'.length))) {
        continue;
      }
      final record = await artifactRecord(directory, name);
      if (record != null) artifacts.add(record);
    }
    artifacts.sort((a, b) => a.fileName.compareTo(b.fileName));
    return artifacts;
  }

  @override
  Future<ManagedArtifact?> lookup(String key) async {
    final directory = await openArtifactDirectory();
    if (directory == null) return null;
    return artifactRecord(directory, key);
  }

  @override
  Future<int> totalSizeBytes() async {
    final directory = await openArtifactDirectory();
    if (directory == null) return 0;
    var total = 0;
    for (final name in await artifactEntryNames(directory)) {
      final file = await artifactFile(directory, name);
      if (file != null) total += file.size;
    }
    return total;
  }

  @override
  Future<ManagedArtifact> fetch(
    Uri url, {
    LlamaLoadProgress? onProgress,
  }) async {
    final directory = await requireArtifactDirectory();
    final key = stableArtifactFileName(url);

    // A stored copy with no in-flight marker is either complete or a
    // leftover from a version that predates the marker; one HEAD settles
    // which. Fresh downloads skip it — the GET discloses the same length.
    int? contentLength;
    if (await artifactFile(directory, key) != null &&
        !await artifactIsPartial(directory, key)) {
      contentLength = await artifactContentLength(url, authToken: _authToken);
    }

    final file = await ensureArtifactFromUrl(
      directory,
      url,
      contentLength: contentLength,
      authToken: _authToken,
      onProgress: onProgress,
    );
    return ManagedArtifact(
      key: key,
      fileName: artifactDisplayName(key),
      sizeBytes: file.size,
    );
  }

  @override
  Future<ManagedArtifact> importStream(
    Stream<List<int>> bytes, {
    required String fileName,
    int? totalBytes,
    LlamaLoadProgress? onProgress,
  }) async {
    final directory = await requireArtifactDirectory();
    final key = await _importKey(directory, fileName);
    final received = await writeArtifactStream(
      directory,
      key,
      bytes,
      totalBytes: totalBytes,
      onProgress: onProgress,
    );
    _logger.logInformation('Imported "$fileName" ($received bytes) as $key.');
    return ManagedArtifact(
      key: key,
      fileName: artifactDisplayName(key),
      sizeBytes: received,
    );
  }

  @override
  Future<ManagedArtifact> importFile(
    String path, {
    bool moveSource = false,
    LlamaLoadProgress? onProgress,
  }) => throw UnsupportedError(
    'The web has no filesystem paths to import from; read the picked file '
    'and call importStream instead.',
  );

  @override
  Future<ManagedArtifactPaths> resolve({
    required String modelKey,
    String? mmprojKey,
    String? draftKey,
  }) async {
    final directory = await requireArtifactDirectory();
    return (
      modelPath: await _resolveOne(directory, modelKey, 'model'),
      mmprojPath: mmprojKey == null
          ? null
          : await _resolveOne(directory, mmprojKey, 'projector'),
      draftPath: draftKey == null
          ? null
          : await _resolveOne(directory, draftKey, 'draft'),
    );
  }

  @override
  Future<void> delete(String key) async {
    final directory = await openArtifactDirectory();
    if (directory == null) return;
    await removeArtifact(directory, key);
    _logger.logInformation('Deleted managed artifact $key.');
  }

  Future<String> _resolveOne(
    web.FileSystemDirectoryHandle directory,
    String key,
    String role,
  ) async {
    if (await artifactRecord(directory, key) == null) {
      throw ArtifactStorageException(
        'The $role artifact "$key" is not in managed storage. Download or '
        'import it before loading the model.',
      );
    }
    return artifactHandle(key);
  }

  Future<String> _importKey(
    web.FileSystemDirectoryHandle directory,
    String fileName,
  ) async {
    final nonce = DateTime.now().microsecondsSinceEpoch.toRadixString(16);
    var key = importedArtifactFileName(fileName, nonce: nonce);
    while (await artifactFile(directory, key) != null) {
      key = importedArtifactFileName(
        fileName,
        nonce: '$nonce${++_importCounter}',
      );
    }
    return key;
  }
}
