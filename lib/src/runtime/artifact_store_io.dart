/// Directory-backed managed artifact storage (`dart:io`).
///
/// Artifacts live as plain files under one directory the host app owns —
/// typically `<applicationSupport>/models`, the same layout
/// [ModelDownloader] already writes, so a store pointed at an existing
/// download cache adopts what is there instead of re-downloading it.
///
/// A file only appears under its final name once it is complete: downloads
/// finalize through [ModelDownloader]'s `.part` rename, and imports do the
/// same. An existing name is therefore always a whole artifact.
library;

import 'dart:io';

import 'package:extensions/logging.dart';

import 'artifact_cache_name.dart';
import 'artifact_store_api.dart';
import 'llama_runtime_api.dart';
import 'model_downloader_io.dart';

/// Creates directory-backed managed artifact storage.
///
/// [directory] is required and is where every managed artifact lands.
/// [authToken] is sent as a `Bearer` token on artifact downloads, for gated
/// Hugging Face repositories.
ArtifactStore createArtifactStore({
  String? directory,
  String? authToken,
  LoggerFactory? loggerFactory,
}) {
  if (directory == null || directory.isEmpty) {
    throw ArgumentError.value(
      directory,
      'directory',
      'Managed artifact storage on this platform needs a directory to '
          'store artifacts in (e.g. "<applicationSupport>/models").',
    );
  }
  return FileArtifactStore(
    directory: directory,
    authToken: authToken,
    loggerFactory: loggerFactory,
  );
}

/// Managed artifact storage backed by a directory on the local filesystem.
final class FileArtifactStore implements ArtifactStore {
  /// Creates a store over [directory], creating the directory on first use.
  ///
  /// [downloader] overrides how URL fetches run (auth token, fakes in
  /// tests); by default one is built from [authToken] and [loggerFactory].
  FileArtifactStore({
    required this.directory,
    String? authToken,
    ModelDownloader? downloader,
    LoggerFactory? loggerFactory,
  }) : _downloader =
           downloader ??
           ModelDownloader(authToken: authToken, loggerFactory: loggerFactory),
       _logger = (loggerFactory ?? NullLoggerFactory.instance).createLogger(
         'FileArtifactStore',
       );

  /// Where managed artifacts are stored.
  final String directory;

  final ModelDownloader _downloader;
  final Logger _logger;

  /// Guards import naming: two imports in the same microsecond would
  /// otherwise derive the same key.
  int _importCounter = 0;

  @override
  Future<List<ManagedArtifact>> list() async {
    final dir = Directory(directory);
    if (!dir.existsSync()) return const <ManagedArtifact>[];
    final artifacts = <ManagedArtifact>[];
    await for (final entry in dir.list(followLinks: false)) {
      if (entry is! File) continue;
      final key = _basename(entry.path);
      if (key.endsWith('.part')) continue;
      artifacts.add(
        ManagedArtifact(
          key: key,
          fileName: artifactDisplayName(key),
          sizeBytes: await entry.length(),
        ),
      );
    }
    artifacts.sort((a, b) => a.fileName.compareTo(b.fileName));
    return artifacts;
  }

  @override
  Future<ManagedArtifact?> lookup(String key) async {
    final file = File(_pathFor(key));
    if (!file.existsSync()) return null;
    return ManagedArtifact(
      key: key,
      fileName: artifactDisplayName(key),
      sizeBytes: await file.length(),
    );
  }

  @override
  Future<int> totalSizeBytes() async {
    final dir = Directory(directory);
    if (!dir.existsSync()) return 0;
    var total = 0;
    await for (final entry in dir.list(followLinks: false)) {
      if (entry is File) total += await entry.length();
    }
    return total;
  }

  @override
  Future<ManagedArtifact> fetch(
    Uri url, {
    LlamaLoadProgress? onProgress,
  }) async {
    final path = await _downloader.download(
      url,
      directory: directory,
      onProgress: onProgress,
    );
    final key = _basename(path);
    return ManagedArtifact(
      key: key,
      fileName: artifactDisplayName(key),
      sizeBytes: await File(path).length(),
    );
  }

  @override
  Future<ManagedArtifact> importStream(
    Stream<List<int>> bytes, {
    required String fileName,
    int? totalBytes,
    LlamaLoadProgress? onProgress,
  }) async {
    final key = _importKey(fileName);
    final destination = File(_pathFor(key));
    final part = File('${destination.path}.part');
    await Directory(directory).create(recursive: true);

    var received = 0;
    final sink = part.openWrite();
    try {
      await for (final chunk in bytes) {
        sink.add(chunk);
        received += chunk.length;
        if (totalBytes != null && totalBytes > 0) {
          onProgress?.call((received / totalBytes).clamp(0, 1).toDouble());
        }
      }
    } catch (_) {
      await sink.close();
      if (part.existsSync()) await part.delete();
      rethrow;
    }
    await sink.close();

    if (totalBytes != null && received != totalBytes) {
      await part.delete();
      throw ArtifactStorageException(
        'The import of "$fileName" ended early: got $received of '
        '$totalBytes bytes.',
      );
    }
    await part.rename(destination.path);
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
  }) async {
    final source = File(path);
    if (!source.existsSync()) {
      throw ArtifactStorageException('There is no file at "$path" to import.');
    }
    final fileName = _basename(path);
    await Directory(directory).create(recursive: true);

    if (moveSource) {
      final key = _importKey(fileName);
      try {
        // Same-volume rename: instant, and no second copy of a
        // multi-gigabyte GGUF on the device.
        final moved = await source.rename(_pathFor(key));
        onProgress?.call(1);
        _logger.logInformation('Imported "$fileName" as $key (moved).');
        return ManagedArtifact(
          key: key,
          fileName: artifactDisplayName(key),
          sizeBytes: await moved.length(),
        );
      } on FileSystemException {
        // Across volumes rename fails; fall through to a streamed copy.
        _logger.logDebug(
          'Cross-volume import of "$fileName"; copying instead of moving.',
        );
      }
    }

    return importStream(
      source.openRead(),
      fileName: fileName,
      totalBytes: await source.length(),
      onProgress: onProgress,
    );
  }

  @override
  Future<ManagedArtifactPaths> resolve({
    required String modelKey,
    String? mmprojKey,
    String? draftKey,
  }) async => (
    modelPath: _resolveOne(modelKey, 'model'),
    mmprojPath: mmprojKey == null ? null : _resolveOne(mmprojKey, 'projector'),
    draftPath: draftKey == null ? null : _resolveOne(draftKey, 'draft'),
  );

  @override
  Future<void> delete(String key) async {
    final file = File(_pathFor(key));
    if (file.existsSync()) await file.delete();
    final part = File('${file.path}.part');
    if (part.existsSync()) await part.delete();
    _logger.logInformation('Deleted managed artifact $key.');
  }

  String _resolveOne(String key, String role) {
    final path = _pathFor(key);
    if (!File(path).existsSync()) {
      throw ArtifactStorageException(
        'The $role artifact "$key" is not in managed storage. Download or '
        'import it before loading the model.',
      );
    }
    return path;
  }

  String _pathFor(String key) => '$directory${Platform.pathSeparator}$key';

  String _importKey(String fileName) {
    final nonce = DateTime.now().microsecondsSinceEpoch.toRadixString(16);
    var key = importedArtifactFileName(fileName, nonce: nonce);
    while (File(_pathFor(key)).existsSync()) {
      key = importedArtifactFileName(
        fileName,
        nonce: '$nonce${++_importCounter}',
      );
    }
    return key;
  }

  static String _basename(String path) {
    final slash = path.lastIndexOf(RegExp(r'[/\\]'));
    return slash < 0 ? path : path.substring(slash + 1);
  }
}
