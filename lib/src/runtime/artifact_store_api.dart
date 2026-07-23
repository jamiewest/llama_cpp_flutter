/// Managed storage for model artifacts — the app-visible alternative to an
/// engine's opaque internal cache.
///
/// An [ArtifactStore] is one place a host app downloads into, imports into,
/// enumerates, sizes, and deletes from, with the same API on every platform:
/// a directory under the app's own storage on iOS/macOS, an OPFS directory
/// in the browser. [ArtifactStore.resolve] turns stored artifacts into the
/// `localPath` / `localMmprojPath` / `localDraftPath` arguments
/// `LlamaRuntime.loadModel` takes, so a model library UI can own the
/// lifecycle of every byte it puts on the user's device.
library;

import 'llama_runtime_api.dart';

/// One artifact held in an [ArtifactStore].
final class ManagedArtifact {
  /// Creates a managed artifact record.
  const ManagedArtifact({
    required this.key,
    required this.fileName,
    required this.sizeBytes,
  });

  /// Stable identifier within the store, and the name it is stored under.
  ///
  /// For artifacts fetched from a URL this is `stableArtifactFileName(url)`,
  /// so a caller holding the URL can always recompute the key rather than
  /// persisting a mapping.
  final String key;

  /// The artifact's original file name, for display.
  final String fileName;

  /// Bytes the artifact occupies in managed storage.
  final int sizeBytes;

  @override
  bool operator ==(Object other) =>
      other is ManagedArtifact &&
      other.key == key &&
      other.fileName == fileName &&
      other.sizeBytes == sizeBytes;

  @override
  int get hashCode => Object.hash(key, fileName, sizeBytes);

  @override
  String toString() =>
      'ManagedArtifact(key: $key, fileName: $fileName, '
      'sizeBytes: $sizeBytes)';
}

/// Load-ready locations of one model's artifacts, shaped for
/// `LlamaRuntime.loadModel`.
///
/// On native platforms these are filesystem paths. On the web they are
/// opaque handles the web runtime resolves against managed storage — pass
/// them to `loadModel` unmodified; they are not URLs and nothing else can
/// dereference them.
typedef ManagedArtifactPaths = ({
  String modelPath,
  String? mmprojPath,
  String? draftPath,
});

/// Thrown when managed storage cannot satisfy a request.
///
/// Underlying platform failures (a failed HTTP download, a filesystem
/// error) propagate as themselves; this type covers store-level conditions
/// the caller can act on, chiefly a missing artifact and a full device.
final class ArtifactStorageException implements Exception {
  /// Creates a storage exception.
  const ArtifactStorageException(this.message, {this.isQuotaExceeded = false});

  /// What went wrong, phrased for a user-facing error surface.
  final String message;

  /// Whether the operation failed because storage is full or the browser's
  /// quota is exhausted — the caller's cue to suggest deleting artifacts.
  final bool isQuotaExceeded;

  @override
  String toString() => 'ArtifactStorageException: $message';
}

/// App-managed storage for model artifacts (weights, projectors, drafters).
///
/// Every method is safe to call concurrently with generation; none of them
/// touch a loaded session. Deleting an artifact a live session loaded is the
/// caller's mistake to avoid — unload first.
abstract interface class ArtifactStore {
  /// Every complete artifact currently stored.
  ///
  /// Partially transferred artifacts are omitted; their bytes still count
  /// toward [totalSizeBytes] until they complete or are deleted.
  Future<List<ManagedArtifact>> list();

  /// The stored artifact under [key], or null when absent or incomplete.
  Future<ManagedArtifact?> lookup(String key);

  /// Bytes managed storage occupies, including partial transfers.
  Future<int> totalSizeBytes();

  /// Ensures [url] is stored, downloading it when absent, and returns it.
  ///
  /// Returns immediately when the artifact is already complete. An
  /// interrupted transfer resumes where it stopped when the server supports
  /// it, and a partial file is never reported as complete. [onProgress]
  /// reports fractions of the whole artifact.
  Future<ManagedArtifact> fetch(Uri url, {LlamaLoadProgress? onProgress});

  /// Imports [bytes] into managed storage under a name derived from
  /// [fileName], and returns the stored artifact.
  ///
  /// [totalBytes] enables progress reporting and a completeness check; pass
  /// it whenever the source size is known. This is the portable import
  /// path — it works on the web, where [importFile] does not.
  Future<ManagedArtifact> importStream(
    Stream<List<int>> bytes, {
    required String fileName,
    int? totalBytes,
    LlamaLoadProgress? onProgress,
  });

  /// Imports the file at [path] into managed storage (native platforms
  /// only; throws [UnsupportedError] on the web).
  ///
  /// Copies by default, which is the only safe behavior when [path] is a
  /// file the user owns — a document picker on macOS hands back the real
  /// path, and moving it would take the file out of the user's folder. Set
  /// [moveSource] only when the caller owns [path] (an iOS picker's
  /// temporary copy, say): the file is then renamed into place when it sits
  /// on the same volume, avoiding a multi-gigabyte duplicate, and copied
  /// otherwise.
  Future<ManagedArtifact> importFile(
    String path, {
    bool moveSource = false,
    LlamaLoadProgress? onProgress,
  });

  /// Resolves stored artifacts into the arguments
  /// `LlamaRuntime.loadModel` takes.
  ///
  /// Throws [ArtifactStorageException] when any requested key is missing or
  /// incomplete, so a load never starts against a half-downloaded file.
  Future<ManagedArtifactPaths> resolve({
    required String modelKey,
    String? mmprojKey,
    String? draftKey,
  });

  /// Permanently deletes the artifact under [key], along with any partial
  /// transfer of it. A no-op when nothing is stored under [key].
  Future<void> delete(String key);
}
