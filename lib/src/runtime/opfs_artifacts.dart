/// OPFS primitives shared by the web artifact store and the web runtime.
///
/// Both put model artifacts in the same origin-private directory under the
/// same names, so a model the runtime downloaded on its own is visible to —
/// and deletable through — [ArtifactStore], and vice versa.
///
/// ## Completeness
///
/// Data is written straight into `<key>`, with a zero-byte `<key>.part`
/// alongside it for as long as the transfer is in flight. An artifact is
/// complete when `<key>` exists and `<key>.part` does not, which mirrors the
/// native store's `.part`-rename semantics without needing
/// `FileSystemFileHandle.move` (Chromium-only). A file left behind by an
/// interrupted transfer keeps its bytes and resumes from its current size.
library;

import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'dart:typed_data';

import 'package:web/web.dart' as web;

import 'artifact_cache_name.dart';
import 'artifact_store_api.dart';
import 'llama_runtime_api.dart';

/// OPFS directory holding managed model artifacts.
///
/// The name predates managed storage — it was the web runtime's cache for
/// models too large to stage in one piece — and is kept so artifacts users
/// already downloaded stay where the runtime left them.
const String artifactDirectoryName = 'llama_cpp_flutter_large_models';

/// Marks a `LlamaRuntime.loadModel` path argument as a managed-storage
/// handle rather than a blob URL.
const String artifactHandlePrefix = 'llama-artifact:';

/// Suffix of the zero-byte sibling marking a transfer as in flight.
const String _partSuffix = '.part';

/// The `loadModel` handle for the artifact stored under [key].
String artifactHandle(String key) => '$artifactHandlePrefix$key';

/// The artifact key inside [path], or null when [path] is not a handle.
String? artifactKeyOf(String path) => path.startsWith(artifactHandlePrefix)
    ? path.substring(artifactHandlePrefix.length)
    : null;

/// Opens the managed-artifact directory, or returns null when OPFS is
/// unavailable (an old browser, or a context without storage access).
Future<web.FileSystemDirectoryHandle?> openArtifactDirectory() async {
  try {
    final opfs = await web.window.navigator.storage.getDirectory().toDart;
    return await opfs
        .getDirectoryHandle(
          artifactDirectoryName,
          web.FileSystemGetDirectoryOptions(create: true),
        )
        .toDart;
  } catch (_) {
    return null;
  }
}

/// The managed-artifact directory, or a throwing failure when OPFS is
/// unavailable.
Future<web.FileSystemDirectoryHandle> requireArtifactDirectory() async {
  final directory = await openArtifactDirectory();
  if (directory == null) {
    throw const ArtifactStorageException(
      'This browser has no origin-private file system, so model artifacts '
      'cannot be stored. Use a browser with OPFS support.',
    );
  }
  return directory;
}

/// The stored file under [key], or null when it does not exist.
Future<web.File?> artifactFile(
  web.FileSystemDirectoryHandle directory,
  String key,
) async {
  try {
    final handle = await directory.getFileHandle(key).toDart;
    return await handle.getFile().toDart;
  } catch (_) {
    return null;
  }
}

/// Whether a transfer of [key] is in flight or was interrupted.
Future<bool> artifactIsPartial(
  web.FileSystemDirectoryHandle directory,
  String key,
) async => await artifactFile(directory, '$key$_partSuffix') != null;

/// The complete artifact stored under [key], or null when it is missing or
/// only partially transferred.
Future<ManagedArtifact?> artifactRecord(
  web.FileSystemDirectoryHandle directory,
  String key,
) async {
  final file = await artifactFile(directory, key);
  if (file == null) return null;
  if (await artifactIsPartial(directory, key)) return null;
  return ManagedArtifact(
    key: key,
    fileName: artifactDisplayName(key),
    sizeBytes: file.size,
  );
}

/// The names of every entry in [directory], files and directories alike.
///
/// `FileSystemDirectoryHandle` exposes its entries through a JS async
/// iterator, which `package:web` does not type; the protocol is driven by
/// hand here.
Future<List<String>> artifactEntryNames(
  web.FileSystemDirectoryHandle directory,
) async {
  final iterator = directory.callMethod<JSObject>('values'.toJS);
  final names = <String>[];
  while (true) {
    final step = await iterator
        .callMethod<JSPromise<JSObject>>('next'.toJS)
        .toDart;
    if (step.getProperty<JSBoolean?>('done'.toJS)?.toDart ?? false) break;
    final handle = step.getProperty<JSObject?>('value'.toJS);
    final name = handle?.getProperty<JSString?>('name'.toJS)?.toDart;
    if (name != null) names.add(name);
  }
  return names;
}

/// Deletes the artifact under [key] and any in-flight marker for it.
Future<void> removeArtifact(
  web.FileSystemDirectoryHandle directory,
  String key,
) async {
  for (final name in <String>[key, '$key$_partSuffix']) {
    try {
      await directory.removeEntry(name).toDart;
    } catch (_) {
      // Already gone.
    }
  }
}

/// Ensures [url] is stored complete and returns it as a disk-backed blob.
///
/// Returns the stored copy without touching the network when it is already
/// complete. An interrupted transfer resumes from the bytes already stored.
/// [contentLength], when known, additionally validates a stored copy that
/// carries no in-flight marker — an artifact written by an older version of
/// this package has none, so a length is what distinguishes a whole legacy
/// file from an interrupted one.
Future<web.File> ensureArtifactFromUrl(
  web.FileSystemDirectoryHandle directory,
  Uri url, {
  int? contentLength,
  String? authToken,
  LlamaLoadProgress? onProgress,
}) async {
  final key = stableArtifactFileName(url);
  final existing = await artifactFile(directory, key);
  final partial = await artifactIsPartial(directory, key);
  if (existing != null &&
      !partial &&
      (contentLength == null || existing.size == contentLength)) {
    onProgress?.call(1);
    return existing;
  }

  // Only a transfer this package interrupted is known to be a prefix of the
  // artifact; anything else starts over.
  final resumeFrom = partial && existing != null ? existing.size : 0;
  final response = await _fetchArtifact(
    url.toString(),
    resumeFrom: resumeFrom,
    authToken: authToken,
  );
  if (resumeFrom > 0 && response.status == 416) {
    // The stored bytes already cover the whole artifact: the page went away
    // between the last write and clearing the in-flight marker. Drop the
    // marked copy and start over rather than wedging on 416 forever.
    await removeArtifact(directory, key);
    return ensureArtifactFromUrl(
      directory,
      url,
      contentLength: contentLength,
      authToken: authToken,
      onProgress: onProgress,
    );
  }
  final resumed = resumeFrom > 0 && response.status == 206;
  final total = _totalLength(response, resumed: resumed) ?? contentLength;

  await _markPartial(directory, key);
  final handle = await directory
      .getFileHandle(key, web.FileSystemGetFileOptions(create: true))
      .toDart;
  final writable = await handle
      .createWritable(
        web.FileSystemCreateWritableOptions(keepExistingData: resumed),
      )
      .toDart;
  var received = resumed ? resumeFrom : 0;
  try {
    if (resumed) await writable.seek(resumeFrom).toDart;
    final body = response.body;
    if (body == null) {
      throw const ArtifactStorageException(
        'The artifact download returned no body.',
      );
    }
    final reader = body.getReader() as web.ReadableStreamDefaultReader;
    while (true) {
      final chunk = await reader.read().toDart;
      if (chunk.done) break;
      final value = chunk.value as JSObject;
      await writable.write(value).toDart;
      received += value.getProperty<JSNumber>('byteLength'.toJS).toDartInt;
      if (total != null && total > 0) {
        onProgress?.call((received / total).clamp(0, 1).toDouble());
      }
    }
  } catch (error) {
    await writable.close().toDart;
    throw _asStorageFailure(error, 'downloading $url');
  }
  await writable.close().toDart;

  if (total != null && received != total) {
    throw ArtifactStorageException(
      'The artifact download ended early: got $received of $total bytes.',
    );
  }
  await _clearPartial(directory, key);
  await _requestPersistence();
  return (await handle.getFile().toDart);
}

/// Writes [bytes] into managed storage under [key].
Future<int> writeArtifactStream(
  web.FileSystemDirectoryHandle directory,
  String key,
  Stream<List<int>> bytes, {
  int? totalBytes,
  LlamaLoadProgress? onProgress,
}) async {
  await _markPartial(directory, key);
  final handle = await directory
      .getFileHandle(key, web.FileSystemGetFileOptions(create: true))
      .toDart;
  final writable = await handle.createWritable().toDart;
  var received = 0;
  try {
    await for (final chunk in bytes) {
      final data = chunk is Uint8List ? chunk : Uint8List.fromList(chunk);
      await writable.write(data.toJS).toDart;
      received += data.length;
      if (totalBytes != null && totalBytes > 0) {
        onProgress?.call((received / totalBytes).clamp(0, 1).toDouble());
      }
    }
  } catch (error) {
    await writable.close().toDart;
    // Unlike a download, an import cannot be resumed — the source stream is
    // spent — so a failed one leaves nothing behind to occupy storage.
    await removeArtifact(directory, key);
    throw _asStorageFailure(error, 'importing $key');
  }
  await writable.close().toDart;

  if (totalBytes != null && received != totalBytes) {
    await removeArtifact(directory, key);
    throw ArtifactStorageException(
      'The import ended early: got $received of $totalBytes bytes.',
    );
  }
  await _clearPartial(directory, key);
  await _requestPersistence();
  return received;
}

Future<void> _markPartial(
  web.FileSystemDirectoryHandle directory,
  String key,
) async {
  await directory
      .getFileHandle(
        '$key$_partSuffix',
        web.FileSystemGetFileOptions(create: true),
      )
      .toDart;
}

Future<void> _clearPartial(
  web.FileSystemDirectoryHandle directory,
  String key,
) async {
  try {
    await directory.removeEntry('$key$_partSuffix').toDart;
  } catch (_) {
    // Never created, or already cleared.
  }
}

/// Asks the browser not to evict managed artifacts under storage pressure.
Future<void> _requestPersistence() async {
  try {
    await web.window.navigator.storage.persist().toDart;
  } catch (_) {
    // Best-effort only.
  }
}

/// The size of [url] according to a `HEAD` request, or null when the server
/// or its CORS policy doesn't disclose it.
Future<int?> artifactContentLength(Uri url, {String? authToken}) async {
  try {
    final response = await web.window
        .callMethodVarArgs<JSPromise<web.Response>>('fetch'.toJS, [
          url.toString().toJS,
          <String, Object?>{
            'method': 'HEAD',
            if (authToken != null)
              'headers': <String, String>{'authorization': 'Bearer $authToken'},
          }.jsify(),
        ])
        .toDart;
    final length = response.headers.get('content-length');
    return length == null ? null : int.tryParse(length);
  } catch (_) {
    return null;
  }
}

/// Fetches [url], optionally as a range request resuming at [resumeFrom].
///
/// A range request that comes back 416 is returned to the caller rather than
/// thrown: the stored prefix covering the whole artifact is recoverable, and
/// only the caller can recover it.
Future<web.Response> _fetchArtifact(
  String url, {
  required int resumeFrom,
  String? authToken,
}) async {
  final init = <String, Object?>{
    // Identity keeps Content-Length/Content-Range meaningful; GGUF bytes
    // don't compress anyway.
    'headers': <String, String>{
      'accept-encoding': 'identity',
      if (resumeFrom > 0) 'range': 'bytes=$resumeFrom-',
      if (authToken != null) 'authorization': 'Bearer $authToken',
    },
  };
  final response = await web.window.callMethodVarArgs<JSPromise<web.Response>>(
    'fetch'.toJS,
    [url.toJS, init.jsify()],
  ).toDart;
  if (!response.ok && !(resumeFrom > 0 && response.status == 416)) {
    throw ArtifactStorageException(
      'The artifact download failed with HTTP ${response.status}.',
    );
  }
  return response;
}

/// The full artifact size implied by [response], or null when the server
/// doesn't disclose it.
int? _totalLength(web.Response response, {required bool resumed}) {
  final headers = response.headers;
  if (resumed) {
    // Content-Range: bytes <first>-<last>/<total>
    final range = headers.get('content-range');
    final slash = range?.lastIndexOf('/') ?? -1;
    if (range == null || slash < 0) return null;
    return int.tryParse(range.substring(slash + 1));
  }
  final length = headers.get('content-length');
  return length == null ? null : int.tryParse(length);
}

/// Wraps [error] as an [ArtifactStorageException], flagging the browser's
/// quota error so callers can tell "device full" from "download broke".
ArtifactStorageException _asStorageFailure(Object error, String action) {
  if (error is ArtifactStorageException) return error;
  final name = error is JSObject
      ? error.getProperty<JSString?>('name'.toJS)?.toDart
      : null;
  final quota =
      name == 'QuotaExceededError' ||
      error.toString().contains('QuotaExceeded');
  return ArtifactStorageException(
    quota
        ? 'There is not enough browser storage left to finish $action. '
              'Delete a model and try again.'
        : 'Storing the artifact failed while $action: $error',
    isQuotaExceeded: quota,
  );
}
