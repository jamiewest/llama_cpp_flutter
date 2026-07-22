/// Native model-artifact downloader (`dart:io`).
///
/// The web runtime streams models straight from their URL (wllama's cache or
/// OPFS); native llama.cpp needs a real file on disk. This downloader closes
/// that gap: point it at an artifact URL — typically a Hugging Face
/// `resolve` URL, see `huggingFaceModelUri` — and it produces a cached local
/// file with streaming progress, resume of interrupted downloads, and an
/// atomic finalize so a partial file is never mistaken for a complete one.
library;

import 'dart:io';

import 'package:extensions/logging.dart';

import '../models/model_spec.dart';
import 'artifact_cache_name.dart';
import 'llama_runtime_api.dart';

/// Local paths of a spec's downloaded artifacts, ready for
/// `LlamaRuntime.loadModel`.
typedef ModelArtifactPaths = ({
  String modelPath,
  String? mmprojPath,
  String? draftPath,
});

/// Downloads model artifacts (GGUF weights, projectors, drafters) into a
/// local cache directory.
///
/// Files are cached by URL (see `stableArtifactFileName`): a completed
/// download is returned immediately on the next call, and an interrupted one
/// resumes from where it stopped via an HTTP `Range` request. The final file
/// only appears under its cache name once fully written, so an existing file
/// is always complete.
final class ModelDownloader {
  /// Creates a downloader.
  ///
  /// [authToken] is sent as a `Bearer` token — required for gated or private
  /// Hugging Face repositories. [createClient] overrides how the underlying
  /// [HttpClient] is created, e.g. to inject a fake in tests; the client is
  /// closed after each download. [loggerFactory] supplies the download
  /// progress logger; null logs nothing.
  ModelDownloader({
    this.authToken,
    HttpClient Function()? createClient,
    LoggerFactory? loggerFactory,
  }) : _createClient = createClient ?? HttpClient.new,
       _logger = (loggerFactory ?? NullLoggerFactory.instance).createLogger(
         'ModelDownloader',
       );

  /// Bearer token added to every request, or null for anonymous access.
  final String? authToken;

  final HttpClient Function() _createClient;
  final Logger _logger;

  /// Downloads [url] into [directory], returning the local file path.
  ///
  /// Returns immediately when the artifact is already cached. [onProgress]
  /// reports fractions of the full file (resumed downloads start partway
  /// in), when the server discloses the total size.
  Future<String> download(
    Uri url, {
    required String directory,
    LlamaLoadProgress? onProgress,
  }) async {
    final dir = Directory(directory);
    await dir.create(recursive: true);
    final destination = File(
      '${dir.path}${Platform.pathSeparator}${stableArtifactFileName(url)}',
    );
    if (destination.existsSync()) {
      _logger.logDebug('Artifact already cached: ${destination.path}.');
      return destination.path;
    }

    final part = File('${destination.path}.part');
    final resumeFrom = part.existsSync() ? part.lengthSync() : 0;
    _logger.logInformation(
      resumeFrom > 0
          ? 'Resuming download of $url from byte $resumeFrom.'
          : 'Downloading $url.',
    );

    final client = _createClient();
    try {
      final request = await client.getUrl(url);
      // Identity keeps Content-Length/Content-Range meaningful; GGUF bytes
      // don't compress anyway.
      request.headers.set(HttpHeaders.acceptEncodingHeader, 'identity');
      if (authToken != null) {
        request.headers.set(
          HttpHeaders.authorizationHeader,
          'Bearer $authToken',
        );
      }
      if (resumeFrom > 0) {
        request.headers.set(HttpHeaders.rangeHeader, 'bytes=$resumeFrom-');
      }
      final response = await request.close();

      final resumed =
          resumeFrom > 0 && response.statusCode == HttpStatus.partialContent;
      if (response.statusCode != HttpStatus.ok && !resumed) {
        await response.drain<void>();
        if (resumeFrom > 0 &&
            response.statusCode == HttpStatus.requestedRangeNotSatisfiable) {
          // The part file already covers the whole artifact (a crash landed
          // between the final write and the rename). Restart cleanly so the
          // retry can't wedge on 416 forever.
          await part.delete();
          return download(url, directory: directory, onProgress: onProgress);
        }
        throw HttpException(
          'The model download failed with HTTP ${response.statusCode}.',
          uri: url,
        );
      }

      final total = _totalLength(response, resumed: resumed);
      var received = resumed ? resumeFrom : 0;
      final sink = part.openWrite(
        mode: resumed ? FileMode.append : FileMode.write,
      );
      try {
        await for (final chunk in response) {
          sink.add(chunk);
          received += chunk.length;
          if (total != null && total > 0) {
            onProgress?.call((received / total).clamp(0, 1).toDouble());
          }
        }
      } finally {
        await sink.close();
      }

      if (total != null && received != total) {
        throw HttpException(
          'The model download ended early: got $received of $total bytes.',
          uri: url,
        );
      }
      await part.rename(destination.path);
      _logger.logInformation(
        'Downloaded $url ($received bytes) to ${destination.path}.',
      );
      return destination.path;
    } finally {
      client.close();
    }
  }

  /// Downloads every artifact [spec] declares (weights, projector, drafter)
  /// into [directory] and returns their local paths.
  ///
  /// [onProgress] tracks the main model download, which dominates; the
  /// projector and drafter are small and download without progress.
  Future<ModelArtifactPaths> downloadSpecArtifacts(
    ModelSpec spec, {
    required String directory,
    LlamaLoadProgress? onProgress,
  }) async {
    final modelPath = await download(
      spec.modelUrl,
      directory: directory,
      onProgress: onProgress,
    );
    final mmprojUrl = spec.mmprojUrl;
    final draftUrl = spec.draftUrl;
    return (
      modelPath: modelPath,
      mmprojPath: mmprojUrl == null
          ? null
          : await download(mmprojUrl, directory: directory),
      draftPath: draftUrl == null
          ? null
          : await download(draftUrl, directory: directory),
    );
  }

  /// The full artifact size implied by [response], or null when the server
  /// doesn't disclose it.
  static int? _totalLength(
    HttpClientResponse response, {
    required bool resumed,
  }) {
    if (resumed) {
      // Content-Range: bytes <first>-<last>/<total>
      final range = response.headers.value(HttpHeaders.contentRangeHeader);
      final slash = range?.lastIndexOf('/') ?? -1;
      if (range == null || slash < 0) return null;
      return int.tryParse(range.substring(slash + 1));
    }
    final length = response.contentLength;
    return length > 0 ? length : null;
  }
}
