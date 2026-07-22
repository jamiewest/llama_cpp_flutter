/// Web/stub surface of the native model downloader.
///
/// The web runtime streams models straight from their URL (wllama's cache or
/// OPFS), so there is nothing to download to a local path there; constructing
/// a downloader on a non-IO platform is a wiring mistake and fails loudly.
library;

import 'package:extensions/logging.dart';

import '../models/model_spec.dart';
import 'llama_runtime_api.dart';

/// Local paths of a spec's downloaded artifacts, ready for
/// `LlamaRuntime.loadModel`.
typedef ModelArtifactPaths = ({
  String modelPath,
  String? mmprojPath,
  String? draftPath,
});

/// Unavailable on this platform; see the `dart:io` implementation.
final class ModelDownloader {
  /// Throws: local artifact downloads only exist on platforms with a
  /// filesystem. The web runtime loads models from their URL directly.
  ModelDownloader({
    String? authToken,
    Object? createClient,
    LoggerFactory? loggerFactory,
  }) {
    throw UnsupportedError(
      'ModelDownloader requires dart:io; on the web, models load directly '
      'from their URL.',
    );
  }

  /// Unreachable; the constructor always throws.
  Future<String> download(
    Uri url, {
    required String directory,
    LlamaLoadProgress? onProgress,
  }) => throw UnsupportedError('ModelDownloader requires dart:io.');

  /// Unreachable; the constructor always throws.
  Future<ModelArtifactPaths> downloadSpecArtifacts(
    ModelSpec spec, {
    required String directory,
    LlamaLoadProgress? onProgress,
  }) => throw UnsupportedError('ModelDownloader requires dart:io.');
}
