import 'package:extensions/logging.dart';

import 'llama_runtime_api.dart';

/// Creates a platform runtime.
///
/// On supported platforms this returns the native llama.cpp runtime
/// (iOS/macOS) or the wllama runtime (web). This stub is what the analyzer
/// and unsupported platforms resolve.
LlamaRuntime createLlamaRuntime({
  String? artifactCacheDirectory,
  LoggerFactory? loggerFactory,
}) => throw UnsupportedError(
  'Local llama inference is not supported on this platform.',
);
