import 'package:extensions/logging.dart';

import 'llama_runtime_api.dart';

/// Creates a platform runtime.
LlamaRuntime createLlamaRuntime({LoggerFactory? loggerFactory}) =>
    throw UnsupportedError(
      'Local llama inference is not supported on this platform.',
    );
