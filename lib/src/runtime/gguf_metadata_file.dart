/// Reads GGUF metadata straight from a local file path.
///
/// Convenience over [readGgufMetadata] for hosts that have a filesystem:
/// it reads a growing header prefix until the requested keys resolve. On
/// the web (no `dart:io`) the function throws [UnsupportedError]; web
/// hosts slice their blob and call [readGgufMetadata] directly.
library;

export 'gguf_metadata_file_stub.dart'
    if (dart.library.io) 'gguf_metadata_file_io.dart';
