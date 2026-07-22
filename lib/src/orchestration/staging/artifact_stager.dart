/// Platform-selected artifact staging: on native, download (or accept)
/// local files and read the GGUF header for memory estimation; on the web,
/// pass through and let wllama stream/cache the URLs itself.
library;

export 'artifact_stager_stub.dart'
    if (dart.library.io) 'artifact_stager_io.dart';
