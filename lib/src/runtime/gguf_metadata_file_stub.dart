import 'gguf_metadata.dart';

/// Web stub: there is no filesystem to read from.
///
/// Web hosts slice their blob and call [readGgufMetadata] directly.
Future<GgufMetadataResult> readGgufMetadataFile(
  String path, {
  required Set<String> keys,
}) async {
  throw UnsupportedError(
    'readGgufMetadataFile requires dart:io; on the web, read the blob '
    'prefix and call readGgufMetadata instead.',
  );
}
