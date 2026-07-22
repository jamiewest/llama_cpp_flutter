import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'gguf_metadata.dart';

/// Reads the values of [keys] from the GGUF file at [path].
///
/// Starts with a modest header prefix and grows it until every requested
/// key is ruled in or out, so the call never reads more of a multi-gigabyte
/// model file than the header requires. Returns
/// [GgufMetadataNeedsLargerPrefix] only when the whole file (or a 1 GiB
/// ceiling) was consumed without resolving the header.
Future<GgufMetadataResult> readGgufMetadataFile(
  String path, {
  required Set<String> keys,
}) async {
  final file = File(path);
  final totalBytes = await file.length();
  const maxPrefixLength = 1024 * 1024 * 1024;
  var prefixLength = 4 * 1024 * 1024;
  while (true) {
    prefixLength = math.min(prefixLength, totalBytes);
    final handle = await file.open();
    final Uint8List prefix;
    try {
      prefix = await handle.read(prefixLength);
    } finally {
      await handle.close();
    }
    final result = readGgufMetadata(headerPrefix: prefix, keys: keys);
    final canGrow = prefixLength < totalBytes && prefixLength < maxPrefixLength;
    if (result is GgufMetadataNeedsLargerPrefix && canGrow) {
      prefixLength *= 8;
      continue;
    }
    return result;
  }
}
