/// Stable, filesystem-safe cache names for downloaded model artifacts.
///
/// Shared by the web runtime's OPFS cache and the native model downloader so
/// the same URL always lands under the same name.
library;

/// A stable cache file name for [url]: a short hash of the full URL plus a
/// sanitized tail of its last path segment (so the file stays recognizably
/// e.g. `…-gemma-4b-it-q4_k_m.gguf`).
String stableArtifactFileName(Uri url) {
  final text = url.toString();
  // djb2 kept within 32 bits; the multiplier is small enough that the
  // intermediate stays exact in JS doubles on the web backend.
  var hash = 5381;
  for (final unit in text.codeUnits) {
    hash = (hash * 33 + unit) & 0xffffffff;
  }
  final tail = url.pathSegments.isEmpty ? 'model' : url.pathSegments.last;
  var safeTail = tail.replaceAll(RegExp(r'[^A-Za-z0-9_.-]'), '_');
  if (safeTail.length > 64) safeTail = safeTail.substring(0, 64);
  return '${hash.toRadixString(16)}-$safeTail';
}
