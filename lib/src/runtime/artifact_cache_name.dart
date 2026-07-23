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
  return '${hash.toRadixString(16)}-${_safeTail(tail)}';
}

/// A cache file name for an artifact imported from local storage, where
/// there is no URL to hash into a stable name.
///
/// [nonce] separates repeat imports of the same file name; callers pass
/// something monotonic (a timestamp in hex). The result sorts alongside
/// [stableArtifactFileName]'s output and splits the same way, so
/// [artifactDisplayName] recovers the original file name from either.
String importedArtifactFileName(String fileName, {required String nonce}) =>
    'local$nonce-${_safeTail(fileName)}';

/// The original file name encoded in a name produced by
/// [stableArtifactFileName] or [importedArtifactFileName].
///
/// Returns [name] unchanged when it carries no hash prefix.
String artifactDisplayName(String name) {
  final dash = name.indexOf('-');
  return dash < 0 || dash == name.length - 1 ? name : name.substring(dash + 1);
}

/// [tail] reduced to filesystem-safe characters and a bounded length.
String _safeTail(String tail) {
  final safe = tail.replaceAll(RegExp(r'[^A-Za-z0-9_.-]'), '_');
  return safe.length > 64 ? safe.substring(0, 64) : safe;
}
