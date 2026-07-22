/// Reads selected string metadata values from a GGUF header prefix.
///
/// Used to identify a model from its own file — e.g. mapping
/// `general.architecture` and `tokenizer.chat_template` to a chat format —
/// instead of guessing from the file name. Pure Dart and byte-oriented so it
/// works on any platform: hosts read the first bytes of the file (native) or
/// blob (web) and hand them here.
library;

import 'dart:typed_data';

import 'gguf_reader.dart';

/// GGUF metadata key holding the model architecture (e.g. `'gemma3'`).
const String ggufArchitectureKey = 'general.architecture';

/// GGUF metadata key holding the model's embedded Jinja chat template.
const String ggufChatTemplateKey = 'tokenizer.chat_template';

/// Outcome of [readGgufMetadata].
sealed class GgufMetadataResult {
  const GgufMetadataResult();
}

/// The prefix ended before all requested keys could be ruled in or out.
///
/// Retry with a larger [readGgufMetadata] `headerPrefix`.
final class GgufMetadataNeedsLargerPrefix extends GgufMetadataResult {
  const GgufMetadataNeedsLargerPrefix();
}

/// The bytes are not a readable GGUF header; the [reason] is user-facing.
final class GgufMetadataUnsupported extends GgufMetadataResult {
  const GgufMetadataUnsupported(this.reason);

  final String reason;
}

/// The metadata values found for the requested keys.
final class GgufMetadata extends GgufMetadataResult {
  const GgufMetadata(this.values, [this.numericValues = const {}]);

  /// Requested keys that were present with a string value. Keys that are
  /// absent, or whose value is not a string, are omitted.
  final Map<String, String> values;

  /// Requested keys that were present with an integer value (any GGUF
  /// int width). Keys that are absent, or whose value is neither a string
  /// nor an integer, are omitted from both maps.
  final Map<String, int> numericValues;
}

/// Reads the string values of [keys] from the GGUF whose first bytes are
/// [headerPrefix].
///
/// Returns as soon as every requested key has been seen, so a modest prefix
/// (GGUF writers emit `general.*` keys first) usually suffices; otherwise
/// the whole key/value section must fit in [headerPrefix].
GgufMetadataResult readGgufMetadata({
  required Uint8List headerPrefix,
  required Set<String> keys,
}) {
  final reader = GgufReader(headerPrefix);
  final values = <String, String>{};
  final numericValues = <String, int>{};
  try {
    if (reader.uint32() != ggufMagic) {
      return const GgufMetadataUnsupported('The file is not a GGUF model.');
    }
    final version = reader.uint32();
    if (version < 2 || version > 3) {
      return GgufMetadataUnsupported(
        'GGUF version $version is not supported (only little-endian v2/v3).',
      );
    }
    reader.uint64(); // tensor count
    final kvCount = reader.uint64();

    var remaining = keys.toSet();
    for (var i = 0; i < kvCount && remaining.isNotEmpty; i++) {
      final key = reader.string();
      final type = reader.uint32();
      if (remaining.remove(key)) {
        if (type == ggufTypeString) {
          values[key] = reader.string();
        } else {
          final value = reader.numericValueOrSkip(type);
          if (value != null) numericValues[key] = value;
        }
      } else {
        reader.skipValue(type);
      }
    }
  } on GgufTruncated {
    return const GgufMetadataNeedsLargerPrefix();
  } on FormatException catch (error) {
    return GgufMetadataUnsupported(error.message);
  }
  return GgufMetadata(values, numericValues);
}
