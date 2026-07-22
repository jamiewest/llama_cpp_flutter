import 'dart:convert';
import 'dart:typed_data';

import 'package:llama_flutter/llama_flutter.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('readGgufMetadata', () {
    test('reads requested string values', () {
      final bytes = _buildHeader(
        kvs: {
          'general.architecture': 'gemma3',
          'general.name': 'test model',
          'tokenizer.chat_template': '{{ bos }}<start_of_turn>…',
        },
      );

      final result =
          readGgufMetadata(
                headerPrefix: bytes,
                keys: {ggufArchitectureKey, ggufChatTemplateKey},
              )
              as GgufMetadata;

      expect(result.values, {
        ggufArchitectureKey: 'gemma3',
        ggufChatTemplateKey: '{{ bos }}<start_of_turn>…',
      });
    });

    test('omits keys that are absent or not strings', () {
      final bytes = _buildHeader(
        kvs: {'general.alignment': 64, 'general.architecture': 'qwen3'},
      );

      final result =
          readGgufMetadata(
                headerPrefix: bytes,
                keys: {'general.alignment', ggufArchitectureKey, 'nope'},
              )
              as GgufMetadata;

      expect(result.values, {ggufArchitectureKey: 'qwen3'});
    });

    test('returns early once every requested key was seen', () {
      // The requested keys come first; everything after them is cut off
      // mid-value, which must not matter.
      final bytes = _buildHeader(
        kvs: {
          'general.architecture': 'lfm2',
          'general.name': 'a filler value that will be truncated',
        },
      );
      final cut = Uint8List.sublistView(bytes, 0, bytes.length - 10);

      final result =
          readGgufMetadata(headerPrefix: cut, keys: {ggufArchitectureKey})
              as GgufMetadata;

      expect(result.values, {ggufArchitectureKey: 'lfm2'});
    });

    test('asks for a larger prefix when cut before the requested keys', () {
      final bytes = _buildHeader(
        kvs: {'general.name': 'x', 'general.architecture': 'gemma3'},
      );
      final cut = Uint8List.sublistView(bytes, 0, 30);

      final result = readGgufMetadata(
        headerPrefix: cut,
        keys: {ggufArchitectureKey},
      );

      expect(result, isA<GgufMetadataNeedsLargerPrefix>());
    });

    test('rejects non-GGUF bytes', () {
      final result = readGgufMetadata(
        headerPrefix: Uint8List(64),
        keys: {ggufArchitectureKey},
      );

      expect(result, isA<GgufMetadataUnsupported>());
    });
  });
}

/// Writes a little-endian GGUF v3 header with [kvs] and no tensors.
Uint8List _buildHeader({required Map<String, Object> kvs}) {
  final builder = BytesBuilder(copy: false);
  final fixed = ByteData(24)
    ..setUint32(0, 0x46554747, Endian.little)
    ..setUint32(4, 3, Endian.little)
    ..setUint64(8, 0, Endian.little)
    ..setUint64(16, kvs.length, Endian.little);
  builder.add(fixed.buffer.asUint8List());

  kvs.forEach((key, value) {
    builder.add(_string(key));
    switch (value) {
      case final String text:
        final typeTag = ByteData(4)..setUint32(0, 8, Endian.little);
        builder
          ..add(typeTag.buffer.asUint8List())
          ..add(_string(text));
      case final int number:
        final tagged = ByteData(8)
          ..setUint32(0, 4, Endian.little)
          ..setUint32(4, number, Endian.little);
        builder.add(tagged.buffer.asUint8List());
      default:
        throw ArgumentError('Unsupported test kv: $value');
    }
  });
  return builder.toBytes();
}

Uint8List _string(String value) {
  final bytes = utf8.encode(value);
  final data = ByteData(8)..setUint64(0, bytes.length, Endian.little);
  final builder = BytesBuilder(copy: false)
    ..add(data.buffer.asUint8List())
    ..add(bytes);
  return builder.toBytes();
}
