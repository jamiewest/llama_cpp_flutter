import 'dart:convert';
import 'dart:typed_data';

import 'package:llama_cpp_flutter/src/runtime/gguf_split.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('planGgufSplit', () {
    test('splits tensors across parts and preserves data byte-for-byte', () {
      // Arrange: four 100-byte tensors, packed under a 256-byte budget.
      final source = _buildGguf(
        kvs: {'general.architecture': 'qwen2'},
        tensorSizes: [100, 100, 100, 100],
      );

      // Act
      final result = planGgufSplit(
        headerPrefix: source.bytes,
        totalBytes: source.bytes.length,
        maxSplitBytes: 256,
      );

      // Assert
      final plan = result as GgufSplitPlan;
      expect(plan.parts, hasLength(2));
      final splits = [
        for (final part in plan.parts) _materialize(source.bytes, part),
      ];
      final decoded = [for (final split in splits) _DecodedGguf.parse(split)];
      expect(decoded[0].tensorData, source.tensorData.sublist(0, 2));
      expect(decoded[1].tensorData, source.tensorData.sublist(2));
    });

    test('writes llama-gguf-split metadata keys on every part', () {
      // Arrange
      final source = _buildGguf(
        kvs: {'general.architecture': 'qwen2'},
        tensorSizes: [100, 100, 100],
      );

      // Act
      final plan =
          planGgufSplit(
                headerPrefix: source.bytes,
                totalBytes: source.bytes.length,
                maxSplitBytes: 128,
              )
              as GgufSplitPlan;

      // Assert
      final decoded = [
        for (final part in plan.parts)
          _DecodedGguf.parse(_materialize(source.bytes, part)),
      ];
      expect(decoded, hasLength(3));
      for (var i = 0; i < decoded.length; i++) {
        expect(decoded[i].kvs['split.no'], i);
        expect(decoded[i].kvs['split.count'], decoded.length);
        expect(decoded[i].kvs['split.tensors.count'], 3);
      }
      expect(decoded[0].kvs['general.architecture'], 'qwen2');
      expect(decoded[1].kvs.containsKey('general.architecture'), isFalse);
    });

    test('carries a non-default alignment into every part', () {
      // Arrange
      final source = _buildGguf(
        kvs: {'general.alignment': 64},
        tensorSizes: [100, 100],
        alignment: 64,
      );

      // Act
      final plan =
          planGgufSplit(
                headerPrefix: source.bytes,
                totalBytes: source.bytes.length,
                maxSplitBytes: 128,
              )
              as GgufSplitPlan;

      // Assert
      for (final part in plan.parts) {
        final decoded = _DecodedGguf.parse(_materialize(source.bytes, part));
        expect(decoded.kvs['general.alignment'], 64);
        expect(part.headerBytes.length % 64, 0);
        for (final offset in decoded.tensorOffsets) {
          expect(offset % 64, 0);
        }
      }
    });

    test('strips pre-existing split keys from a merged file', () {
      // Arrange
      final source = _buildGguf(
        kvs: {'split.no': 0, 'split.count': 9, 'general.architecture': 'x'},
        tensorSizes: [100, 100],
      );

      // Act
      final plan =
          planGgufSplit(
                headerPrefix: source.bytes,
                totalBytes: source.bytes.length,
                maxSplitBytes: 128,
              )
              as GgufSplitPlan;

      // Assert: the rewritten keys appear once, with the new values.
      final decoded = _DecodedGguf.parse(
        _materialize(source.bytes, plan.parts.first),
      );
      expect(decoded.kvs['split.count'], plan.parts.length);
      expect(decoded.kvs['split.no'], 0);
      expect(decoded.kvs['general.architecture'], 'x');
    });

    test('gives an oversized tensor its own part', () {
      // Arrange
      final source = _buildGguf(kvs: const {}, tensorSizes: [50, 400, 50]);

      // Act
      final plan =
          planGgufSplit(
                headerPrefix: source.bytes,
                totalBytes: source.bytes.length,
                maxSplitBytes: 128,
              )
              as GgufSplitPlan;

      // Assert
      expect(plan.parts, hasLength(3));
      final decoded = [
        for (final part in plan.parts)
          _DecodedGguf.parse(_materialize(source.bytes, part)),
      ];
      expect(decoded[1].tensorData.single, source.tensorData[1]);
    });

    test('asks for a larger prefix when the header is cut off', () {
      // Arrange
      final source = _buildGguf(
        kvs: {'general.architecture': 'qwen2'},
        tensorSizes: [100],
      );

      // Act
      final result = planGgufSplit(
        headerPrefix: Uint8List.sublistView(source.bytes, 0, 40),
        totalBytes: source.bytes.length,
      );

      // Assert
      expect(result, isA<GgufSplitNeedsLargerPrefix>());
    });

    test('rejects non-GGUF bytes', () {
      // Arrange
      final bytes = Uint8List.fromList(List.filled(64, 7));

      // Act
      final result = planGgufSplit(headerPrefix: bytes, totalBytes: 64);

      // Assert
      expect(result, isA<GgufSplitUnsupported>());
    });

    test('rejects a truncated file whose prefix is the whole file', () {
      // Arrange
      final source = _buildGguf(kvs: const {}, tensorSizes: [100]);
      final cut = Uint8List.sublistView(source.bytes, 0, 40);

      // Act
      final result = planGgufSplit(headerPrefix: cut, totalBytes: 40);

      // Assert
      expect(result, isA<GgufSplitUnsupported>());
    });
  });
}

/// Joins a planned part's header with its slice of [source].
Uint8List _materialize(Uint8List source, GgufSplitPart part) {
  final builder = BytesBuilder(copy: false)
    ..add(part.headerBytes)
    ..add(Uint8List.sublistView(source, part.dataStart, part.dataEnd));
  return builder.toBytes();
}

final class _BuiltGguf {
  const _BuiltGguf(this.bytes, this.tensorData);

  final Uint8List bytes;

  /// Each tensor's exact data bytes, in declaration order.
  final List<Uint8List> tensorData;
}

/// Writes a little-endian GGUF v3 file with one F32 tensor per entry in
/// [tensorSizes] (sizes in bytes, multiples of 4), data filled with a
/// per-tensor byte pattern.
_BuiltGguf _buildGguf({
  required Map<String, Object> kvs,
  required List<int> tensorSizes,
  int alignment = 32,
}) {
  final builder = BytesBuilder(copy: false);
  final fixed = ByteData(24)
    ..setUint32(0, 0x46554747, Endian.little)
    ..setUint32(4, 3, Endian.little)
    ..setUint64(8, tensorSizes.length, Endian.little)
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

  var offset = 0;
  final offsets = <int>[];
  for (var i = 0; i < tensorSizes.length; i++) {
    offsets.add(offset);
    builder.add(_string('tensor_$i'));
    final info = ByteData(24)
      ..setUint32(0, 1, Endian.little)
      ..setUint64(4, tensorSizes[i] ~/ 4, Endian.little)
      ..setUint32(12, 0, Endian.little)
      ..setUint64(16, offset, Endian.little);
    builder.add(info.buffer.asUint8List());
    offset = (offset + tensorSizes[i] + alignment - 1) ~/ alignment * alignment;
  }

  final headerPadding =
      (builder.length + alignment - 1) ~/ alignment * alignment -
      builder.length;
  builder.add(Uint8List(headerPadding));

  final tensorData = <Uint8List>[];
  for (var i = 0; i < tensorSizes.length; i++) {
    final data = Uint8List.fromList(
      List.generate(tensorSizes[i], (b) => (i + 1) * 10 + b % 10),
    );
    tensorData.add(data);
    builder.add(data);
    final isLast = i == tensorSizes.length - 1;
    if (!isLast) {
      builder.add(Uint8List(offsets[i + 1] - offsets[i] - tensorSizes[i]));
    }
  }
  return _BuiltGguf(builder.toBytes(), tensorData);
}

Uint8List _string(String value) {
  final bytes = utf8.encode(value);
  final data = ByteData(8)..setUint64(0, bytes.length, Endian.little);
  final builder = BytesBuilder(copy: false)
    ..add(data.buffer.asUint8List())
    ..add(bytes);
  return builder.toBytes();
}

/// A minimal independent GGUF reader used to verify emitted splits.
final class _DecodedGguf {
  _DecodedGguf._(this.kvs, this.tensorOffsets, this.tensorData);

  final Map<String, Object?> kvs;
  final List<int> tensorOffsets;

  /// Tensor bytes in info order, sized from F32 dims.
  final List<Uint8List> tensorData;

  static _DecodedGguf parse(Uint8List bytes) {
    final data = ByteData.sublistView(bytes);
    var at = 0;
    int u32() {
      final v = data.getUint32(at, Endian.little);
      at += 4;
      return v;
    }

    int u64() {
      final v = data.getUint64(at, Endian.little);
      at += 8;
      return v;
    }

    String str() {
      final length = u64();
      final v = utf8.decode(Uint8List.sublistView(bytes, at, at + length));
      at += length;
      return v;
    }

    expect(u32(), 0x46554747, reason: 'split must start with GGUF magic');
    expect(u32(), 3, reason: 'split must keep the source version');
    final tensorCount = u64();
    final kvCount = u64();

    var alignment = 32;
    final kvs = <String, Object?>{};
    for (var i = 0; i < kvCount; i++) {
      final key = str();
      final type = u32();
      final value = switch (type) {
        2 => () {
          final v = data.getUint16(at, Endian.little);
          at += 2;
          return v;
        }(),
        4 => u32(),
        5 => data.getInt32((at += 4) - 4, Endian.little),
        8 => str(),
        _ => fail('unexpected kv type $type for $key'),
      };
      expect(kvs.containsKey(key), isFalse, reason: 'duplicate kv $key');
      kvs[key] = value;
      if (key == 'general.alignment') alignment = value as int;
    }

    final offsets = <int>[];
    final sizes = <int>[];
    for (var i = 0; i < tensorCount; i++) {
      str();
      final dims = u32();
      var elements = 1;
      for (var d = 0; d < dims; d++) {
        elements *= u64();
      }
      expect(u32(), 0, reason: 'test tensors are F32');
      offsets.add(u64());
      sizes.add(elements * 4);
    }

    final dataStart = (at + alignment - 1) ~/ alignment * alignment;
    for (var i = at; i < dataStart; i++) {
      expect(bytes[i], 0, reason: 'header padding must be zeros');
    }
    final tensorData = <Uint8List>[
      for (var i = 0; i < offsets.length; i++)
        Uint8List.sublistView(
          bytes,
          dataStart + offsets[i],
          dataStart + offsets[i] + sizes[i],
        ),
    ];
    expect(offsets.isEmpty || offsets.first == 0, isTrue);
    return _DecodedGguf._(kvs, offsets, tensorData);
  }
}
