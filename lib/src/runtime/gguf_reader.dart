/// Low-level cursor over a GGUF header prefix, shared by the split planner
/// (`planGgufSplit`) and the metadata reader (`readGgufMetadata`).
///
/// Pure Dart and byte-oriented so both consumers are unit-testable off the
/// web.
library;

import 'dart:convert';
import 'dart:typed_data';

/// `'GGUF'` little-endian.
const int ggufMagic = 0x46554747;

/// GGUF metadata value-type tags.
const int ggufTypeUint8 = 0;
const int ggufTypeInt8 = 1;
const int ggufTypeUint16 = 2;
const int ggufTypeInt16 = 3;
const int ggufTypeUint32 = 4;
const int ggufTypeInt32 = 5;
const int ggufTypeFloat32 = 6;
const int ggufTypeBool = 7;
const int ggufTypeString = 8;
const int ggufTypeArray = 9;
const int ggufTypeUint64 = 10;
const int ggufTypeInt64 = 11;
const int ggufTypeFloat64 = 12;

/// Signals that the buffer ended before the value being read.
final class GgufTruncated implements Exception {
  const GgufTruncated();
}

/// A little-endian cursor over a GGUF header prefix.
///
/// Reads throw [GgufTruncated] when the prefix ends mid-value and
/// [FormatException] when the content is structurally invalid.
final class GgufReader {
  GgufReader(Uint8List bytes)
    : _data = ByteData.sublistView(bytes),
      _length = bytes.length;

  final ByteData _data;
  final int _length;

  int offset = 0;

  int uint32() {
    _ensure(4);
    final value = _data.getUint32(offset, Endian.little);
    offset += 4;
    return value;
  }

  int uint64() {
    _ensure(8);
    final value = _data.getUint64(offset, Endian.little);
    offset += 8;
    return value;
  }

  String string() {
    final length = uint64();
    _ensure(length);
    final value = utf8.decode(
      Uint8List.sublistView(_data, offset, offset + length),
      allowMalformed: true,
    );
    offset += length;
    return value;
  }

  /// Reads an integer-typed value, or skips and returns null.
  int? numericValueOrSkip(int type) {
    switch (type) {
      case ggufTypeUint8 || ggufTypeInt8:
        _ensure(1);
        final value = _data.getUint8(offset);
        offset += 1;
        return value;
      case ggufTypeUint16 || ggufTypeInt16:
        _ensure(2);
        final value = _data.getUint16(offset, Endian.little);
        offset += 2;
        return value;
      case ggufTypeUint32 || ggufTypeInt32:
        return uint32();
      case ggufTypeUint64 || ggufTypeInt64:
        return uint64();
      default:
        skipValue(type);
        return null;
    }
  }

  /// Reads a string-typed value, or skips and returns null.
  String? stringValueOrSkip(int type) {
    if (type == ggufTypeString) return string();
    skipValue(type);
    return null;
  }

  void skipValue(int type) {
    switch (type) {
      case ggufTypeUint8 || ggufTypeInt8 || ggufTypeBool:
        _skip(1);
      case ggufTypeUint16 || ggufTypeInt16:
        _skip(2);
      case ggufTypeUint32 || ggufTypeInt32 || ggufTypeFloat32:
        _skip(4);
      case ggufTypeUint64 || ggufTypeInt64 || ggufTypeFloat64:
        _skip(8);
      case ggufTypeString:
        _skip(uint64());
      case ggufTypeArray:
        final elementType = uint32();
        final count = uint64();
        for (var i = 0; i < count; i++) {
          skipValue(elementType);
        }
      default:
        throw const FormatException(
          'The GGUF header contains an unknown value type.',
        );
    }
  }

  void _skip(int count) {
    _ensure(count);
    offset += count;
  }

  void _ensure(int count) {
    if (count < 0 || offset + count > _length) {
      throw const GgufTruncated();
    }
  }
}
