import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:llama_cpp_flutter/src/runtime/gguf_reader.dart';

void main() {
  group('64-bit accessors', () {
    // `ByteData.get/setUint64` throw on dart2js, and the only caller of
    // the split planner is the web runtime — so these must be exercised
    // with `--platform chrome` to mean anything.
    test('round-trips values across the 32-bit boundary', () {
      const values = [
        0,
        1,
        0xff,
        0xffffffff, // largest low word
        0x100000000, // first value needing the high word
        0x100000001,
        2186186784, // the 2.04 GiB model that motivated this
        0x7fffffffffff, // 47-bit, still exact as a double
      ];

      for (final value in values) {
        final data = ByteData(8);
        ggufWriteUint64(data, 0, value);
        expect(ggufReadUint64(data, 0), value, reason: 'round-trip $value');
      }
    });

    test('writes little-endian bytes', () {
      final data = ByteData(8);
      // Six significant bytes: wide enough to span both 32-bit halves,
      // narrow enough to be an exact JavaScript integer literal.
      ggufWriteUint64(data, 0, 0x030405060708);
      expect(data.buffer.asUint8List(), [8, 7, 6, 5, 4, 3, 0, 0]);
    });

    test('honours a non-zero offset without touching neighbours', () {
      final data = ByteData(24)
        ..setUint32(0, 0xdeadbeef, Endian.little)
        ..setUint32(20, 0xfeedface, Endian.little);
      ggufWriteUint64(data, 8, 0x1234567890);

      expect(ggufReadUint64(data, 8), 0x1234567890);
      expect(data.getUint32(0, Endian.little), 0xdeadbeef);
      expect(data.getUint32(20, Endian.little), 0xfeedface);
    });

    test('rejects a value beyond the exact-integer range', () {
      // A corrupt header must fail loudly rather than yield a rounded
      // offset that would silently produce a garbage split.
      final data = ByteData(8)
        ..setUint32(0, 0, Endian.little)
        ..setUint32(4, 0xffffffff, Endian.little);

      expect(() => ggufReadUint64(data, 0), throwsFormatException);
    });
  });

  group('GgufReader', () {
    test('reads a uint64 above 2^31 and advances the cursor', () {
      final data = ByteData(12);
      ggufWriteUint64(data, 0, 2186186784);
      data.setUint32(8, 7, Endian.little);

      final reader = GgufReader(data.buffer.asUint8List());
      expect(reader.uint64(), 2186186784);
      expect(reader.offset, 8);
      expect(reader.uint32(), 7);
    });

    test('throws GgufTruncated when the prefix ends mid-value', () {
      final reader = GgufReader(Uint8List(4));
      expect(reader.uint64, throwsA(isA<GgufTruncated>()));
    });
  });
}
