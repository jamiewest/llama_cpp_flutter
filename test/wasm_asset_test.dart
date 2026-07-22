@TestOn('vm')
library;

import 'dart:io';

import 'package:flutter_test/flutter_test.dart';

/// Byte-level sanity checks on the committed wllama wasm asset. (The asset's
/// sha256 is checked against tool/versions.env by CI, which ties the bytes to
/// the pinned @wllama/wllama npm release; `flutter test --platform chrome`
/// cannot load assets, so these checks run on the VM.)
void main() {
  test('bundled wllama.wasm looks like a real WebAssembly module', () {
    final file = File('lib/assets/wasm/wllama.wasm');
    expect(file.existsSync(), isTrue, reason: 'asset missing');

    final raf = file.openSync();
    final magic = raf.readSync(4);
    raf.closeSync();

    expect(magic, [0x00, 0x61, 0x73, 0x6D], reason: 'missing \\0asm magic');
    expect(
      file.lengthSync(),
      inInclusiveRange(2 << 20, 64 << 20),
      reason: 'implausible wasm size',
    );
  });
}
