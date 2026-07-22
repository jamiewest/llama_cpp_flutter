@TestOn('browser')
library;

import 'package:llama_cpp_flutter/llama_cpp_flutter.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('web runtime uses the packaged WASM asset path', () {
    expect(createLlamaRuntime(), isA<LlamaRuntime>());
    expect(
      llamaWasmAssetPath,
      'assets/packages/llama_cpp_flutter/lib/assets/wasm/wllama.wasm',
    );
  });
}
