export 'llama_runtime_stub.dart'
    if (dart.library.io) 'llama_runtime_native.dart'
    if (dart.library.js_interop) 'llama_runtime_web.dart';
