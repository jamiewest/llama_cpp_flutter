/// On-device LLM inference for Flutter, backed by llama.cpp — the core,
/// cross-platform API: [LlamaRuntime]/[LlamaSession], [ModelSpec], model
/// downloading, chat-format resolution, and the `ChatClient` adapter for
/// the agents framework.
///
/// Deliberately small. The deeper layers live in their own entrypoints:
///
/// - `package:llama_cpp_flutter/chat.dart` — concrete chat formats,
///   templates, stream decoders, [LlamaChatClient], prompt diagnostics.
/// - `package:llama_cpp_flutter/gguf.dart` — GGUF metadata reading and
///   artifact cache naming.
/// - `package:llama_cpp_flutter/orchestration.dart` — memory-aware
///   multi-agent orchestration over one loaded model.
/// - `package:llama_cpp_flutter/bridge.dart` — the low-level iOS/macOS
///   plugin bridge.
library;

export 'src/llm/chat_format.dart';
export 'src/llm/chat_format_detection.dart';
export 'src/llm/chat_format_registry.dart';
export 'src/llm/image_tiling.dart';
export 'src/llm/llama/llama_chat_client.dart' show SessionProvider;
export 'src/llm/llama/llama_chat_client_factory.dart';
export 'src/models/hugging_face.dart';
export 'src/models/model_spec.dart';
export 'src/runtime/artifact_store.dart';
export 'src/runtime/llama_runtime.dart';
export 'src/runtime/llama_runtime_api.dart';
export 'src/runtime/model_downloader.dart';
