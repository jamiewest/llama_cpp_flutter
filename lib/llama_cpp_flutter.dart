/// On-device LLM inference for Flutter, backed by llama.cpp: GGUF chat
/// clients for the agents framework (native xcframework on iOS/macOS, wllama
/// on web) plus memory-aware orchestration of per-agent llama.cpp contexts
/// over a single loaded model.
///
/// This barrel exposes the neutral runtime API ([LlamaRuntime] /
/// [LlamaSession]). The low-level darwin plugin bridge (`LlamaCppFlutter` and
/// its own session type) lives in `package:llama_cpp_flutter/bridge.dart`.
library;

export 'src/diagnostics/prompt_inspector.dart';
export 'src/llm/chat_format.dart';
export 'src/llm/chat_format_detection.dart';
export 'src/llm/chat_format_registry.dart';
export 'src/llm/chatml/chatml_chat_format.dart';
export 'src/llm/chatml/chatml_chat_template.dart';
export 'src/llm/common/marked_tool_call_decoder.dart';
export 'src/llm/common/parsed_turn.dart';
export 'src/llm/gemma/gemma_chat_format.dart';
export 'src/llm/gemma/gemma_chat_template.dart';
export 'src/llm/gemma/gemma_stream_decoder.dart';
export 'src/llm/image_tiling.dart';
export 'src/llm/lfm2/lfm2_chat_format.dart';
export 'src/llm/lfm2/lfm2_chat_template.dart';
export 'src/llm/lfm2/lfm2_stream_decoder.dart';
export 'src/llm/llama3/llama3_chat_format.dart';
export 'src/llm/llama3/llama3_chat_template.dart';
export 'src/llm/mistral/mistral_chat_format.dart';
export 'src/llm/mistral/mistral_chat_template.dart';
export 'src/llm/qwen/qwen_chat_format.dart';
export 'src/llm/qwen/qwen_chat_template.dart';
export 'src/llm/qwen/qwen_stream_decoder.dart';
export 'src/llm/llama/llama_chat_client.dart';
export 'src/llm/llama/llama_chat_client_factory.dart';
export 'src/models/hugging_face.dart';
export 'src/models/model_spec.dart';
export 'src/runtime/artifact_cache_name.dart';
export 'src/runtime/gguf_metadata.dart';
export 'src/runtime/gguf_metadata_file.dart';
export 'src/runtime/llama_runtime.dart';
export 'src/runtime/llama_runtime_api.dart';
export 'src/runtime/model_downloader.dart';
export 'src/runtime/stop_sequence_filter.dart';

// Orchestration: KV-cache swapping between agents, GGUF-based memory
// estimation, and dynamic context sizing (see SPEC.md).
export 'src/orchestration/agent_profile.dart';
export 'src/orchestration/memory/budget_planner.dart';
export 'src/orchestration/memory/memory_estimator.dart';
export 'src/orchestration/memory/memory_monitor.dart';
export 'src/orchestration/memory/memory_monitor_factory.dart';
export 'src/orchestration/orchestrator.dart';
export 'src/orchestration/orchestrator_events.dart';
export 'src/orchestration/state/snapshot_store.dart';
export 'src/orchestration/state/snapshot_types.dart';
