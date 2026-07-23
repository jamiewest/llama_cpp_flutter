/// Chat-format internals: per-family prompt templates, stream decoders,
/// tool-call parsing, the [LlamaChatClient] implementation, and prompt
/// diagnostics.
///
/// Most apps only need `resolveChatFormat`/`detectChatFormatNameForGguf`
/// from the main `package:llama_cpp_flutter/llama_cpp_flutter.dart`
/// entrypoint; import this one to construct or extend formats directly —
/// e.g. registering a custom family with `registerChatFormat`, tuning a
/// built-in template, or reusing the decoders.
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
export 'src/llm/lfm2/lfm2_chat_format.dart';
export 'src/llm/lfm2/lfm2_chat_template.dart';
export 'src/llm/lfm2/lfm2_stream_decoder.dart';
export 'src/llm/llama/llama_chat_client.dart';
export 'src/llm/llama3/llama3_chat_format.dart';
export 'src/llm/llama3/llama3_chat_template.dart';
export 'src/llm/mistral/mistral_chat_format.dart';
export 'src/llm/mistral/mistral_chat_template.dart';
export 'src/llm/qwen/qwen_chat_format.dart';
export 'src/llm/qwen/qwen_chat_template.dart';
export 'src/llm/qwen/qwen_stream_decoder.dart';
export 'src/runtime/stop_sequence_filter.dart';
