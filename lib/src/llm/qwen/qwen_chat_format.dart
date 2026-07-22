/// The Qwen2.5 / Qwen3 implementation of the model-family seam.
library;

import 'package:extensions/ai.dart';

import '../chat_format.dart';
import 'qwen_chat_template.dart';
import 'qwen_stream_decoder.dart';

/// Qwen's [ChatFormat]: [QwenChatTemplate] rendering paired with
/// [QwenStreamDecoder], which surfaces the `<think>` reasoning channel.
class QwenChatFormat implements ChatFormat {
  /// Creates a [QwenChatFormat].
  const QwenChatFormat();

  static const QwenChatTemplate _template = QwenChatTemplate();
  static const QwenStreamDecoder _decoder = QwenStreamDecoder();

  @override
  bool get supportsThinking => true;

  @override
  RenderedPrompt render(
    Iterable<ChatMessage> messages, {
    Iterable<AIFunctionDeclaration> tools = const <AIFunctionDeclaration>[],
    bool enableThinking = false,
  }) =>
      _template.render(messages, tools: tools, enableThinking: enableThinking);

  @override
  Stream<ChatResponseUpdate> decode(Stream<String> tokens) =>
      _decoder.decode(tokens);
}
