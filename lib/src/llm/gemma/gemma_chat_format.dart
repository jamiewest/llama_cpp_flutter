/// The Gemma 4 implementation of the model-family seam.
library;

import 'package:extensions/ai.dart';

import '../chat_format.dart';
import 'gemma_chat_template.dart';
import 'gemma_stream_decoder.dart';

/// Gemma 4's [ChatFormat]: [GemmaChatTemplate] rendering paired with
/// [GemmaStreamDecoder] output splitting.
class GemmaChatFormat implements ChatFormat {
  const GemmaChatFormat();

  static const GemmaChatTemplate _template = GemmaChatTemplate();
  static const GemmaStreamDecoder _decoder = GemmaStreamDecoder();

  @override
  bool get supportsThinking => true;

  @override
  RenderedPrompt render(
    Iterable<ChatMessage> messages, {
    Iterable<AIFunctionDeclaration> tools = const <AIFunctionDeclaration>[],
    bool enableThinking = false,
  }) {
    final prompt = _template.render(
      messages,
      tools: tools,
      enableThinking: enableThinking,
    );
    return RenderedPrompt(
      text: prompt.text,
      stopSequences: prompt.stopSequences,
      media: prompt.media,
    );
  }

  @override
  Stream<ChatResponseUpdate> decode(Stream<String> tokens) =>
      _decoder.decode(tokens);
}
