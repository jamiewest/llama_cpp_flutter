/// The generic ChatML implementation of the model-family seam.
library;

import 'package:extensions/ai.dart';

import '../chat_format.dart';
import '../common/marked_tool_call_decoder.dart';
import '../common/parsed_turn.dart';
import 'chatml_chat_template.dart';

/// ChatML's [ChatFormat]: [ChatmlChatTemplate] rendering paired with a
/// [MarkedToolCallDecoder] that buffers on `<tool_call>`.
class ChatmlChatFormat implements ChatFormat {
  /// Creates a [ChatmlChatFormat].
  const ChatmlChatFormat();

  static const ChatmlChatTemplate _template = ChatmlChatTemplate();

  @override
  bool get supportsThinking => false;

  @override
  RenderedPrompt render(
    Iterable<ChatMessage> messages, {
    Iterable<AIFunctionDeclaration> tools = const <AIFunctionDeclaration>[],
    bool enableThinking = false,
  }) => _template.render(messages, tools: tools);

  @override
  Stream<ChatResponseUpdate> decode(Stream<String> tokens) =>
      const MarkedToolCallDecoder(
        openMarker: '<tool_call>',
        parse: _parse,
      ).decode(tokens);

  static ParsedTurn _parse(String generated) => _template.parse(generated);
}
