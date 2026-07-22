/// The Mistral / Mixtral implementation of the model-family seam.
library;

import 'package:extensions/ai.dart';

import '../chat_format.dart';
import '../common/marked_tool_call_decoder.dart';
import '../common/parsed_turn.dart';
import 'mistral_chat_template.dart';

/// Mistral's [ChatFormat]: [MistralChatTemplate] rendering paired with a
/// [MarkedToolCallDecoder] that buffers on `[TOOL_CALLS]`.
class MistralChatFormat implements ChatFormat {
  /// Creates a [MistralChatFormat].
  const MistralChatFormat();

  static const MistralChatTemplate _template = MistralChatTemplate();

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
        openMarker: MistralChatTemplate.toolCalls,
        parse: _parse,
      ).decode(tokens);

  static ParsedTurn _parse(String generated) => _template.parse(generated);
}
