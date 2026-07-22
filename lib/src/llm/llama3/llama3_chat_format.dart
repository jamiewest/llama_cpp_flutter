/// The Llama 3 / 3.1 implementation of the model-family seam.
library;

import 'package:extensions/ai.dart';

import '../chat_format.dart';
import '../common/marked_tool_call_decoder.dart';
import '../common/parsed_turn.dart';
import 'llama3_chat_template.dart';

/// Llama 3's [ChatFormat]: [Llama3ChatTemplate] rendering paired with a
/// [MarkedToolCallDecoder] that buffers on `<|python_tag|>`.
class Llama3ChatFormat implements ChatFormat {
  /// Creates a [Llama3ChatFormat].
  const Llama3ChatFormat();

  static const Llama3ChatTemplate _template = Llama3ChatTemplate();

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
        openMarker: Llama3ChatTemplate.pythonTag,
        parse: _parse,
      ).decode(tokens);

  static ParsedTurn _parse(String generated) => _template.parse(generated);
}
