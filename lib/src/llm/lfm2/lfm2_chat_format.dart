/// The LFM2 / LFM2-VL implementation of the model-family seam.
library;

import 'package:extensions/ai.dart';

import '../chat_format.dart';
import 'lfm2_chat_template.dart';
import 'lfm2_stream_decoder.dart';

/// LFM's [ChatFormat]: [Lfm2ChatTemplate] rendering paired with
/// [Lfm2StreamDecoder] output splitting.
///
/// Covers the text and vision variants for LFM2 and LFM2.5. They share the
/// same ChatML wire format and tool-call markers, but differ in whether tool
/// declarations/results are wrapped in Liquid's LFM2-only tool tags.
class Lfm2ChatFormat implements ChatFormat {
  const Lfm2ChatFormat({
    this.toolTagStyle = LfmToolTagStyle.lfm2,
    this.toolCallSyntax = LfmToolCallSyntax.pythonic,
  });

  /// Whether tool declarations/results use LFM2 tags or LFM2.5 plain JSON.
  final LfmToolTagStyle toolTagStyle;

  /// How assistant tool-call examples are rendered back into chat history.
  final LfmToolCallSyntax toolCallSyntax;

  Lfm2ChatTemplate get _template => Lfm2ChatTemplate(
    toolTagStyle: toolTagStyle,
    toolCallSyntax: toolCallSyntax,
  );

  @override
  bool get supportsThinking => false;

  @override
  RenderedPrompt render(
    Iterable<ChatMessage> messages, {
    Iterable<AIFunctionDeclaration> tools = const <AIFunctionDeclaration>[],
    bool enableThinking = false,
  }) {
    final prompt = _template.render(messages, tools: tools);
    return RenderedPrompt(
      text: prompt.text,
      stopSequences: prompt.stopSequences,
      media: prompt.images,
    );
  }

  @override
  Stream<ChatResponseUpdate> decode(Stream<String> tokens) =>
      Lfm2StreamDecoder(_template).decode(tokens);
}
