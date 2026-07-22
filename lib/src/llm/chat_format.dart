/// Model-family seam between M.E.AI chat abstractions and a text-completion
/// inference engine.
library;

import 'dart:typed_data';

import 'package:extensions/ai.dart';

/// A rendered prompt ready for a text-completion inference engine.
class RenderedPrompt {
  const RenderedPrompt({
    required this.text,
    required this.stopSequences,
    this.media = const <Uint8List>[],
  });

  /// The formatted prompt text, ready for token generation.
  final String text;

  /// Strings that terminate a generation turn.
  final List<String> stopSequences;

  /// Encoded media bytes (image or audio) referenced by the media markers
  /// embedded in [text], in the order the markers appear. mtmd auto-detects
  /// each blob's kind from its magic bytes, so a single ordered list suffices.
  /// Empty for a text-only prompt.
  final List<Uint8List> media;
}

/// Pairs prompt rendering with output decoding for one model family.
///
/// A family's wire format is a unit: the markers [render] writes are the same
/// markers [decode] splits on, so the two halves are deliberately one
/// interface rather than separate renderer/decoder seams. Token-in/token-out
/// engines (llama.cpp, LiteRT-LM) compose a chat client from a [ChatFormat];
/// message-native engines (e.g. Apple Foundation Models) implement
/// `ChatClient` directly and bypass this interface entirely.
///
/// Implementations must be stateless: [decode] is invoked once per turn and
/// must be reentrant, keeping any parse state local to the call.
abstract interface class ChatFormat {
  /// Whether this family has a dedicated reasoning ("thinking") channel.
  ///
  /// When false, callers should not request thinking and [render]
  /// implementations ignore `enableThinking`.
  bool get supportsThinking;

  /// Renders [messages] (with optional [tools]) into the family's wire format.
  RenderedPrompt render(
    Iterable<ChatMessage> messages, {
    Iterable<AIFunctionDeclaration> tools,
    bool enableThinking,
  });

  /// Splits a raw generated-token stream into M.E.AI updates, separating
  /// prose ([TextContent]), reasoning ([TextReasoningContent]), and tool
  /// calls ([FunctionCallContent]).
  ///
  /// Text and tool calls must be emitted in separate updates:
  /// `FunctionInvokingChatClient` suppresses any update carrying a function
  /// call, so combining them would drop the prose.
  Stream<ChatResponseUpdate> decode(Stream<String> tokens);
}
