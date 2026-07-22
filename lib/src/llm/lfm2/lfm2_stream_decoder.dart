/// Splits a raw LFM2 token stream into Microsoft.Extensions.AI updates.
library;

import 'package:extensions/ai.dart';

import 'lfm2_chat_template.dart';

/// Turns the `Stream<String>` from `LlamaFlutter.generate` into a stream of
/// [ChatResponseUpdate]s for an M.E.AI chat client.
///
/// LFM2 has no reasoning channel, so output is two kinds of content:
///   * **Prose** — ordinary answer text, emitted as [TextContent].
///   * **Tool calls** — the first `<|tool_call_start|>` marker flips the
///     decoder into buffering mode: everything from the marker onward is
///     accumulated and, when the stream ends, parsed into
///     [FunctionCallContent] via [Lfm2ChatTemplate.parse].
///
/// Text and tool calls are emitted in **separate** updates on purpose:
/// `FunctionInvokingChatClient` suppresses any update that carries a function
/// call, so combining them in one update would drop the prose.
class Lfm2StreamDecoder {
  const Lfm2StreamDecoder([this.template = const Lfm2ChatTemplate()]);

  final Lfm2ChatTemplate template;

  static const String _callOpen = Lfm2ChatTemplate.toolCallStart;

  Stream<ChatResponseUpdate> decode(Stream<String> tokens) async* {
    // Hold back the longest-marker-minus-one trailing chars so the
    // `<|tool_call_start|>` marker is never split across two pieces and
    // emitted as content.
    final holdback = _callOpen.length - 1;

    var buf = '';
    final tail = StringBuffer();
    var buffering = false;

    await for (final piece in tokens) {
      if (buffering) {
        tail.write(piece);
        continue;
      }
      buf += piece;

      final callAt = buf.indexOf(_callOpen);
      if (callAt >= 0) {
        final prose = buf.substring(0, callAt);
        if (prose.isNotEmpty) yield _text(prose);
        tail.write(buf.substring(callAt));
        buf = '';
        buffering = true;
        continue;
      }
      if (buf.length > holdback) {
        final emit = buf.substring(0, buf.length - holdback);
        if (emit.isNotEmpty) yield _text(emit);
        buf = buf.substring(buf.length - holdback);
      }
    }

    if (buffering) {
      // A truncated or malformed tool call (e.g. the run hit maxTokens
      // mid-call) must not error the whole turn; surface the raw tail as text
      // instead so the user sees what the model produced.
      Lfm2Turn turn;
      try {
        turn = template.parse(tail.toString());
      } on FormatException {
        yield _text(tail.toString());
        return;
      }
      if (turn.text.isNotEmpty) yield _text(turn.text);
      if (turn.calls.isNotEmpty) {
        yield ChatResponseUpdate(
          role: ChatRole.assistant,
          contents: List<AIContent>.of(turn.calls),
        );
      }
    } else if (buf.isNotEmpty) {
      yield _text(buf);
    }
  }

  static ChatResponseUpdate _text(String text) =>
      ChatResponseUpdate.fromText(ChatRole.assistant, text);
}
