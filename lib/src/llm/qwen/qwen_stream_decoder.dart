/// Splits a raw Qwen token stream into Microsoft.Extensions.AI updates.
library;

import 'package:extensions/ai.dart';

import 'qwen_chat_template.dart';

/// Turns the `Stream<String>` from the runtime into [ChatResponseUpdate]s,
/// separating three kinds of content:
///   * **Thinking** — a leading `<think>…</think>` block (Qwen3), surfaced as
///     [TextReasoningContent] with the markers stripped.
///   * **Prose** — ordinary answer text, emitted as [TextContent].
///   * **Tool calls** — the first `<tool_call>` flips the decoder into
///     buffering mode; on stream end the tail is parsed into
///     [FunctionCallContent] via [QwenChatTemplate.parse].
///
/// Text and tool calls are emitted in **separate** updates: a
/// `FunctionInvokingChatClient` suppresses any update carrying a function call,
/// so combining them would drop the prose.
class QwenStreamDecoder {
  /// Creates a [QwenStreamDecoder].
  const QwenStreamDecoder([this.template = const QwenChatTemplate()]);

  /// The template whose `parse` recovers tool calls from the buffered tail.
  final QwenChatTemplate template;

  static const String _thinkOpen = QwenChatTemplate.thinkOpen;
  static const String _thinkClose = QwenChatTemplate.thinkClose;
  static const String _callOpen = '<tool_call>';

  /// Splits [tokens] into reasoning, prose, and tool-call updates.
  Stream<ChatResponseUpdate> decode(Stream<String> tokens) async* {
    final holdback =
        <int>[
          _thinkOpen.length,
          _thinkClose.length,
          _callOpen.length,
        ].reduce((a, b) => a > b ? a : b) -
        1;

    var buf = '';
    final tail = StringBuffer();
    var thinking = false;
    var buffering = false;

    await for (final piece in tokens) {
      if (buffering) {
        tail.write(piece);
        continue;
      }
      buf += piece;

      var progressed = true;
      while (progressed) {
        progressed = false;

        if (thinking) {
          final closeAt = buf.indexOf(_thinkClose);
          if (closeAt >= 0) {
            final reasoning = buf.substring(0, closeAt);
            if (reasoning.isNotEmpty) yield _reasoning(reasoning);
            buf = buf.substring(closeAt + _thinkClose.length);
            thinking = false;
            progressed = true;
            continue;
          }
          if (buf.length > holdback) {
            final emit = buf.substring(0, buf.length - holdback);
            if (emit.isNotEmpty) yield _reasoning(emit);
            buf = buf.substring(buf.length - holdback);
          }
          break;
        }

        final thinkAt = buf.indexOf(_thinkOpen);
        final callAt = buf.indexOf(_callOpen);

        if (callAt >= 0 && (thinkAt < 0 || callAt <= thinkAt)) {
          final prose = buf.substring(0, callAt);
          if (prose.isNotEmpty) yield _text(prose);
          tail.write(buf.substring(callAt));
          buf = '';
          buffering = true;
          break;
        }
        if (thinkAt >= 0) {
          final prose = buf.substring(0, thinkAt);
          if (prose.isNotEmpty) yield _text(prose);
          buf = buf.substring(thinkAt + _thinkOpen.length);
          thinking = true;
          progressed = true;
          continue;
        }
        if (buf.length > holdback) {
          final emit = buf.substring(0, buf.length - holdback);
          if (emit.isNotEmpty) yield _text(emit);
          buf = buf.substring(buf.length - holdback);
        }
        break;
      }
    }

    if (buffering) {
      try {
        final turn = template.parse(tail.toString());
        if (turn.text.isNotEmpty) yield _text(turn.text);
        if (turn.calls.isNotEmpty) {
          yield ChatResponseUpdate(
            role: ChatRole.assistant,
            contents: List<AIContent>.of(turn.calls),
          );
        }
      } on FormatException {
        yield _text(tail.toString());
      }
    } else if (thinking) {
      if (buf.isNotEmpty) yield _reasoning(buf);
    } else if (buf.isNotEmpty) {
      yield _text(buf);
    }
  }

  static ChatResponseUpdate _text(String text) =>
      ChatResponseUpdate.fromText(ChatRole.assistant, text);

  static ChatResponseUpdate _reasoning(String text) => ChatResponseUpdate(
    role: ChatRole.assistant,
    contents: <AIContent>[TextReasoningContent(text)],
  );
}
