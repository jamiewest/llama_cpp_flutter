/// Splits a raw Gemma token stream into Microsoft.Extensions.AI updates.
library;

import 'package:extensions/ai.dart';

import 'gemma_chat_template.dart';

/// Turns the `Stream<String>` from `LlamaFlutter.generate` into a stream of
/// [ChatResponseUpdate]s for an M.E.AI chat client.
///
/// Three kinds of content are separated as they stream in:
///   * **Thinking** — when thinking is enabled the model opens its turn with a
///     `<|channel>thought\n…\n<channel|>` block (see the Gemma 4 prompt format).
///     Its contents are emitted as [TextReasoningContent], with the `<|channel>`
///     markers and the `thought` label stripped.
///   * **Prose** — ordinary answer text, emitted as [TextContent].
///   * **Tool calls** — the first `<|tool_call>` marker flips the decoder into
///     buffering mode: everything from the marker onward is accumulated and,
///     when the stream ends, parsed into [FunctionCallContent] via
///     [GemmaChatTemplate.parse].
///
/// Text and tool calls are emitted in **separate** updates on purpose:
/// `FunctionInvokingChatClient` suppresses any update that carries a function
/// call, so combining them in one update would drop the prose.
///
/// A confused model can also emit control markup that doesn't belong in a
/// well-formed turn — a `<channel|>` close with no matching opener, or a
/// self-invented `<|turn>role` header. Both are stripped from prose rather
/// than surfaced, so raw control tokens never leak into user-visible text.
class GemmaStreamDecoder {
  const GemmaStreamDecoder([this.template = const GemmaChatTemplate()]);

  final GemmaChatTemplate template;

  static const String _channelOpen = GemmaChatTemplate.channelOpen;
  static const String _channelClose = GemmaChatTemplate.channelClose;
  static const String _callOpen = GemmaChatTemplate.toolCallOpen;
  static const String _turnOpen = GemmaChatTemplate.turnOpen;

  /// The thinking channel always carries the `thought` label right after
  /// `<|channel>`; it is dropped from the surfaced reasoning text.
  static const String _thoughtLabel = 'thought';

  Stream<ChatResponseUpdate> decode(Stream<String> tokens) async* {
    // Hold back the longest-marker-minus-one trailing chars so a marker that
    // straddles two pieces is never emitted as content.
    final holdback =
        <int>[
          _channelOpen.length,
          _channelClose.length,
          _callOpen.length,
          _turnOpen.length,
        ].reduce((a, b) => a > b ? a : b) -
        1;

    var buf = '';
    final tail = StringBuffer();
    var thinking = false;
    var buffering = false;
    var labelPending = false;
    var turnLabelPending = false;

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
          // Drop the leading `thought` label once we have enough to see it
          // whole (or the channel has already closed).
          if (labelPending) {
            if (buf.length > _thoughtLabel.length ||
                buf.contains(_channelClose)) {
              buf = _stripLabel(buf);
              labelPending = false;
              progressed = true;
              continue;
            }
            break;
          }

          final closeAt = buf.indexOf(_channelClose);
          if (closeAt >= 0) {
            final reasoning = buf.substring(0, closeAt);
            if (reasoning.isNotEmpty) yield _reasoning(reasoning);
            buf = buf.substring(closeAt + _channelClose.length);
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

        // A stray `<|turn>role` header is dropped through its newline.
        if (turnLabelPending) {
          final newline = buf.indexOf('\n');
          if (newline < 0) break;
          buf = buf.substring(newline + 1);
          turnLabelPending = false;
          progressed = true;
          continue;
        }

        // Act on whichever marker appears first; the rest of the buffer is
        // re-scanned after the state change.
        var at = -1;
        var marker = '';
        void consider(int index, String m) {
          if (index >= 0 && (at < 0 || index < at)) {
            at = index;
            marker = m;
          }
        }

        consider(buf.indexOf(_callOpen), _callOpen);
        consider(buf.indexOf(_channelOpen), _channelOpen);
        consider(buf.indexOf(_channelClose), _channelClose);
        consider(buf.indexOf(_turnOpen), _turnOpen);

        if (at >= 0) {
          final prose = buf.substring(0, at);
          if (prose.isNotEmpty) yield _text(prose);
          if (marker == _callOpen) {
            tail.write(buf.substring(at));
            buf = '';
            buffering = true;
            break;
          }
          buf = buf.substring(at + marker.length);
          if (marker == _channelOpen) {
            thinking = true;
            labelPending = true;
          } else if (marker == _turnOpen) {
            turnLabelPending = true;
          }
          // An unmatched `<channel|>` close needs no state: it is dropped.
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
      // A truncated or malformed tool call (e.g. the run hit maxTokens
      // mid-call) must not error the whole turn; surface the raw tail as
      // text instead so the user sees what the model produced.
      GemmaTurn turn;
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
    } else if (thinking) {
      if (labelPending) buf = _stripLabel(buf);
      if (buf.isNotEmpty) yield _reasoning(buf);
    } else if (turnLabelPending) {
      // The stream ended inside a stray `<|turn>role` header; drop it.
    } else if (buf.isNotEmpty) {
      yield _text(buf);
    }
  }

  static final RegExp _labelPattern = RegExp(
    '^\\s*$_thoughtLabel[ \\t]*\\r?\\n?',
  );

  static String _stripLabel(String s) {
    final match = _labelPattern.firstMatch(s);
    return match == null ? s : s.substring(match.end);
  }

  static ChatResponseUpdate _text(String text) =>
      ChatResponseUpdate.fromText(ChatRole.assistant, text);

  static ChatResponseUpdate _reasoning(String text) => ChatResponseUpdate(
    role: ChatRole.assistant,
    contents: <AIContent>[TextReasoningContent(text)],
  );
}
