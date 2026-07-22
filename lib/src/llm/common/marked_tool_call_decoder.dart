/// A reusable token-stream decoder for families whose tool calls begin at a
/// single open marker and run to the end of the turn.
library;

import 'package:extensions/ai.dart';

import 'parsed_turn.dart';

/// Splits a raw token stream into Microsoft.Extensions.AI updates for any
/// family without a reasoning channel.
///
/// Prose before [openMarker] is emitted as [TextContent]. The first
/// [openMarker] flips the decoder into buffering mode: everything from the
/// marker onward is accumulated and, when the stream ends, handed to [parse],
/// which returns the trailing prose plus the [FunctionCallContent]s. A
/// [FormatException] from [parse] (a truncated or malformed call) surfaces the
/// raw tail as text rather than erroring the whole turn.
///
/// Text and tool calls are emitted in **separate** updates on purpose:
/// `FunctionInvokingChatClient` suppresses any update that carries a function
/// call, so combining them in one update would drop the prose.
class MarkedToolCallDecoder {
  /// Creates a decoder that buffers on [openMarker] and delegates to [parse].
  const MarkedToolCallDecoder({required this.openMarker, required this.parse});

  /// The literal marker that begins a tool-call region.
  final String openMarker;

  /// Parses a buffered tail (which begins with [openMarker]) into a turn.
  final ParsedTurn Function(String generated) parse;

  /// Splits [tokens] into prose and tool-call updates.
  Stream<ChatResponseUpdate> decode(Stream<String> tokens) async* {
    // Hold back the marker-minus-one trailing chars so a marker straddling two
    // pieces is never emitted as content.
    final holdback = openMarker.length - 1;

    var buf = '';
    final tail = StringBuffer();
    var buffering = false;

    await for (final piece in tokens) {
      if (buffering) {
        tail.write(piece);
        continue;
      }
      buf += piece;

      final at = buf.indexOf(openMarker);
      if (at >= 0) {
        final prose = buf.substring(0, at);
        if (prose.isNotEmpty) yield _text(prose);
        tail.write(buf.substring(at));
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
      ParsedTurn turn;
      try {
        turn = parse(tail.toString());
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
