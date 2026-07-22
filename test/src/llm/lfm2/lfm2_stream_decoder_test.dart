import 'package:llama_flutter/llama_flutter.dart';
import 'package:extensions/ai.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  const decoder = Lfm2StreamDecoder();

  Future<List<ChatResponseUpdate>> run(List<String> pieces) =>
      decoder.decode(Stream.fromIterable(pieces)).toList();

  String proseOf(List<ChatResponseUpdate> updates) => updates
      .expand((u) => u.contents.whereType<TextContent>())
      .map((c) => c.text)
      .join();

  List<FunctionCallContent> callsOf(List<ChatResponseUpdate> updates) => updates
      .expand((u) => u.contents.whereType<FunctionCallContent>())
      .toList();

  group('Lfm2StreamDecoder', () {
    test('passes prose through as text', () async {
      final updates = await run(['Hello', ', ', 'world']);
      expect(proseOf(updates), 'Hello, world');
      expect(callsOf(updates), isEmpty);
    });

    test('holds back a marker split across pieces', () async {
      final updates = await run([
        'before <|tool_',
        'call_start|>[f(x=1)]<|tool_call_end|>',
      ]);
      expect(proseOf(updates), 'before ');
      expect(callsOf(updates).single.name, 'f');
      expect(callsOf(updates).single.arguments, {'x': 1});
    });

    test('emits a tool call in a separate update from prose', () async {
      final updates = await run([
        'Checking. <|tool_call_start|>[get(id="7")]<|tool_call_end|>',
      ]);

      final callUpdates = updates.where(
        (u) => u.contents.any((c) => c is FunctionCallContent),
      );
      expect(callUpdates, hasLength(1));
      expect(
        callUpdates.single.contents.whereType<TextContent>(),
        isEmpty,
        reason: 'function-call updates must not also carry prose',
      );
      expect(proseOf(updates), 'Checking. ');
      expect(callsOf(updates).single.arguments, {'id': '7'});
    });

    test('parses a JSON tool call split across chunks', () async {
      final updates = await run([
        'Checking. <|tool_call_start|>{"name": "get", ',
        '"arguments": {"id": "7"}}<|tool_call_end|>',
      ]);

      expect(proseOf(updates), 'Checking. ');
      expect(callsOf(updates), hasLength(1));
      expect(callsOf(updates).single.name, 'get');
      expect(callsOf(updates).single.arguments, {'id': '7'});
    });

    test('surfaces a truncated tool call as raw text', () async {
      final updates = await run(['<|tool_call_start|>[broken(']);
      expect(callsOf(updates), isEmpty);
      expect(proseOf(updates), '<|tool_call_start|>[broken(');
    });
  });
}
