import 'package:llama_flutter/llama_flutter.dart';
import 'package:extensions/ai.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  const decoder = GemmaStreamDecoder();

  Future<({String text, String reasoning, List<FunctionCallContent> calls})>
  decode(List<String> pieces) async {
    final updates = await decoder.decode(Stream.fromIterable(pieces)).toList();
    return (
      text: updates
          .expand((u) => u.contents.whereType<TextContent>())
          .map((c) => c.text)
          .join(),
      reasoning: updates
          .expand((u) => u.contents.whereType<TextReasoningContent>())
          .map((c) => c.text)
          .join(),
      calls: updates
          .expand((u) => u.contents.whereType<FunctionCallContent>())
          .toList(),
    );
  }

  group('GemmaStreamDecoder well-formed output', () {
    test('separates thinking, prose, and a tool call', () async {
      final result = await decode([
        '<|channel>thought\nplan the answer\n<channel|>',
        'The answer is 4.',
        '<|tool_call>call:log{note:<|"|>done<|"|>}<tool_call|>',
      ]);

      expect(result.reasoning, 'plan the answer\n');
      expect(result.text, 'The answer is 4.');
      expect(result.calls, hasLength(1));
      expect(result.calls.single.name, 'log');
      expect(result.calls.single.arguments, {'note': 'done'});
    });
  });

  group('GemmaStreamDecoder stray-marker hardening', () {
    test('drops an unmatched <channel|> close in prose', () async {
      final result = await decode(['The answer<channel|> is 4.']);

      expect(result.text, 'The answer is 4.');
      expect(result.reasoning, isEmpty);
    });

    test('drops an unmatched close split across pieces', () async {
      final result = await decode(['Hi<chan', 'nel|> there']);

      expect(result.text, 'Hi there');
    });

    test('drops a model-invented turn header', () async {
      final result = await decode(['<|turn>model\nHello there.']);

      expect(result.text, 'Hello there.');
    });

    test('drops a turn header split across pieces', () async {
      final result = await decode(['<|tu', 'rn>mod', 'el\nOk']);

      expect(result.text, 'Ok');
    });

    test('drops a dangling turn header at end of stream', () async {
      final result = await decode(['Done.<|turn>model']);

      expect(result.text, 'Done.');
    });

    test('still decodes thinking and calls after a stray header', () async {
      final result = await decode([
        '<|turn>model\n<|channel>thought\nplan<channel|>Answer',
        '<|tool_call>call:log{note:<|"|>x<|"|>}<tool_call|>',
      ]);

      expect(result.reasoning, 'plan');
      expect(result.text, 'Answer');
      expect(result.calls.single.name, 'log');
    });
  });
}
