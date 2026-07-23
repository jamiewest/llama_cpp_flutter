import 'package:llama_cpp_flutter/chat.dart';
import 'package:extensions/ai.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  const template = Llama3ChatTemplate();
  const format = Llama3ChatFormat();

  group('Llama3ChatTemplate.render', () {
    test('plain system + user turn uses the header format', () {
      final prompt = template.render([
        ChatMessage.fromText(ChatRole.system, 'You are helpful.'),
        ChatMessage.fromText(ChatRole.user, 'Hi'),
      ]);

      expect(
        prompt.text,
        '<|start_header_id|>system<|end_header_id|>\n\n'
        'You are helpful.<|eot_id|>\n'
        '<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>\n'
        '<|start_header_id|>assistant<|end_header_id|>\n\n',
      );
      expect(prompt.stopSequences, <String>['<|eot_id|>', '<|eom_id|>']);
    });

    test('assistant tool call replays with python_tag and ends with eom', () {
      final prompt = template.render([
        ChatMessage.fromText(ChatRole.user, 'Weather in Paris?'),
        ChatMessage(
          role: ChatRole.assistant,
          contents: [
            FunctionCallContent(
              callId: 'call_0',
              name: 'get_weather',
              arguments: const {'location': 'Paris'},
            ),
          ],
        ),
      ], addGenerationPrompt: false);

      expect(
        prompt.text,
        contains(
          '<|start_header_id|>assistant<|end_header_id|>\n\n'
          '<|python_tag|>{"name":"get_weather","parameters":'
          '{"location":"Paris"}}<|eom_id|>\n',
        ),
      );
    });

    test('tool result feeds back in an ipython turn', () {
      final prompt = template.render([
        ChatMessage(
          role: ChatRole.tool,
          contents: [
            FunctionResultContent(
              callId: 'call_0',
              name: 'get_weather',
              result: const {'temp': 20},
            ),
          ],
        ),
      ], addGenerationPrompt: false);

      expect(
        prompt.text,
        contains(
          '<|start_header_id|>ipython<|end_header_id|>\n\n'
          '{"temp":20}<|eot_id|>\n',
        ),
      );
    });
  });

  group('Llama3ChatFormat.decode', () {
    test('separates prose from a python_tag tool call', () async {
      final updates = await format
          .decode(
            Stream<String>.value(
              'Sure. <|python_tag|>{"name": "f", "parameters": {"x": 1}}',
            ),
          )
          .toList();

      final text = updates
          .expand((u) => u.contents.whereType<TextContent>())
          .map((c) => c.text)
          .join();
      final calls = updates.expand(
        (u) => u.contents.whereType<FunctionCallContent>(),
      );

      expect(text.trim(), 'Sure.');
      expect(calls.single.name, 'f');
      expect(calls.single.arguments, {'x': 1});
    });
  });
}
