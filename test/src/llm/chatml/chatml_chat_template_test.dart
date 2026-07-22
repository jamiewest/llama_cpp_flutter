import 'package:llama_cpp_flutter/llama_cpp_flutter.dart';
import 'package:extensions/ai.dart';
import 'package:flutter_test/flutter_test.dart';

class _TestFunction extends AIFunctionDeclaration {
  _TestFunction({required super.name, super.description});
}

void main() {
  const template = ChatmlChatTemplate();
  const format = ChatmlChatFormat();

  group('ChatmlChatTemplate.render', () {
    test('plain system + user turn', () {
      final prompt = template.render([
        ChatMessage.fromText(ChatRole.system, 'You are helpful.'),
        ChatMessage.fromText(ChatRole.user, 'Hi'),
      ]);

      expect(
        prompt.text,
        '<|im_start|>system\nYou are helpful.<|im_end|>\n'
        '<|im_start|>user\nHi<|im_end|>\n'
        '<|im_start|>assistant\n',
      );
      expect(prompt.stopSequences, <String>['<|im_end|>']);
    });

    test('tools render a Hermes <tools> block in the system turn', () {
      final prompt = template.render(
        [ChatMessage.fromText(ChatRole.user, 'Weather?')],
        tools: [_TestFunction(name: 'get_weather', description: 'Get weather')],
      );

      expect(prompt.text, contains('<tools>'));
      expect(
        prompt.text,
        contains(
          '{"type":"function","function":'
          '{"name":"get_weather","description":"Get weather"}}',
        ),
      );
      expect(prompt.text, contains('<tool_call></tool_call>'));
    });

    test('assistant tool call replays as a <tool_call> block', () {
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
        '<|im_start|>user\nWeather in Paris?<|im_end|>\n'
        '<|im_start|>assistant\n'
        '<tool_call>\n{"name":"get_weather","arguments":{"location":"Paris"}}\n'
        '</tool_call><|im_end|>\n',
      );
    });
  });

  group('ChatmlChatFormat.decode', () {
    test('separates prose from a streamed tool call', () async {
      final updates = await format
          .decode(
            Stream<String>.value(
              'Let me check. <tool_call>\n'
              '{"name": "f", "arguments": {"x": 1}}\n</tool_call>',
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

      expect(text.trim(), 'Let me check.');
      expect(calls.single.name, 'f');
      expect(calls.single.arguments, {'x': 1});
    });

    test('round-trips a rendered tool call through decode', () async {
      final call = FunctionCallContent(
        callId: 'call_0',
        name: 'do_it',
        arguments: const {'a': 'x', 'n': 2, 'flag': true},
      );
      final body = template
          .render([
            ChatMessage(role: ChatRole.assistant, contents: [call]),
          ], addGenerationPrompt: false)
          .text
          .replaceFirst('<|im_start|>assistant\n', '')
          .replaceFirst('<|im_end|>\n', '');

      final calls = await format
          .decode(Stream<String>.value(body))
          .expand((u) => u.contents.whereType<FunctionCallContent>())
          .toList();

      expect(calls.single.name, 'do_it');
      expect(calls.single.arguments, call.arguments);
    });
  });
}
