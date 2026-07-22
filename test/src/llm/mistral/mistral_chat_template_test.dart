import 'package:llama_cpp_flutter/llama_cpp_flutter.dart';
import 'package:extensions/ai.dart';
import 'package:flutter_test/flutter_test.dart';

class _TestFunction extends AIFunctionDeclaration {
  _TestFunction({required super.name, super.description});
}

void main() {
  const template = MistralChatTemplate();
  const format = MistralChatFormat();

  group('MistralChatTemplate.render', () {
    test('wraps a user turn in [INST] … [/INST]', () {
      final prompt = template.render([
        ChatMessage.fromText(ChatRole.user, 'Hi'),
      ]);

      expect(prompt.text, '[INST] Hi [/INST]');
      expect(prompt.stopSequences, <String>['</s>']);
    });

    test('prepends the system prompt to the first user turn', () {
      final prompt = template.render([
        ChatMessage.fromText(ChatRole.system, 'You are helpful.'),
        ChatMessage.fromText(ChatRole.user, 'Hi'),
      ]);

      expect(prompt.text, '[INST] You are helpful.\n\nHi [/INST]');
    });

    test('puts the AVAILABLE_TOOLS block before the last user turn', () {
      final prompt = template.render(
        [ChatMessage.fromText(ChatRole.user, 'Weather?')],
        tools: [_TestFunction(name: 'get_weather', description: 'Get weather')],
      );

      expect(
        prompt.text,
        '[AVAILABLE_TOOLS][{"type":"function","function":'
        '{"name":"get_weather","description":"Get weather"}}][/AVAILABLE_TOOLS]'
        '[INST] Weather? [/INST]',
      );
    });

    test('assistant tool call replays as a [TOOL_CALLS] block', () {
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
      ]);

      expect(
        prompt.text,
        '[INST] Weather in Paris? [/INST]'
        '[TOOL_CALLS][{"name":"get_weather","arguments":'
        '{"location":"Paris"}}]</s>',
      );
    });
  });

  group('MistralChatFormat.decode', () {
    test('separates prose from a [TOOL_CALLS] block', () async {
      final updates = await format
          .decode(
            Stream<String>.value(
              'Checking.[TOOL_CALLS][{"name": "f", "arguments": {"x": 1}}]',
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

      expect(text.trim(), 'Checking.');
      expect(calls.single.name, 'f');
      expect(calls.single.arguments, {'x': 1});
    });
  });
}
