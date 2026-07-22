import 'package:llama_flutter/llama_flutter.dart';
import 'package:extensions/ai.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  const template = GemmaChatTemplate();

  group('GemmaChatTemplate.render turn state after tool rounds', () {
    final call = FunctionCallContent(
      callId: 'call_0',
      name: 'get_weather',
      arguments: const {'city': 'Paris'},
    );
    final result = ChatMessage(
      role: ChatRole.tool,
      contents: [
        FunctionResultContent(
          callId: 'call_0',
          name: 'get_weather',
          result: 'sunny',
        ),
      ],
    );

    test('a completed tool round with prose closes the turn and appends the '
        'generation prompt', () {
      // Regression: the model emitted plan text alongside its tool call.
      // The prompt used to end at `<turn|>` with no `<|turn>model` header,
      // so the model invented its own turn/channel markup, which leaked
      // into user-visible text.
      final prompt = template.render([
        ChatMessage.fromText(ChatRole.user, 'Weather in Paris?'),
        ChatMessage(
          role: ChatRole.assistant,
          contents: [call, TextContent('I will check the weather.')],
        ),
        result,
      ]);

      expect(
        prompt.text,
        endsWith(
          '<tool_response|>I will check the weather.<turn|>\n'
          '<|turn>model\n',
        ),
      );
    });

    test('a completed tool round without prose stays mid-turn', () {
      final prompt = template.render([
        ChatMessage.fromText(ChatRole.user, 'Weather in Paris?'),
        ChatMessage(role: ChatRole.assistant, contents: [call]),
        result,
      ]);

      expect(
        prompt.text,
        endsWith('response:get_weather{value:<|"|>sunny<|"|>}<tool_response|>'),
      );
    });

    test('a pending tool call ends at the tool-response opener', () {
      final prompt = template.render([
        ChatMessage.fromText(ChatRole.user, 'Weather in Paris?'),
        ChatMessage(role: ChatRole.assistant, contents: [call]),
      ]);

      expect(prompt.text, endsWith('<tool_call|><|tool_response>'));
    });

    test('a user message after a completed tool round starts a fresh turn', () {
      final prompt = template.render([
        ChatMessage.fromText(ChatRole.user, 'Weather in Paris?'),
        ChatMessage(
          role: ChatRole.assistant,
          contents: [call, TextContent('It is sunny.')],
        ),
        result,
        ChatMessage.fromText(ChatRole.user, 'And tomorrow?'),
      ]);

      expect(
        prompt.text,
        endsWith(
          'It is sunny.<turn|>\n'
          '<|turn>user\nAnd tomorrow?<turn|>\n'
          '<|turn>model\n',
        ),
      );
    });
  });
}
