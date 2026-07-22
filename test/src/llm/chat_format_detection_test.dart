import 'package:llama_flutter/llama_flutter.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('detectChatFormatName', () {
    test('identifies families from their embedded template', () {
      expect(
        detectChatFormatName(chatTemplate: '…<start_of_turn>model…'),
        'gemma',
      );
      expect(
        detectChatFormatName(
          chatTemplate: '…<|start_header_id|>assistant<|end_header_id|>…',
        ),
        'llama3',
      );
      expect(
        detectChatFormatName(chatTemplate: '…[INST] {{ prompt }}…'),
        'mistral',
      );
      expect(
        detectChatFormatName(
          chatTemplate: '…<|im_start|>user…<tool_response>…',
        ),
        'qwen',
      );
      expect(
        detectChatFormatName(
          chatTemplate: '…<|im_start|>…{% if enable_thinking %}…',
        ),
        'qwen',
      );
      expect(
        detectChatFormatName(
          chatTemplate: '…<|im_start|>…<|tool_list_start|>…',
        ),
        'lfm2',
      );
      expect(
        detectChatFormatName(chatTemplate: '…<|im_start|>user…'),
        'chatml',
      );
    });

    test('the template outranks an ambiguous architecture', () {
      // Mistral GGUFs commonly declare architecture 'llama'.
      expect(
        detectChatFormatName(architecture: 'llama', chatTemplate: '…[INST]…'),
        'mistral',
      );
    });

    test('falls back to the architecture when no template matches', () {
      expect(detectChatFormatName(architecture: 'gemma3'), 'gemma');
      expect(detectChatFormatName(architecture: 'qwen3moe'), 'qwen');
      expect(detectChatFormatName(architecture: 'lfm2'), 'lfm2');
    });

    test('refuses to guess for a bare llama architecture', () {
      expect(detectChatFormatName(architecture: 'llama'), isNull);
      expect(detectChatFormatName(), isNull);
    });

    test('detected names resolve in the registry', () {
      for (final name in ['gemma', 'llama3', 'mistral', 'qwen', 'lfm2']) {
        expect(resolveChatFormat(name), isNotNull, reason: name);
      }
    });
  });
}
