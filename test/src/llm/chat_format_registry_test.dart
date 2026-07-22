import 'package:llama_cpp_flutter/llama_cpp_flutter.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  group('resolveChatFormat', () {
    test('resolves each known family to its format', () {
      expect(resolveChatFormat('gemma'), isA<GemmaChatFormat>());
      expect(resolveChatFormat('lfm2'), isA<Lfm2ChatFormat>());
      expect(resolveChatFormat('lfm2-vl'), isA<Lfm2ChatFormat>());
      expect(resolveChatFormat('lfm2.5'), isA<Lfm2ChatFormat>());
      expect(resolveChatFormat('lfm2.5-vl'), isA<Lfm2ChatFormat>());
      expect(resolveChatFormat('lfm25'), isA<Lfm2ChatFormat>());
      expect(resolveChatFormat('lfm25-vl'), isA<Lfm2ChatFormat>());
      expect(resolveChatFormat('chatml'), isA<ChatmlChatFormat>());
      expect(resolveChatFormat('llama3'), isA<Llama3ChatFormat>());
      expect(resolveChatFormat('mistral'), isA<MistralChatFormat>());
      expect(resolveChatFormat('qwen'), isA<QwenChatFormat>());
    });

    test('an unset name resolves to the default (gemma)', () {
      expect(resolveChatFormat(null), isA<GemmaChatFormat>());
      expect(resolveChatFormat(''), isA<GemmaChatFormat>());
    });

    test('an unknown name resolves to null', () {
      expect(resolveChatFormat('nope'), isNull);
    });

    test('LFM aliases select the correct tool tag style', () {
      expect(
        (resolveChatFormat('lfm2')! as Lfm2ChatFormat).toolTagStyle,
        LfmToolTagStyle.lfm2,
      );
      expect(
        (resolveChatFormat('lfm2-vl')! as Lfm2ChatFormat).toolTagStyle,
        LfmToolTagStyle.lfm2,
      );
      expect(
        (resolveChatFormat('lfm2.5')! as Lfm2ChatFormat).toolTagStyle,
        LfmToolTagStyle.lfm25,
      );
      expect(
        (resolveChatFormat('lfm2.5-vl')! as Lfm2ChatFormat).toolTagStyle,
        LfmToolTagStyle.lfm25,
      );
      expect(
        (resolveChatFormat('lfm25')! as Lfm2ChatFormat).toolTagStyle,
        LfmToolTagStyle.lfm25,
      );
      expect(
        (resolveChatFormat('lfm25-vl')! as Lfm2ChatFormat).toolTagStyle,
        LfmToolTagStyle.lfm25,
      );
    });

    test('supportedChatFormatNames lists every family', () {
      expect(
        supportedChatFormatNames,
        containsAll(<String>[
          'gemma',
          'lfm2',
          'lfm2-vl',
          'lfm2.5',
          'lfm2.5-vl',
          'lfm25',
          'lfm25-vl',
          'chatml',
          'llama3',
          'mistral',
          'qwen',
        ]),
      );
    });
  });

  group('registerChatFormat', () {
    test('registered formats resolve and appear in the supported names', () {
      const custom = ChatmlChatFormat();
      registerChatFormat('my-new-family', custom);

      expect(resolveChatFormat('my-new-family'), same(custom));
      expect(supportedChatFormatNames, contains('my-new-family'));
    });

    test('a registered format overrides a built-in of the same name', () {
      const replacement = ChatmlChatFormat();
      registerChatFormat('mistral', replacement);

      expect(resolveChatFormat('mistral'), same(replacement));
    });

    test('rejects an empty name', () {
      expect(
        () => registerChatFormat('  ', const ChatmlChatFormat()),
        throwsArgumentError,
      );
    });
  });
}
