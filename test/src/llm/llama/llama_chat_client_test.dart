import 'dart:async';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:llama_cpp_flutter/chat.dart';
import 'package:llama_cpp_flutter/llama_cpp_flutter.dart';
import 'package:extensions/ai.dart';
import 'package:extensions/system.dart';
import 'package:flutter_test/flutter_test.dart';

class _TestFunction extends AIFunctionDeclaration {
  _TestFunction({required super.name});
}

final class _RecordingSession implements LlamaSession {
  _RecordingSession({this.stats});

  String? prompt;
  Iterable<Uint8List>? media;
  Iterable<String>? stopSequences;
  List<LlamaChatTurn>? turns;

  /// Stats reported through `onStats` after the token stream, when set.
  final LlamaGenerationStats? stats;

  @override
  Stream<String> generate(
    String prompt, {
    int maxTokens = 256,
    double temperature = 0.8,
    int? topK,
    double? topP,
    int? seed,
    List<String> stopSequences = const <String>[],
    List<Uint8List>? media,
    List<LlamaChatTurn>? turns,
    int sequenceId = 0,
    LlamaStatsCallback? onStats,
  }) {
    this.prompt = prompt;
    this.stopSequences = stopSequences;
    this.media = media;
    this.turns = turns;
    if (stats != null) onStats?.call(stats!);
    return Stream<String>.value('Hello!');
  }

  @override
  Future<void> cancel() async {}

  @override
  LlamaSessionCapabilities get capabilities => const LlamaSessionCapabilities(
    canPersistState: false,
    reportsStateSize: false,
  );

  @override
  Future<int> saveState(String path, {int sequenceId = 0}) async => 0;

  @override
  Future<int> loadState(String path, {int sequenceId = 0}) async =>
      throw UnsupportedError('fake');

  @override
  Future<int> stateSizeBytes({int sequenceId = 0}) async => 0;

  @override
  Future<LlamaStashResult> stashState(String key, {int sequenceId = 0}) async =>
      (tokens: 0, bytes: 0);

  @override
  Future<int> restoreStashedState(String key, {int sequenceId = 0}) async =>
      throw UnsupportedError('fake');

  @override
  Future<int> dropStashedState(String key) async => 0;

  @override
  Future<void> clearSequence(int sequenceId) async {}

  @override
  Future<void> setImageTokenBudget(int? imageTokenBudget) async {}

  @override
  Future<void> dispose() async {}
}

/// A session whose token stream stays open until [cancel] closes it, for
/// exercising the cancellation path.
final class _CancellableSession implements LlamaSession {
  final StreamController<String> tokens = StreamController<String>();
  bool cancelCalled = false;
  bool generateCalled = false;

  @override
  Stream<String> generate(
    String prompt, {
    int maxTokens = 256,
    double temperature = 0.8,
    int? topK,
    double? topP,
    int? seed,
    List<String> stopSequences = const <String>[],
    List<Uint8List>? media,
    List<LlamaChatTurn>? turns,
    int sequenceId = 0,
    LlamaStatsCallback? onStats,
  }) {
    generateCalled = true;
    return tokens.stream;
  }

  @override
  Future<void> cancel() async {
    cancelCalled = true;
    await tokens.close();
  }

  @override
  LlamaSessionCapabilities get capabilities => const LlamaSessionCapabilities(
    canPersistState: false,
    reportsStateSize: false,
  );

  @override
  Future<int> saveState(String path, {int sequenceId = 0}) async => 0;

  @override
  Future<int> loadState(String path, {int sequenceId = 0}) async =>
      throw UnsupportedError('fake');

  @override
  Future<int> stateSizeBytes({int sequenceId = 0}) async => 0;

  @override
  Future<LlamaStashResult> stashState(String key, {int sequenceId = 0}) async =>
      (tokens: 0, bytes: 0);

  @override
  Future<int> restoreStashedState(String key, {int sequenceId = 0}) async =>
      throw UnsupportedError('fake');

  @override
  Future<int> dropStashedState(String key) async => 0;

  @override
  Future<void> clearSequence(int sequenceId) async {}

  @override
  Future<void> setImageTokenBudget(int? imageTokenBudget) async {}

  @override
  Future<void> dispose() async {}
}

void main() {
  group('LlamaChatClient prompt preparation', () {
    test('renders tools and every message in author order', () async {
      final session = _RecordingSession();
      final client = LlamaChatClient(
        sessionProvider: () async => session,
        format: const Lfm2ChatFormat(),
        contextSize: 4096,
      );

      await client.getResponse(
        messages: [
          ChatMessage.fromText(ChatRole.user, 'Hi'),
          ChatMessage.fromText(ChatRole.assistant, 'Hello!'),
          ChatMessage.fromText(ChatRole.user, 'What is left to do?'),
        ],
        options: ChatOptions(
          instructions: 'You are a helpful assistant.',
          tools: [_TestFunction(name: 'TodoList_GetRemaining')],
        ),
        cancellationToken: CancellationToken.none,
      );

      final prompt = session.prompt!;
      expect(prompt, contains('List of tools: <|tool_list_start|>'));
      expect(prompt, contains('"name": "TodoList_GetRemaining"'));
      expect(prompt, contains('You are a helpful assistant.'));
      // Messages reach the prompt in the order the caller wrote them; this
      // client no longer reorders any of them.
      expect(
        prompt.indexOf('Hi'),
        lessThan(prompt.indexOf('What is left to do?')),
      );
      expect(prompt.trimRight(), endsWith('<|im_start|>assistant'));
      expect(session.turns, isNull);
    });

    test('passes structured turns to the session for image requests', () async {
      final session = _RecordingSession();
      final client = LlamaChatClient(
        sessionProvider: () async => session,
        format: const Lfm2ChatFormat(),
        contextSize: 4096,
      );
      final imageBytes = Uint8List.fromList([1, 2, 3]);

      await client.getResponse(
        messages: [
          ChatMessage.fromText(ChatRole.assistant, 'How can I help?'),
          ChatMessage(
            role: ChatRole.user,
            contents: [
              TextContent('What is in this picture?'),
              DataContent(imageBytes, mediaType: 'image/png'),
            ],
          ),
        ],
        options: ChatOptions(instructions: 'You are a helpful assistant.'),
        cancellationToken: CancellationToken.none,
      );

      expect(session.media, hasLength(1));
      final turns = session.turns;
      expect(turns, isNotNull);
      expect(turns!.map((turn) => turn.role), [
        LlamaChatRole.system,
        LlamaChatRole.assistant,
        LlamaChatRole.user,
      ]);
      expect(turns.first.text, 'You are a helpful assistant.');
      expect(turns[1].images, isEmpty);
      expect(turns.last.text, 'What is in this picture?');
      expect(turns.last.images.single, same(imageBytes));
      expect(turns.last.audio, isEmpty);
    });

    test('tiles attached images into crops when configured', () async {
      final session = _RecordingSession();
      final client = LlamaChatClient(
        sessionProvider: () async => session,
        format: const Lfm2ChatFormat(),
        contextSize: 4096,
        imageTiling: const ImageTiling(minimumEdge: 10),
      );
      final recorder = ui.PictureRecorder();
      ui.Canvas(recorder).drawRect(
        const ui.Rect.fromLTWH(0, 0, 100, 80),
        ui.Paint()..color = const ui.Color(0xFF336699),
      );
      final image = await recorder.endRecording().toImage(100, 80);
      final imageBytes = (await image.toByteData(
        format: ui.ImageByteFormat.png,
      ))!.buffer.asUint8List();

      await client.getResponse(
        messages: [
          ChatMessage(
            role: ChatRole.user,
            contents: [
              TextContent('Transcribe this page'),
              DataContent(imageBytes, mediaType: 'image/png'),
            ],
          ),
        ],
        cancellationToken: CancellationToken.none,
      );

      // One 2x2 grid of crops: four media entries, four markers in the
      // rendered prompt, and four images on the structured user turn.
      expect(session.media, hasLength(4));
      expect(
        Lfm2ChatTemplate.mediaMarker.allMatches(session.prompt!).length,
        4,
      );
      expect(session.turns!.last.images, hasLength(4));
    });

    test('passes audio media and typed audio turns for Gemma 4', () async {
      final session = _RecordingSession();
      final client = LlamaChatClient(
        sessionProvider: () async => session,
        format: const GemmaChatFormat(),
        contextSize: 4096,
      );
      final audioBytes = Uint8List.fromList([82, 73, 70, 70]); // "RIFF"

      await client.getResponse(
        messages: [
          ChatMessage(
            role: ChatRole.user,
            contents: [
              TextContent('Transcribe this'),
              DataContent(audioBytes, mediaType: 'audio/wav'),
            ],
          ),
        ],
        cancellationToken: CancellationToken.none,
      );

      // The rendered prompt carries one media marker for the clip, and the flat
      // media channel (native mtmd) carries its bytes.
      expect(session.prompt, contains(GemmaChatTemplate.mediaMarker));
      expect(session.media, hasLength(1));
      expect(session.media!.single, same(audioBytes));
      // The message-level path (web/wllama) labels the clip as audio, not image.
      final userTurn = session.turns!.last;
      expect(userTurn.audio.single, same(audioBytes));
      expect(userTurn.images, isEmpty);
    });

    test('rejects audio for a model whose format cannot accept it', () async {
      final session = _RecordingSession();
      final client = LlamaChatClient(
        sessionProvider: () async => session,
        format: const Lfm2ChatFormat(),
        contextSize: 4096,
      );

      await expectLater(
        client.getResponse(
          messages: [
            ChatMessage(
              role: ChatRole.user,
              contents: [
                TextContent('Transcribe this'),
                DataContent(
                  Uint8List.fromList([82, 73, 70, 70]),
                  mediaType: 'audio/wav',
                ),
              ],
            ),
          ],
          cancellationToken: CancellationToken.none,
        ),
        throwsA(isA<UnsupportedError>()),
      );
      // Nothing reached the session: the guard fired before generation.
      expect(session.prompt, isNull);
    });
  });

  group('chatTurnsFromMessages', () {
    test('keeps the tool role typed and skips non-image data', () {
      final turns = chatTurnsFromMessages([
        ChatMessage(
          role: ChatRole.tool,
          contents: [TextContent('{"result": 42}')],
        ),
        ChatMessage(
          role: ChatRole.user,
          contents: [
            DataContent(
              Uint8List.fromList([9, 9]),
              mediaType: 'application/pdf',
            ),
          ],
        ),
      ]);

      expect(turns.first.role, LlamaChatRole.tool);
      expect(turns.first.text, '{"result": 42}');
      expect(turns.last.images, isEmpty);
    });

    test('preserves interleaved text and media ordering', () {
      final first = Uint8List.fromList([1]);
      final second = Uint8List.fromList([2]);
      final turns = chatTurnsFromMessages([
        ChatMessage(
          role: ChatRole.user,
          contents: [
            TextContent('Compare'),
            DataContent(first, mediaType: 'image/png'),
            TextContent('with'),
            DataContent(second, mediaType: 'image/png'),
          ],
        ),
      ]);

      final parts = turns.single.parts;
      expect(parts, hasLength(4));
      expect((parts[0] as LlamaTextPart).text, 'Compare');
      expect((parts[1] as LlamaImagePart).bytes, same(first));
      expect((parts[1] as LlamaImagePart).mimeType, 'image/png');
      expect((parts[2] as LlamaTextPart).text, 'with');
      expect((parts[3] as LlamaImagePart).bytes, same(second));
      expect(turns.single.text, 'Comparewith');
      expect(turns.single.images, [same(first), same(second)]);
    });
  });

  group('formatResolver', () {
    test(
      'format resolved during session load applies to the first request',
      () async {
        final session = _RecordingSession();
        ChatFormat? resolved;
        final client = LlamaChatClient(
          sessionProvider: () async {
            // Simulates the host sniffing the GGUF while loading the model.
            resolved = const GemmaChatFormat();
            return session;
          },
          format: const Lfm2ChatFormat(),
          formatResolver: () => resolved,
          contextSize: 4096,
        );

        await client.getResponse(
          messages: [ChatMessage.fromText(ChatRole.user, 'Hi')],
        );

        expect(session.prompt, contains('<|turn>'));
        expect(session.prompt, isNot(contains('<|im_start|>')));
      },
    );

    test('null resolver result falls back to the constructor format', () async {
      final session = _RecordingSession();
      final client = LlamaChatClient(
        sessionProvider: () async => session,
        format: const Lfm2ChatFormat(),
        formatResolver: () => null,
        contextSize: 4096,
      );

      await client.getResponse(
        messages: [ChatMessage.fromText(ChatRole.user, 'Hi')],
      );

      expect(session.prompt, contains('<|im_start|>'));
    });
  });

  group('usage reporting', () {
    test('emits a trailing usage-only update from runtime stats', () async {
      final session = _RecordingSession(
        stats: const LlamaGenerationStats(
          promptTokenCount: 100,
          cachedTokenCount: 60,
          generatedTokenCount: 25,
        ),
      );
      final client = LlamaChatClient(
        sessionProvider: () async => session,
        format: const Lfm2ChatFormat(),
        contextSize: 4096,
      );

      final updates = await client
          .getStreamingResponse(
            messages: [ChatMessage.fromText(ChatRole.user, 'Hi')],
          )
          .toList();

      final usage = updates.last.usage;
      expect(usage, isNotNull);
      expect(usage!.inputTokenCount, 100);
      expect(usage.outputTokenCount, 25);
      expect(usage.totalTokenCount, 125);
      expect(usage.cachedInputTokenCount, 60);
      expect(usage.additionalCounts, isNull);
      expect(updates.last.finishReason, isNull);
    });

    test('surfaces finish reason and timing counts when reported', () async {
      final session = _RecordingSession(
        stats: const LlamaGenerationStats(
          promptTokenCount: 100,
          cachedTokenCount: 60,
          generatedTokenCount: 25,
          finishReason: LlamaFinishReason.eogToken,
          prefillDuration: Duration(milliseconds: 120),
          decodeDuration: Duration(milliseconds: 500),
          draftedTokenCount: 30,
          acceptedTokenCount: 21,
        ),
      );
      final client = LlamaChatClient(
        sessionProvider: () async => session,
        format: const Lfm2ChatFormat(),
        contextSize: 4096,
      );

      final updates = await client
          .getStreamingResponse(
            messages: [ChatMessage.fromText(ChatRole.user, 'Hi')],
          )
          .toList();

      expect(updates.last.finishReason, ChatFinishReason.stop);
      expect(updates.last.usage?.additionalCounts, {
        'prefillMicroseconds': 120000,
        'decodeMicroseconds': 500000,
        'draftedTokenCount': 30,
        'acceptedTokenCount': 21,
      });
    });

    test(
      'getResponse folds the trailing usage into ChatResponse.usage',
      () async {
        final session = _RecordingSession(
          stats: const LlamaGenerationStats(
            promptTokenCount: 40,
            cachedTokenCount: 0,
            generatedTokenCount: 8,
            finishReason: LlamaFinishReason.maxTokens,
          ),
        );
        final client = LlamaChatClient(
          sessionProvider: () async => session,
          format: const Lfm2ChatFormat(),
          contextSize: 4096,
        );

        final response = await client.getResponse(
          messages: [ChatMessage.fromText(ChatRole.user, 'Hi')],
        );

        expect(response.usage?.inputTokenCount, 40);
        expect(response.usage?.outputTokenCount, 8);
        expect(response.finishReason, ChatFinishReason.length);
        expect(response.text, 'Hello!');
      },
    );

    test('emits no usage update when the runtime reports none', () async {
      final session = _RecordingSession();
      final client = LlamaChatClient(
        sessionProvider: () async => session,
        format: const Lfm2ChatFormat(),
        contextSize: 4096,
      );

      final updates = await client
          .getStreamingResponse(
            messages: [ChatMessage.fromText(ChatRole.user, 'Hi')],
          )
          .toList();

      expect(updates.every((u) => u.usage == null), isTrue);
    });
  });

  group('cancellation', () {
    test('forwards a cancellation request to the session', () async {
      final session = _CancellableSession();
      final client = LlamaChatClient(
        sessionProvider: () async => session,
        format: const Lfm2ChatFormat(),
        contextSize: 4096,
      );
      final source = CancellationTokenSource();

      final done = client
          .getStreamingResponse(
            messages: [ChatMessage.fromText(ChatRole.user, 'Hi')],
            cancellationToken: source.token,
          )
          .drain<void>();
      // Let the generator reach the token stream, then cancel mid-flight.
      await pumpEventQueue();
      expect(session.generateCalled, isTrue);
      source.cancel();
      await done;

      expect(session.cancelCalled, isTrue);
    });

    test('an already-cancelled token skips generation entirely', () async {
      final session = _CancellableSession();
      final client = LlamaChatClient(
        sessionProvider: () async => session,
        format: const Lfm2ChatFormat(),
        contextSize: 4096,
      );
      final source = CancellationTokenSource()..cancel();

      final updates = await client
          .getStreamingResponse(
            messages: [ChatMessage.fromText(ChatRole.user, 'Hi')],
            cancellationToken: source.token,
          )
          .toList();

      expect(updates, isEmpty);
      expect(session.generateCalled, isFalse);
    });
  });
}
