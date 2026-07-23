import 'dart:async';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:llama_cpp_flutter/llama_cpp_flutter.dart';

/// Fake session whose [generate] replays [pieces], reports [stats] through
/// `onStats` before closing (when set), or fails with [error].
final class _FakeSession implements LlamaSession {
  _FakeSession({this.pieces = const <String>[], this.stats, this.error});

  final List<String> pieces;
  final LlamaGenerationStats? stats;
  final Object? error;

  bool cancelled = false;

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
    final controller = StreamController<String>(
      onCancel: () => cancelled = true,
    );
    Future<void>.microtask(() async {
      for (final piece in pieces) {
        if (cancelled) return;
        controller.add(piece);
      }
      if (error != null) {
        controller.addError(error!);
      } else if (stats != null) {
        onStats?.call(stats!);
      }
      await controller.close();
    });
    return controller.stream;
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

void main() {
  const stats = LlamaGenerationStats(
    promptTokenCount: 12,
    cachedTokenCount: 4,
    generatedTokenCount: 2,
    finishReason: LlamaFinishReason.eogToken,
  );

  group('generateEvents', () {
    test('streams text events and ends with completed stats', () async {
      final session = _FakeSession(pieces: ['Hel', 'lo'], stats: stats);

      final events = await session.generateEvents('hi').toList();

      expect(events, hasLength(3));
      expect((events[0] as LlamaTextEvent).text, 'Hel');
      expect((events[1] as LlamaTextEvent).text, 'lo');
      final completed = events.last as LlamaCompletedEvent;
      expect(completed.stats, stats);
      expect(completed.stats!.finishReason, LlamaFinishReason.eogToken);
    });

    test('completed event is guaranteed even without engine stats', () async {
      final session = _FakeSession(pieces: ['ok']);

      final events = await session.generateEvents('hi').toList();

      expect(events.last, isA<LlamaCompletedEvent>());
      expect((events.last as LlamaCompletedEvent).stats, isNull);
    });

    test('a failed run errors without a completed event', () async {
      final session = _FakeSession(pieces: ['par'], error: StateError('boom'));

      final events = <LlamaGenerationEvent>[];
      Object? caught;
      try {
        await for (final event in session.generateEvents('hi')) {
          events.add(event);
        }
      } on StateError catch (e) {
        caught = e;
      }

      expect((events.single as LlamaTextEvent).text, 'par');
      expect(caught, isA<StateError>());
      expect(events.whereType<LlamaCompletedEvent>(), isEmpty);
    });

    test('cancelling the subscription cancels the underlying run', () async {
      final session = _FakeSession(pieces: List.filled(100, 'x'));

      final subscription = session.generateEvents('hi').listen(null);
      await subscription.cancel();

      expect(session.cancelled, isTrue);
    });
  });
}
