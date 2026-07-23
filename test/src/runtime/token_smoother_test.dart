import 'dart:async';
import 'dart:math' as math;

import 'package:characters/characters.dart';
import 'package:fake_async/fake_async.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:llama_cpp_flutter/chat.dart';

void main() {
  group('TokenSmoother', () {
    test('emits bursty chunks as paced graphemes', () async {
      final chunks = await Stream.fromIterable([
        'abc',
      ]).smoothed(tick: _tick, window: _wideWindow).toList();

      expect(chunks.join(), 'abc');
      expect(chunks.length, greaterThan(1));
    });

    test('keeps multi-codepoint grapheme clusters intact', () async {
      final chunks = await Stream.fromIterable([
        '👨‍👩‍👧‍👦!',
      ]).smoothed(tick: _tick, window: _wideWindow).toList();

      expect(chunks.join(), '👨‍👩‍👧‍👦!');
      expect(chunks, contains('👨‍👩‍👧‍👦'));
    });

    test('emits atomic chunks whole', () async {
      final chunks = await Stream.fromIterable(['<marker>'])
          .smoothed(
            tick: _tick,
            window: _wideWindow,
            atomic: (chunk) => chunk.startsWith('<'),
          )
          .toList();

      expect(chunks, ['<marker>']);
    });

    test('waits for buffered text to drain before completing', () async {
      final controller = StreamController<String>();
      final done = Completer<void>();
      var isDone = false;
      final chunks = <String>[];
      controller.stream
          .smoothed(tick: _tick, window: _wideWindow)
          .listen(
            chunks.add,
            onDone: () {
              isDone = true;
              done.complete();
            },
          );

      controller.add('ab');
      await controller.close();

      expect(isDone, isFalse);

      await done.future;

      expect(chunks.join(), 'ab');
      expect(isDone, isTrue);
    });

    test('cancels the upstream subscription when canceled', () async {
      var canceled = false;
      final controller = StreamController<String>(
        onCancel: () {
          canceled = true;
        },
      );
      final sub = controller.stream
          .smoothed(tick: _tick, window: _wideWindow)
          .listen((_) {});

      await sub.cancel();

      expect(canceled, isTrue);
      await controller.close();
    });

    test('propagates upstream errors and drops buffered text', () async {
      final controller = StreamController<String>();
      final error = Completer<Object>();
      final chunks = <String>[];
      controller.stream
          .smoothed(tick: _tick, window: _wideWindow)
          .listen(chunks.add, onError: error.complete);

      controller.add('abc');
      controller.addError(StateError('boom'));

      expect(await error.future, isA<StateError>());
      expect(chunks, isEmpty);
      await controller.close();
    });

    test('keeps delivering text sent after a non-terminal error', () async {
      final controller = StreamController<String>();
      final error = Completer<Object>();
      final chunks = <String>[];
      final done = Completer<void>();
      controller.stream
          .smoothed(tick: _tick, window: _wideWindow)
          .listen(chunks.add, onError: error.complete, onDone: done.complete);

      controller.addError(StateError('boom'));
      await error.future;
      controller.add('after');
      await controller.close();
      await done.future;

      expect(chunks.join(), 'after');
    });

    group('rate tracking', () {
      test('paces a slow steady source evenly instead of stalling', () {
        // 4 graphemes every 200ms — one "token" per 200ms, the shape a
        // 5 tok/s model produces. A backlog drainer empties the buffer well
        // inside the 200ms and then stalls, so gaps alternate short/long.
        final emitted = _run(
          chunkCount: 30,
          chunk: 'abcd',
          spacing: const Duration(milliseconds: 200),
        );

        expect(emitted.map((e) => e.grapheme).join(), 'abcd' * 30);

        // Drop the leading ramp-up and the tail drain; measure the middle.
        final steady = emitted.sublist(20, emitted.length - 20);
        final gaps = _gaps(steady);
        final median = _median(gaps);

        expect(median, greaterThan(0));
        expect(
          gaps.reduce(math.max),
          lessThan(median * 2.5),
          reason:
              'gaps between graphemes should be near-uniform; a stalling '
              'pacer shows a long gap once per source chunk',
        );
      });

      test('keeps lag behind the source bounded', () {
        final emitted = _run(
          chunkCount: 30,
          chunk: 'abcd',
          spacing: const Duration(milliseconds: 200),
        );

        // At each source chunk boundary, how far behind is the output?
        var worstLag = 0;
        for (var i = 1; i <= 30; i++) {
          final at = i * 200;
          final delivered = i * 4;
          final shown = emitted.where((e) => e.atMs <= at).length;
          worstLag = math.max(worstLag, delivered - shown);
        }

        // One `window` of lead at 20 graphemes/sec is ~5 graphemes.
        expect(worstLag, lessThan(15));
      });

      test('speeds up when generation speeds up', () {
        // Same total text: 15 slow chunks, then 15 four times faster.
        final emitted = _run(
          chunkCount: 15,
          chunk: 'abcd',
          spacing: const Duration(milliseconds: 200),
          thenChunkCount: 15,
          thenSpacing: const Duration(milliseconds: 50),
        );

        expect(emitted.map((e) => e.grapheme).join(), 'abcd' * 30);

        final slowPhase = emitted.where((e) => e.atMs < 3000).toList();
        final fastPhase = emitted
            .where((e) => e.atMs >= 3000 && e.atMs < 3600)
            .toList();

        final slowRate = _rate(slowPhase);
        final fastRate = _rate(fastPhase);

        expect(
          fastRate,
          greaterThan(slowRate * 2),
          reason: 'release rate should follow the arrival rate up',
        );
      });

      test('drains the tail within about one window of the source ending', () {
        final emitted = _run(
          chunkCount: 10,
          chunk: 'abcd',
          spacing: const Duration(milliseconds: 200),
        );

        // The source closes right after the last chunk at t=2000ms.
        expect(emitted.last.atMs - 2000, lessThan(500));
      });
    });
  });
}

const _tick = Duration(milliseconds: 1);
const _wideWindow = Duration(milliseconds: 100);

/// One grapheme and the fake-clock time it reached the listener.
typedef _Emission = ({int atMs, String grapheme});

/// Feeds [chunkCount] copies of [chunk] spaced [spacing] apart (optionally
/// followed by a second phase at a different pace) through a default-tuned
/// smoother on a fake clock, and returns every grapheme with its timestamp.
List<_Emission> _run({
  required int chunkCount,
  required String chunk,
  required Duration spacing,
  int thenChunkCount = 0,
  Duration thenSpacing = Duration.zero,
}) {
  final emitted = <_Emission>[];
  FakeAsync().run((async) {
    final controller = StreamController<String>();
    controller.stream.smoothed().listen((out) {
      for (final grapheme in out.characters) {
        emitted.add((atMs: async.elapsed.inMilliseconds, grapheme: grapheme));
      }
    });
    async.flushMicrotasks();

    for (var i = 0; i < chunkCount; i++) {
      controller.add(chunk);
      async.elapse(spacing);
    }
    for (var i = 0; i < thenChunkCount; i++) {
      controller.add(chunk);
      async.elapse(thenSpacing);
    }
    unawaited(controller.close());
    async.elapse(const Duration(seconds: 5));
  });
  return emitted;
}

List<int> _gaps(List<_Emission> emitted) => [
  for (var i = 1; i < emitted.length; i++)
    emitted[i].atMs - emitted[i - 1].atMs,
];

double _median(List<int> values) {
  final sorted = [...values]..sort();
  return sorted[sorted.length ~/ 2].toDouble();
}

/// Graphemes per second across the span [emitted] covers.
double _rate(List<_Emission> emitted) {
  final span = emitted.last.atMs - emitted.first.atMs;
  return span == 0 ? double.infinity : emitted.length * 1000 / span;
}
