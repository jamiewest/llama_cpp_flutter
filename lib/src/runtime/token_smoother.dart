import 'dart:async';
import 'dart:collection';
import 'dart:math' as math;

import 'package:characters/characters.dart';

/// Whether [chunk] should be released whole rather than paced grapheme by
/// grapheme. The default treats every chunk as ordinary text.
typedef AtomicChunkPredicate = bool Function(String chunk);

bool _neverAtomic(String chunk) => false;

/// Re-paces a token stream into a steady, typewriter-style character stream.
///
/// Token streams arrive in bursts — a prompt-processing pause, then several
/// tokens in one frame, then nothing while the next batch decodes — which
/// makes streamed text jump and stutter on screen. [TokenSmoother] buffers
/// incoming chunks and releases them a grapheme at a time on a frame timer,
/// at a rate that continuously tracks how fast text is actually arriving.
///
/// The release rate is **not** derived from the backlog alone. Draining
/// whatever is buffered over a fixed interval empties the buffer during a
/// decode pause and then stalls, which just relocates the stutter. Instead
/// the smoother estimates the arrival rate and releases at that rate, using
/// the backlog only as a correction:
///
/// ```text
/// arrival  ← arrival + α·(unitsThisTick − arrival)        // rate estimate
/// lead     ← arrival · framesPerWindow                    // buffer to hold
/// target   ← arrival + (backlog − lead) / correctionFrames
/// release  ← release + α·(target − release)               // rate is smoothed
/// ```
///
/// In steady state `backlog == lead` and the output rate equals the input
/// rate, so a 5 tok/s model and a 60 tok/s model both render evenly — only
/// the speed differs. When generation speeds up or slows down, the backlog
/// term pulls the release rate toward the new arrival rate over roughly
/// [smoothing], so the change is a ramp rather than a jump. Lag behind the
/// source settles around [window].
///
/// Because the rate is fractional and accumulated across ticks, the smoother
/// can emit slower than one grapheme per [tick] — that is what lets it match
/// a slow model instead of outrunning it.
///
/// This is a presentation concern, so it is opt-in: apply it at the widget
/// that renders the text, not inside generation. Headless callers (batch
/// runs, orchestration, tests) should leave the stream unpaced.
///
/// ```dart
/// final display = session.generate(prompt).smoothed();
/// ```
///
/// A chunk for which [atomic] returns true is never split; it is released
/// whole in a single emission. Use this for in-band markers or control
/// sequences that must not appear half-rendered.
///
/// The transformer is single-subscription. Cancelling the output subscription
/// cancels the upstream subscription, so stopping a generation propagates
/// back to the runtime.
class TokenSmoother extends StreamTransformerBase<String, String> {
  /// Creates a token stream smoother.
  const TokenSmoother({
    this.tick = const Duration(milliseconds: 16),
    this.window = const Duration(milliseconds: 250),
    this.smoothing = const Duration(milliseconds: 150),
    this.atomic = _neverAtomic,
  });

  /// How often the release budget is recomputed. One frame by default.
  final Duration tick;

  /// How much text the smoother aims to keep buffered, expressed as time at
  /// the current arrival rate — and so also the steady-state lag behind the
  /// source. Larger values ride out longer decode pauses at the cost of the
  /// text trailing further behind generation.
  final Duration window;

  /// Time constant of the rate filter: how quickly the release rate follows
  /// a change in generation speed. Larger values are steadier but slower to
  /// react to the model speeding up or slowing down.
  final Duration smoothing;

  /// Decides which chunks are emitted whole instead of paced.
  final AtomicChunkPredicate atomic;

  @override
  Stream<String> bind(Stream<String> stream) {
    final framesPerWindow = math.max(
      1,
      (window.inMicroseconds / tick.inMicroseconds).ceil(),
    );
    // The backlog correction must act over a longer horizon than the lead it
    // is correcting toward, otherwise `arrival + (backlog - arrival·F)/F`
    // collapses to `backlog/F` — the feed-forward term cancels exactly and
    // the smoother degenerates into a fixed-interval backlog drainer.
    final correctionFrames = 2 * framesPerWindow;
    // Per-tick EMA coefficient for a `smoothing` time constant.
    final alpha = smoothing <= Duration.zero
        ? 1.0
        : 1 - math.exp(-tick.inMicroseconds / smoothing.inMicroseconds);

    final queue = Queue<String>();
    StreamSubscription<String>? upstream;
    Timer? timer;
    var sourceDone = false;
    // Units queued since the last tick; the raw signal for the rate estimate.
    var arrivedSinceTick = 0;
    // Units per tick, exponentially smoothed.
    var arrivalRate = 0.0;
    var releaseRate = 0.0;
    // Fractional carry, so a rate below one unit per tick still makes
    // progress instead of rounding away to nothing.
    var credit = 0.0;
    // Ticks left to fully drain once the source is done. Counting it down
    // bounds the tail at `window` without jolting the pace at the handover.
    var tailFrames = framesPerWindow;
    late StreamController<String> controller;

    void updateRates() {
      arrivalRate += alpha * (arrivedSinceTick - arrivalRate);
      arrivedSinceTick = 0;

      final lead = arrivalRate * framesPerWindow;
      final target = arrivalRate + (queue.length - lead) / correctionFrames;
      releaseRate += alpha * (target - releaseRate);
      if (releaseRate < 0) releaseRate = 0;

      if (sourceDone && queue.isNotEmpty) {
        tailFrames = math.max(1, tailFrames - 1);
        releaseRate = math.max(releaseRate, queue.length / tailFrames);
      }
    }

    void emitBudget() {
      if (queue.isEmpty) {
        credit = 0;
        return;
      }
      credit += releaseRate;
      final budget = math.min(credit.floor(), queue.length);
      if (budget <= 0) return;
      credit -= budget;
      final out = StringBuffer();
      for (var i = 0; i < budget; i++) {
        out.write(queue.removeFirst());
      }
      if (queue.isEmpty) credit = 0;
      controller.add(out.toString());
    }

    void stopTimer() {
      timer?.cancel();
      timer = null;
    }

    void onTick(Timer _) {
      updateRates();
      emitBudget();
      if (queue.isEmpty && sourceDone) {
        stopTimer();
        controller.close();
      }
    }

    void onData(String chunk) {
      if (chunk.isEmpty) return;
      if (atomic(chunk)) {
        queue.add(chunk);
        arrivedSinceTick++;
      } else {
        for (final grapheme in chunk.characters) {
          queue.add(grapheme);
          arrivedSinceTick++;
        }
      }
    }

    void onDone() {
      sourceDone = true;
      if (queue.isEmpty) {
        stopTimer();
        controller.close();
      }
    }

    controller = StreamController<String>(
      onListen: () {
        // The timer runs for the whole subscription, not just while the
        // queue is non-empty: the rate estimate has to keep decaying through
        // a decode pause, or the first chunk after the pause would be paced
        // against a stale, much higher rate and dump.
        timer = Timer.periodic(tick, onTick);
        upstream = stream.listen(
          onData,
          onError: (Object error, StackTrace stackTrace) {
            // Buffered text is dropped — it belongs to a run that failed —
            // but the timer keeps running: an error need not end the stream,
            // and stopping it here would strand anything sent afterwards.
            queue.clear();
            credit = 0;
            arrivalRate = 0;
            releaseRate = 0;
            controller.addError(error, stackTrace);
          },
          onDone: onDone,
        );
      },
      onCancel: () {
        stopTimer();
        final sub = upstream;
        upstream = null;
        return sub?.cancel();
      },
    );

    return controller.stream;
  }
}

/// Convenience methods for smoothing text streams.
extension SmoothedStream on Stream<String> {
  /// Returns this stream re-paced by a [TokenSmoother].
  Stream<String> smoothed({
    Duration tick = const Duration(milliseconds: 16),
    Duration window = const Duration(milliseconds: 250),
    Duration smoothing = const Duration(milliseconds: 150),
    AtomicChunkPredicate atomic = _neverAtomic,
  }) {
    return transform(
      TokenSmoother(
        tick: tick,
        window: window,
        smoothing: smoothing,
        atomic: atomic,
      ),
    );
  }
}
