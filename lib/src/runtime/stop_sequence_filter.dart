/// Filters a text stream so generation stops at the first stop sequence.
final class StopSequenceFilter {
  /// Creates a filter for [stopSequences].
  ///
  /// [onStop] is invoked once when a stop sequence match terminates the
  /// stream, letting the caller distinguish a stop-terminated run from one
  /// that ended naturally (e.g. to abort the underlying generation).
  const StopSequenceFilter(this.stopSequences, {this.onStop});

  /// Stop markers that must be stripped from output.
  final List<String> stopSequences;

  /// Invoked when a stop sequence match ends the stream.
  final void Function()? onStop;

  /// Applies this filter to [tokens].
  Stream<String> bind(Stream<String> tokens) async* {
    final stops = stopSequences.where((s) => s.isNotEmpty).toList();
    if (stops.isEmpty) {
      yield* tokens;
      return;
    }

    final holdback =
        stops.map((s) => s.length).reduce((a, b) => a > b ? a : b) - 1;
    var buffer = '';

    await for (final token in tokens) {
      buffer += token;
      final stopAt = _firstStop(buffer, stops);
      if (stopAt != null) {
        onStop?.call();
        final emit = buffer.substring(0, stopAt);
        if (emit.isNotEmpty) yield emit;
        return;
      }

      if (buffer.length > holdback) {
        final emit = buffer.substring(0, buffer.length - holdback);
        if (emit.isNotEmpty) yield emit;
        buffer = buffer.substring(buffer.length - holdback);
      }
    }

    if (buffer.isNotEmpty) yield buffer;
  }

  static int? _firstStop(String buffer, List<String> stops) {
    int? best;
    for (final stop in stops) {
      final index = buffer.indexOf(stop);
      if (index >= 0 && (best == null || index < best)) {
        best = index;
      }
    }
    return best;
  }
}
