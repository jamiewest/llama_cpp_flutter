import 'dart:async';
import 'dart:isolate';

import 'package:extensions/logging.dart';
import 'package:flutter/services.dart';

import 'llama_worker.dart';
import 'messages.g.dart';

/// Why a generation run stopped.
enum LlamaFinishReason {
  /// The model produced an end-of-generation token.
  eogToken,

  /// The run reached its `maxTokens` cap.
  maxTokens,

  /// A stop sequence matched.
  stopSequence,

  /// The context window filled before the run ended naturally.
  contextFull,

  /// The run was cancelled.
  cancelled,

  /// The native runtime ended the run early after an unrecoverable internal
  /// condition; text streamed before this point is valid.
  aborted,
}

/// Token accounting and timing for one completed generation run.
class LlamaGenerationStats {
  /// Creates a [LlamaGenerationStats].
  const LlamaGenerationStats({
    required this.promptTokenCount,
    required this.cachedTokenCount,
    required this.generatedTokenCount,
    required this.finishReason,
    required this.prefillDuration,
    required this.decodeDuration,
    this.draftedTokenCount,
    this.acceptedTokenCount,
  });

  /// Prompt tokens fed to the model (including the reused prefix).
  final int promptTokenCount;

  /// Prompt tokens served from the reused KV-cache prefix.
  final int cachedTokenCount;

  /// Tokens generated.
  final int generatedTokenCount;

  /// Why the run stopped.
  final LlamaFinishReason finishReason;

  /// Wall-clock time spent ingesting the prompt.
  final Duration prefillDuration;

  /// Wall-clock time spent generating tokens.
  final Duration decodeDuration;

  /// Tokens proposed by the draft model; null when the session runs without
  /// speculative decoding.
  final int? draftedTokenCount;

  /// Drafted tokens the main model accepted; null when the session runs
  /// without speculative decoding.
  final int? acceptedTokenCount;

  /// Prompt-ingestion speed over the tokens that were actually decoded
  /// (prompt minus cached), in tokens per second.
  double get prefillTokensPerSecond =>
      _rate(promptTokenCount - cachedTokenCount, prefillDuration);

  /// Generation speed, in tokens per second.
  double get decodeTokensPerSecond =>
      _rate(generatedTokenCount, decodeDuration);

  static double _rate(int tokens, Duration elapsed) {
    final seconds = elapsed.inMicroseconds / Duration.microsecondsPerSecond;
    return seconds > 0 ? tokens / seconds : 0;
  }
}

/// Maps the wire-level [FinishReason] onto the public enum.
LlamaFinishReason _finishReason(FinishReason reason) => switch (reason) {
  FinishReason.eogToken => LlamaFinishReason.eogToken,
  FinishReason.maxTokens => LlamaFinishReason.maxTokens,
  FinishReason.stopSequence => LlamaFinishReason.stopSequence,
  FinishReason.contextFull => LlamaFinishReason.contextFull,
  FinishReason.cancelled => LlamaFinishReason.cancelled,
  FinishReason.aborted => LlamaFinishReason.aborted,
};

/// Owns the worker isolate and routes commands/replies between it and the
/// main isolate. One instance backs a single [LlamaFlutter] facade.
class LlamaIsolate {
  /// Creates the isolate owner. [loggerFactory] supplies the logger used for
  /// worker lifecycle and generation diagnostics; null logs nothing.
  LlamaIsolate({LoggerFactory? loggerFactory})
    : _logger = (loggerFactory ?? NullLoggerFactory.instance).createLogger(
        'LlamaIsolate',
      );

  final Logger _logger;
  final ReceivePort _fromWorker = ReceivePort();
  final Completer<void> _ready = Completer<void>();
  final Map<int, Completer<int>> _loads = <int, Completer<int>>{};
  final Map<int, Completer<int>> _stateOps = <int, Completer<int>>{};
  final Map<int, Completer<({int tokens, int bytes})>> _stashOps =
      <int, Completer<({int tokens, int bytes})>>{};
  final Map<int, StreamController<String>> _generations =
      <int, StreamController<String>>{};
  final Map<int, void Function(LlamaGenerationStats)> _statsCallbacks =
      <int, void Function(LlamaGenerationStats)>{};

  /// Generations requested while the same session's previous run was still in
  /// flight, started once that run's `done` event arrives. Tokens carry only a
  /// sessionId, so starting the new run earlier would route the old run's
  /// remaining events (including its `done`, which closes the stream) into the
  /// new controller.
  final Map<int, _PendingGeneration> _pending = <int, _PendingGeneration>{};

  SendPort? _commands;
  StreamSubscription<TokenEvent>? _tokens;
  int _nextRequestId = 1;
  Future<void>? _starting;

  /// Spawns the worker and waits for its handshake.
  ///
  /// Safe to call concurrently: every caller awaits the same startup, so a
  /// second `loadModel` issued while the worker is still spawning cannot
  /// observe a half-initialized isolate.
  Future<void> start() => _starting ??= _start();

  Future<void> _start() async {
    final token = RootIsolateToken.instance;
    if (token == null) {
      throw StateError(
        'RootIsolateToken unavailable; call from the root isolate.',
      );
    }

    _fromWorker.listen(_handleMessage);
    _logger.logDebug('Spawning llama worker isolate.');
    await Isolate.spawn(
      llamaWorkerMain,
      WorkerInit(token, _fromWorker.sendPort),
    );
    await _ready.future;
    _logger.logDebug('Llama worker isolate ready.');

    // The token EventChannel must be listened to on the root isolate; a
    // background isolate cannot register a platform message handler. Native
    // generation runs on the worker's command calls, but its TokenEvents are
    // delivered here and routed to the per-session controllers.
    _tokens = streamTokens().listen(_handleTokenEvent);
  }

  /// Loads a model and resolves to its session id.
  Future<int> loadModel(ModelLoadRequest request) async {
    final requestId = _nextRequestId++;
    final completer = Completer<int>();
    _loads[requestId] = completer;
    _commands!.send(LoadCommand(requestId, request));
    return completer.future;
  }

  /// Starts generation and returns a stream of decoded token text.
  ///
  /// If the session's previous run is still in flight it is cancelled, and
  /// the new run starts only once its `done` event arrives (see [_pending]).
  ///
  /// [onStats] is invoked once with the run's token accounting when the
  /// native side reports it on the `done` event.
  Stream<String> generate(
    GenerationRequest request, {
    void Function(LlamaGenerationStats)? onStats,
  }) {
    late final StreamController<String> controller;
    controller = StreamController<String>(
      onCancel: () {
        if (_pending[request.sessionId]?.controller == controller) {
          _pending.remove(request.sessionId);
        } else {
          _commands!.send(CancelCommand(request.sessionId));
        }
      },
    );

    if (_generations.containsKey(request.sessionId)) {
      _pending.remove(request.sessionId)?.controller.close();
      _pending[request.sessionId] = (
        request: request,
        controller: controller,
        onStats: onStats,
      );
      _commands!.send(CancelCommand(request.sessionId));
    } else {
      _generations[request.sessionId] = controller;
      if (onStats != null) _statsCallbacks[request.sessionId] = onStats;
      _commands!.send(GenerateCommand(request));
    }
    return controller.stream;
  }

  Future<void> cancel(int sessionId) async {
    _commands!.send(CancelCommand(sessionId));
  }

  /// Saves (or, with `save: false`, restores) one sequence's KV-cache
  /// state.
  ///
  /// Resolves to the snapshot's token count. Worker commands are serviced in
  /// order, so a state operation issued before a [generate] runs before it.
  Future<int> sessionState(
    int sessionId,
    String path, {
    required bool save,
    int sequenceId = 0,
  }) {
    final requestId = _nextRequestId++;
    final completer = Completer<int>();
    _stateOps[requestId] = completer;
    _commands!.send(
      StateCommand(
        requestId,
        sessionId,
        path,
        save: save,
        sequenceId: sequenceId,
      ),
    );
    return completer.future;
  }

  /// Measures the byte size of one sequence's current state (KV cache plus
  /// bookkeeping) without writing it anywhere.
  Future<int> stateSize(int sessionId, {int sequenceId = 0}) {
    final requestId = _nextRequestId++;
    final completer = Completer<int>();
    _stateOps[requestId] = completer;
    _commands!.send(
      StateSizeCommand(requestId, sessionId, sequenceId: sequenceId),
    );
    return completer.future;
  }

  /// Stashes one sequence's KV state in native memory under [key].
  ///
  /// Resolves to the stash's (tokens, bytes); `(0, 0)` when the sequence
  /// held nothing reusable.
  Future<({int tokens, int bytes})> stashState(
    int sessionId,
    int sequenceId,
    String key,
  ) {
    final requestId = _nextRequestId++;
    final completer = Completer<({int tokens, int bytes})>();
    _stashOps[requestId] = completer;
    _commands!.send(StashCommand(requestId, sessionId, sequenceId, key));
    return completer.future;
  }

  /// Restores a stashed KV state into a sequence; resolves to the tokens
  /// restored.
  Future<int> restoreStashedState(int sessionId, int sequenceId, String key) {
    final requestId = _nextRequestId++;
    final completer = Completer<int>();
    _stateOps[requestId] = completer;
    _commands!.send(RestoreStashCommand(requestId, sessionId, sequenceId, key));
    return completer.future;
  }

  /// Drops a stashed KV state; resolves to the bytes freed.
  Future<int> dropStashedState(int sessionId, String key) {
    final requestId = _nextRequestId++;
    final completer = Completer<int>();
    _stateOps[requestId] = completer;
    _commands!.send(DropStashCommand(requestId, sessionId, key));
    return completer.future;
  }

  /// Changes the session's vision-encoder token budget; null restores the
  /// model's metadata default.
  Future<void> setImageTokenBudget(int sessionId, int? imageTokenBudget) {
    final requestId = _nextRequestId++;
    final completer = Completer<int>();
    _stateOps[requestId] = completer;
    _commands!.send(
      SetImageBudgetCommand(requestId, sessionId, imageTokenBudget),
    );
    return completer.future;
  }

  /// Erases one sequence's KV cache and prompt ledger.
  Future<void> clearSequence(int sessionId, int sequenceId) {
    final requestId = _nextRequestId++;
    final completer = Completer<int>();
    _stateOps[requestId] = completer;
    _commands!.send(ClearSequenceCommand(requestId, sessionId, sequenceId));
    return completer.future;
  }

  Future<void> disposeSession(int sessionId) async {
    _generations.remove(sessionId)?.close();
    _statsCallbacks.remove(sessionId);
    _pending.remove(sessionId)?.controller.close();
    _commands!.send(DisposeCommand(sessionId));
  }

  /// Shuts the worker down and releases all stream controllers.
  Future<void> dispose() async {
    _logger.logDebug('Shutting down llama worker isolate.');
    await _tokens?.cancel();
    _commands?.send(const ShutdownCommand());
    for (final controller in _generations.values) {
      await controller.close();
    }
    _generations.clear();
    _statsCallbacks.clear();
    for (final pending in _pending.values) {
      await pending.controller.close();
    }
    _pending.clear();
    _fromWorker.close();
  }

  void _handleMessage(Object? message) {
    switch (message) {
      case SendPort commandPort:
        _commands = commandPort;
        if (!_ready.isCompleted) _ready.complete();
      case LoadResult(:final requestId, :final sessionId, :final error):
        final completer = _loads.remove(requestId);
        if (completer == null) return;
        if (error != null) {
          _logger.logError('Model load failed.', error: LlamaException(error));
          completer.completeError(LlamaException(error));
        } else {
          _logger.logInformation('Model loaded (session $sessionId).');
          completer.complete(sessionId);
        }
      case StateResult(:final requestId, :final value, :final error):
        final completer = _stateOps.remove(requestId);
        if (completer == null) return;
        if (error != null) {
          completer.completeError(LlamaException(error));
        } else {
          completer.complete(value);
        }
      case StashCommandResult(
        :final requestId,
        :final tokens,
        :final bytes,
        :final error,
      ):
        final completer = _stashOps.remove(requestId);
        if (completer == null) return;
        if (error != null) {
          completer.completeError(LlamaException(error));
        } else {
          completer.complete((tokens: tokens ?? 0, bytes: bytes ?? 0));
        }
      case GenerateFailed(:final sessionId, :final error):
        // The generate command itself failed (a platform-channel error), so
        // no token stream — and no `done` event — will ever arrive for this
        // run. Error the stream instead of leaving it open forever.
        _logger.logError(
          'Generation failed to start (session $sessionId).',
          error: LlamaException(error),
        );
        _statsCallbacks.remove(sessionId);
        _generations.remove(sessionId)
          ?..addError(LlamaException(error))
          ..close();
        _startPending(sessionId);
    }
  }

  /// Routes a native token event to the controller for its session.
  void _handleTokenEvent(TokenEvent event) {
    final controller = _generations[event.sessionId];
    if (controller == null) return;
    if (event.error != null) {
      _logger.logError(
        'Generation errored (session ${event.sessionId}).',
        error: LlamaException(event.error!),
      );
      controller.addError(LlamaException(event.error!));
    } else if (event.text != null) {
      controller.add(event.text!);
    }
    if (event.done) {
      _generations.remove(event.sessionId);
      final onStats = _statsCallbacks.remove(event.sessionId);
      final stats = event.stats;
      if (stats != null && event.error == null) {
        _logger.logDebug(
          'Generation finished (session ${event.sessionId}): '
          '${stats.promptTokenCount} prompt '
          '(${stats.cachedTokenCount} cached), '
          '${stats.generatedTokenCount} generated, '
          '${stats.finishReason.name}.',
        );
      }
      if (onStats != null && event.error == null && stats != null) {
        onStats(
          LlamaGenerationStats(
            promptTokenCount: stats.promptTokenCount,
            cachedTokenCount: stats.cachedTokenCount,
            generatedTokenCount: stats.generatedTokenCount,
            finishReason: _finishReason(stats.finishReason),
            prefillDuration: Duration(microseconds: stats.prefillMicroseconds),
            decodeDuration: Duration(microseconds: stats.decodeMicroseconds),
            draftedTokenCount: stats.draftedTokenCount,
            acceptedTokenCount: stats.acceptedTokenCount,
          ),
        );
      }
      controller.close();
      _startPending(event.sessionId);
    }
  }

  /// Starts the queued follow-up run for [sessionId], if any.
  void _startPending(int sessionId) {
    final pending = _pending.remove(sessionId);
    if (pending == null) return;
    _generations[sessionId] = pending.controller;
    if (pending.onStats != null) {
      _statsCallbacks[sessionId] = pending.onStats!;
    }
    _commands!.send(GenerateCommand(pending.request));
  }
}

/// A generation waiting for the same session's previous run to finish.
typedef _PendingGeneration = ({
  GenerationRequest request,
  StreamController<String> controller,
  void Function(LlamaGenerationStats)? onStats,
});

/// Thrown when native model loading or generation fails.
class LlamaException implements Exception {
  LlamaException(this.message);
  final String message;

  @override
  String toString() => 'LlamaException: $message';
}
