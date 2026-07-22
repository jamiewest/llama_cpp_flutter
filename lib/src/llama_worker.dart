import 'dart:isolate';

import 'package:flutter/services.dart';

import 'messages.g.dart';

/// Startup payload handed to the worker isolate.
class WorkerInit {
  WorkerInit(this.token, this.sendPort);

  /// Root isolate token, required to bind platform channels in the background.
  final RootIsolateToken token;

  /// Port the worker uses to send messages back to the main isolate.
  final SendPort sendPort;
}

// --- Commands (main isolate -> worker) ---

/// Request to load a model; correlated back by [requestId].
class LoadCommand {
  LoadCommand(this.requestId, this.request);
  final int requestId;
  final ModelLoadRequest request;
}

/// Request to start generation for an already-loaded session.
class GenerateCommand {
  GenerateCommand(this.request);
  final GenerationRequest request;
}

class CancelCommand {
  CancelCommand(this.sessionId);
  final int sessionId;
}

class DisposeCommand {
  DisposeCommand(this.sessionId);
  final int sessionId;
}

/// Request to save or restore one sequence's KV-cache state; correlated
/// back by [requestId].
class StateCommand {
  StateCommand(
    this.requestId,
    this.sessionId,
    this.path, {
    required this.save,
    this.sequenceId = 0,
  });
  final int requestId;
  final int sessionId;
  final String path;
  final int sequenceId;

  /// True saves the state to [path]; false restores from it.
  final bool save;
}

/// Request to measure one sequence's state byte size; correlated back by
/// [requestId].
class StateSizeCommand {
  StateSizeCommand(this.requestId, this.sessionId, {this.sequenceId = 0});
  final int requestId;
  final int sessionId;
  final int sequenceId;
}

/// Request to stash one sequence's KV state in native memory under a key;
/// correlated back by [requestId].
class StashCommand {
  StashCommand(this.requestId, this.sessionId, this.sequenceId, this.key);
  final int requestId;
  final int sessionId;
  final int sequenceId;
  final String key;
}

/// Request to restore a stashed KV state into a sequence; correlated back
/// by [requestId].
class RestoreStashCommand {
  RestoreStashCommand(
    this.requestId,
    this.sessionId,
    this.sequenceId,
    this.key,
  );
  final int requestId;
  final int sessionId;
  final int sequenceId;
  final String key;
}

/// Request to drop a stashed KV state; correlated back by [requestId].
class DropStashCommand {
  DropStashCommand(this.requestId, this.sessionId, this.key);
  final int requestId;
  final int sessionId;
  final String key;
}

/// Request to erase one sequence's KV cache; correlated back by
/// [requestId].
class ClearSequenceCommand {
  ClearSequenceCommand(this.requestId, this.sessionId, this.sequenceId);
  final int requestId;
  final int sessionId;
  final int sequenceId;
}

/// Request to change the vision-encoder token budget; correlated back by
/// [requestId].
class SetImageBudgetCommand {
  SetImageBudgetCommand(this.requestId, this.sessionId, this.imageTokenBudget);
  final int requestId;
  final int sessionId;
  final int? imageTokenBudget;
}

/// Tears the worker down.
class ShutdownCommand {
  const ShutdownCommand();
}

// --- Replies (worker -> main isolate) ---

/// Result of a [LoadCommand]; [error] is null on success.
class LoadResult {
  LoadResult(this.requestId, this.sessionId, this.error);
  final int requestId;
  final int? sessionId;
  final String? error;
}

/// Result of a state-family command; [error] is null on success and
/// [value] carries the operation's count (tokens for save/restore, bytes
/// for size/drop, absent for clear).
class StateResult {
  StateResult(this.requestId, this.value, this.error);
  final int requestId;
  final int? value;
  final String? error;
}

/// Result of a [StashCommand]; [error] is null on success.
class StashCommandResult {
  StashCommandResult(this.requestId, this.tokens, this.bytes, this.error);
  final int requestId;
  final int? tokens;
  final int? bytes;
  final String? error;
}

/// Reports that a [GenerateCommand] failed before native generation began,
/// so no token stream (and no `done` event) will follow for this session.
class GenerateFailed {
  GenerateFailed(this.sessionId, this.error);
  final int sessionId;
  final String error;
}

/// Entry point for the worker isolate.
///
/// Binds the background binary messenger so Pigeon channels work off the root
/// isolate, then services commands and forwards the multiplexed token stream.
Future<void> llamaWorkerMain(WorkerInit init) async {
  BackgroundIsolateBinaryMessenger.ensureInitialized(init.token);
  final toMain = init.sendPort;
  final api = LlamaHostApi();

  final commands = ReceivePort();
  // Handshake: hand the command port back to the main isolate.
  toMain.send(commands.sendPort);

  // The token EventChannel is listened to on the root isolate (see
  // LlamaIsolate); a background isolate cannot register a platform message
  // handler. This isolate only issues method-channel commands.
  await for (final message in commands) {
    switch (message) {
      case LoadCommand(:final requestId, :final request):
        try {
          final sessionId = await api.loadModel(request);
          toMain.send(LoadResult(requestId, sessionId, null));
        } catch (e) {
          toMain.send(LoadResult(requestId, null, e.toString()));
        }
      case GenerateCommand(:final request):
        // An uncaught platform-channel error would kill this isolate and
        // silently hang every future command; report it instead so the main
        // isolate can error the stream.
        try {
          await api.startGeneration(request);
        } catch (e) {
          toMain.send(GenerateFailed(request.sessionId, e.toString()));
        }
      case CancelCommand(:final sessionId):
        try {
          await api.cancelGeneration(sessionId);
        } catch (_) {
          // Best-effort: the session may already be gone.
        }
      case DisposeCommand(:final sessionId):
        try {
          await api.disposeSession(sessionId);
        } catch (_) {
          // Best-effort: the session may already be gone.
        }
      case StateCommand(
        :final requestId,
        :final sessionId,
        :final path,
        :final save,
        :final sequenceId,
      ):
        try {
          final count = save
              ? await api.saveSessionState(sessionId, path, sequenceId)
              : await api.loadSessionState(sessionId, path, sequenceId);
          toMain.send(StateResult(requestId, count, null));
        } catch (e) {
          toMain.send(StateResult(requestId, null, e.toString()));
        }
      case StateSizeCommand(
        :final requestId,
        :final sessionId,
        :final sequenceId,
      ):
        try {
          final size = await api.getSessionStateSize(sessionId, sequenceId);
          toMain.send(StateResult(requestId, size, null));
        } catch (e) {
          toMain.send(StateResult(requestId, null, e.toString()));
        }
      case StashCommand(
        :final requestId,
        :final sessionId,
        :final sequenceId,
        :final key,
      ):
        try {
          final result = await api.stashSessionState(
            sessionId,
            sequenceId,
            key,
          );
          toMain.send(
            StashCommandResult(requestId, result.tokens, result.bytes, null),
          );
        } catch (e) {
          toMain.send(StashCommandResult(requestId, null, null, e.toString()));
        }
      case RestoreStashCommand(
        :final requestId,
        :final sessionId,
        :final sequenceId,
        :final key,
      ):
        try {
          final tokens = await api.restoreStashedState(
            sessionId,
            sequenceId,
            key,
          );
          toMain.send(StateResult(requestId, tokens, null));
        } catch (e) {
          toMain.send(StateResult(requestId, null, e.toString()));
        }
      case DropStashCommand(:final requestId, :final sessionId, :final key):
        try {
          final bytes = await api.dropStashedState(sessionId, key);
          toMain.send(StateResult(requestId, bytes, null));
        } catch (e) {
          toMain.send(StateResult(requestId, null, e.toString()));
        }
      case ClearSequenceCommand(
        :final requestId,
        :final sessionId,
        :final sequenceId,
      ):
        try {
          await api.clearSequence(sessionId, sequenceId);
          toMain.send(StateResult(requestId, 0, null));
        } catch (e) {
          toMain.send(StateResult(requestId, null, e.toString()));
        }
      case SetImageBudgetCommand(
        :final requestId,
        :final sessionId,
        :final imageTokenBudget,
      ):
        try {
          await api.setImageTokenBudget(sessionId, imageTokenBudget);
          toMain.send(StateResult(requestId, 0, null));
        } catch (e) {
          toMain.send(StateResult(requestId, null, e.toString()));
        }
      case ShutdownCommand():
        commands.close();
        Isolate.exit();
    }
  }
}
