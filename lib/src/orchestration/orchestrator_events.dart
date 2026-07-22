import 'memory/budget_planner.dart';
import 'memory/memory_monitor.dart';

/// Telemetry the orchestrator emits so hosts can drive UI ("reloading
/// context…") and logging.
sealed class OrchestratorEvent {
  const OrchestratorEvent();
}

/// A model finished loading (or reloading after a resize).
final class ModelLoadedEvent extends OrchestratorEvent {
  /// Creates the event.
  const ModelLoadedEvent({required this.contextTokens});

  /// The context size the session was created with.
  final int contextTokens;
}

/// An agent became the session's resident context.
final class AgentActivatedEvent extends OrchestratorEvent {
  /// Creates the event.
  const AgentActivatedEvent({
    required this.agentId,
    required this.restoredTokens,
    this.wasResident = false,
  });

  /// The activated agent.
  final String agentId;

  /// Tokens restored from its snapshot or stash; `0` means the next
  /// generation re-prefills from scratch (unless [wasResident]).
  final int restoredTokens;

  /// True when the agent's KV state was already live in its own sequence
  /// (multi-sequence mode) — nothing was swapped or re-prefilled.
  final bool wasResident;
}

/// An agent's KV state was copied into the engine's in-memory stash (the
/// fast swap path — no disk involved).
final class StateStashedEvent extends OrchestratorEvent {
  /// Creates the event.
  const StateStashedEvent({
    required this.agentId,
    required this.tokens,
    required this.bytes,
  });

  /// The agent whose state was stashed.
  final String agentId;

  /// Tokens the stash covers.
  final int tokens;

  /// Stash size in bytes (held in process memory until restored, evicted,
  /// or the model is reloaded).
  final int bytes;
}

/// A stashed KV state was dropped to enforce the stash byte cap; the agent
/// re-prefills on its next turn.
final class StashEvictedEvent extends OrchestratorEvent {
  /// Creates the event.
  const StashEvictedEvent({required this.agentId, required this.bytes});

  /// The agent whose stash was dropped.
  final String agentId;

  /// Bytes freed.
  final int bytes;
}

/// An agent's KV state was snapshotted before switching away.
final class SnapshotSavedEvent extends OrchestratorEvent {
  /// Creates the event.
  const SnapshotSavedEvent({required this.agentId, required this.tokens});

  /// The agent whose state was saved.
  final String agentId;

  /// Tokens the snapshot covers.
  final int tokens;
}

/// An agent was added to the registry.
final class AgentRegisteredEvent extends OrchestratorEvent {
  /// Creates the event.
  const AgentRegisteredEvent({required this.agentId});

  /// The registered agent.
  final String agentId;
}

/// An agent was removed from the registry; its KV state (live sequence,
/// stash, disk snapshot) was cleaned up.
final class AgentUnregisteredEvent extends OrchestratorEvent {
  /// Creates the event.
  const AgentUnregisteredEvent({required this.agentId});

  /// The removed agent.
  final String agentId;
}

/// A snapshot could not be restored (or no longer fits) and was dropped.
final class SnapshotInvalidatedEvent extends OrchestratorEvent {
  /// Creates the event.
  const SnapshotInvalidatedEvent({required this.agentId});

  /// The agent whose snapshot was dropped.
  final String agentId;
}

/// The context was recreated at a new size.
final class ContextResizedEvent extends OrchestratorEvent {
  /// Creates the event.
  const ContextResizedEvent({
    required this.fromTokens,
    required this.toTokens,
    required this.reason,
  });

  /// The replaced context size.
  final int fromTokens;

  /// The new context size.
  final int toTokens;

  /// Why the resize happened.
  final ResizeReason reason;
}

/// Even the minimum acceptable context no longer fits in memory.
final class MemoryCriticalEvent extends OrchestratorEvent {
  /// Creates the event.
  const MemoryCriticalEvent({
    required this.snapshot,
    required this.requiredBytes,
  });

  /// The memory measurement that triggered the verdict.
  final MemorySnapshot snapshot;

  /// Bytes the minimum context would need.
  final int requiredBytes;
}
