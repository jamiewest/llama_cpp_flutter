import 'snapshot_types.dart';

/// Web no-op store: wllama cannot restore KV state, so agent switches
/// re-prefill and there is nothing to persist.
class SnapshotStore {
  /// Creates the store; [rootDirectory] is kept only for API symmetry.
  SnapshotStore(this.rootDirectory);

  /// Unused on the web.
  final String rootDirectory;

  /// Returns a placeholder path; never written because web sessions report
  /// `canPersistState` false.
  String statePathFor(String modelKey, String agentId) =>
      '$rootDirectory/orchestrator/$modelKey/$agentId.llstate';

  /// Always null: nothing is ever stored.
  Future<AgentSnapshot?> find(
    String modelKey,
    String agentId, {
    int? maxTokens,
  }) async => null;

  /// No-op.
  Future<void> record(
    String modelKey,
    String agentId, {
    required int tokens,
    required int contextTokens,
  }) async {}

  /// No-op.
  Future<void> invalidate(String modelKey, String agentId) async {}

  /// No-op.
  Future<void> invalidateWhere(
    String modelKey,
    bool Function(AgentSnapshot snapshot) test,
  ) async {}
}
