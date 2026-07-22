/// Metadata about one agent's saved KV-cache snapshot.
class AgentSnapshot {
  /// Creates a snapshot record.
  const AgentSnapshot({
    required this.agentId,
    required this.statePath,
    required this.tokens,
    required this.contextTokens,
  });

  /// The agent the snapshot belongs to.
  final String agentId;

  /// Path of the state file written by `LlamaSession.saveState`.
  final String statePath;

  /// Tokens the snapshot covers.
  final int tokens;

  /// The `n_ctx` the session had when the snapshot was taken. A snapshot
  /// can only be restored into a context at least [tokens] large.
  final int contextTokens;
}
