import 'dart:convert';
import 'dart:io';

import 'snapshot_types.dart';

/// Tracks per-agent KV snapshots under
/// `<root>/orchestrator/<modelKey>/<agentId>.llstate` with a JSON sidecar
/// recording the token count and the context size at save time.
class SnapshotStore {
  /// Creates a store rooted at [rootDirectory].
  SnapshotStore(this.rootDirectory);

  /// Directory all snapshot files live under.
  final String rootDirectory;

  /// The state-file path `LlamaSession.saveState` should write for
  /// [agentId] under [modelKey]. Creates the parent directory.
  String statePathFor(String modelKey, String agentId) {
    final dir = Directory(_modelDir(modelKey))..createSync(recursive: true);
    return '${dir.path}/${_sanitize(agentId)}.llstate';
  }

  /// The snapshot recorded for [agentId], or null when there is none or
  /// it no longer fits [maxTokens] (a shrunken context).
  Future<AgentSnapshot?> find(
    String modelKey,
    String agentId, {
    int? maxTokens,
  }) async {
    final sidecar = File(_sidecarPath(modelKey, agentId));
    if (!await sidecar.exists()) return null;
    try {
      final data = jsonDecode(await sidecar.readAsString());
      if (data is! Map<String, Object?>) return null;
      final snapshot = AgentSnapshot(
        agentId: agentId,
        statePath: statePathFor(modelKey, agentId),
        tokens: data['tokens'] as int,
        contextTokens: data['contextTokens'] as int,
      );
      if (!await File(snapshot.statePath).exists()) return null;
      if (maxTokens != null && snapshot.tokens > maxTokens) return null;
      return snapshot;
    } on Object {
      return null;
    }
  }

  /// Records that a snapshot of [tokens] tokens was saved for [agentId]
  /// while the context was [contextTokens] large.
  Future<void> record(
    String modelKey,
    String agentId, {
    required int tokens,
    required int contextTokens,
  }) async {
    final sidecar = File(_sidecarPath(modelKey, agentId));
    await sidecar.parent.create(recursive: true);
    await sidecar.writeAsString(
      jsonEncode({'tokens': tokens, 'contextTokens': contextTokens}),
    );
  }

  /// Drops [agentId]'s snapshot (sidecar and state file), if present.
  Future<void> invalidate(String modelKey, String agentId) async {
    for (final path in [
      _sidecarPath(modelKey, agentId),
      statePathFor(modelKey, agentId),
    ]) {
      final file = File(path);
      if (await file.exists()) await file.delete();
    }
  }

  /// Drops every snapshot under [modelKey] matching [test] — e.g. all
  /// snapshots too large for a shrunken context.
  Future<void> invalidateWhere(
    String modelKey,
    bool Function(AgentSnapshot snapshot) test,
  ) async {
    final dir = Directory(_modelDir(modelKey));
    if (!await dir.exists()) return;
    await for (final entry in dir.list()) {
      if (entry is! File || !entry.path.endsWith('.json')) continue;
      final agentId = entry.uri.pathSegments.last.replaceAll('.json', '');
      final snapshot = await find(modelKey, agentId);
      if (snapshot != null && test(snapshot)) {
        await invalidate(modelKey, agentId);
      }
    }
  }

  String _modelDir(String modelKey) =>
      '$rootDirectory/orchestrator/${_sanitize(modelKey)}';

  String _sidecarPath(String modelKey, String agentId) =>
      '${_modelDir(modelKey)}/${_sanitize(agentId)}.json';

  static String _sanitize(String value) =>
      value.replaceAll(RegExp(r'[^A-Za-z0-9._-]'), '_');
}
