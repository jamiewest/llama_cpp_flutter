/// Memory-aware multi-agent orchestration over one loaded model: KV-cache
/// swapping between agents, GGUF-based memory estimation, dynamic context
/// sizing, and per-agent disk snapshots (see `SPEC.md` and
/// `doc/orchestration.md`).
library;

export 'src/orchestration/agent_profile.dart';
export 'src/orchestration/memory/budget_planner.dart';
export 'src/orchestration/memory/memory_estimator.dart';
export 'src/orchestration/memory/memory_monitor.dart';
export 'src/orchestration/memory/memory_monitor_factory.dart';
export 'src/orchestration/orchestrator.dart';
export 'src/orchestration/orchestrator_events.dart';
export 'src/orchestration/state/snapshot_store.dart';
export 'src/orchestration/state/snapshot_types.dart';
