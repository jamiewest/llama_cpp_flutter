/// Per-agent KV-cache snapshot bookkeeping.
///
/// The native implementation keeps one state file plus a JSON sidecar per
/// agent under the orchestrator's cache directory. The web implementation
/// is a no-op: wllama has no state restore, so there is nothing to store.
library;

export 'snapshot_store_stub.dart' if (dart.library.io) 'snapshot_store_io.dart';
