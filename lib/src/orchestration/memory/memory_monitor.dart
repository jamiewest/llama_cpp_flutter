/// System-memory observation for the orchestrator's budget planner.
library;

/// A point-in-time view of system memory.
class MemorySnapshot {
  /// Creates a snapshot.
  const MemorySnapshot({
    required this.totalBytes,
    required this.availableBytes,
    this.appFootprintBytes,
    this.isEstimated = false,
  });

  /// Physical memory installed on the device.
  final int totalBytes;

  /// Memory the app could still allocate without pressuring the system.
  ///
  /// On iOS this is the jetsam headroom (`os_proc_available_memory`); on
  /// macOS it is derived from VM statistics (free + inactive + purgeable
  /// pages). Treat it as an honest approximation, not a guarantee.
  final int availableBytes;

  /// This process's physical footprint, when the platform reports it.
  final int? appFootprintBytes;

  /// True when the values are fixed fallbacks (web, unsupported platforms)
  /// rather than measurements.
  final bool isEstimated;
}

/// Samples system memory on demand.
///
/// Implementations are platform-specific; obtain one from
/// `createSystemMemoryMonitor()` or inject a fake in tests.
abstract interface class SystemMemoryMonitor {
  /// Takes a fresh measurement.
  Future<MemorySnapshot> sample();
}

/// A monitor that always reports the same snapshot.
///
/// The web/unsupported-platform fallback, and a convenient test double.
final class FixedMemoryMonitor implements SystemMemoryMonitor {
  /// Creates a monitor pinned to [snapshot].
  FixedMemoryMonitor(this.snapshot);

  /// The snapshot every [sample] returns. Mutable so tests can steer it.
  MemorySnapshot snapshot;

  @override
  Future<MemorySnapshot> sample() async => snapshot;
}

/// Conservative defaults for platforms with no memory API: assume a small
/// device and let the policy headroom do the rest.
const MemorySnapshot fallbackMemorySnapshot = MemorySnapshot(
  totalBytes: 4 * 1024 * 1024 * 1024,
  availableBytes: 2 * 1024 * 1024 * 1024,
  isEstimated: true,
);
