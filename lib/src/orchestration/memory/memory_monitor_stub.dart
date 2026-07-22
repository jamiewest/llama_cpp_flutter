import 'memory_monitor.dart';

/// Web fallback: no memory API worth trusting, so report fixed
/// conservative values (the wasm32 heap cap is the real ceiling there).
SystemMemoryMonitor createSystemMemoryMonitor() =>
    FixedMemoryMonitor(fallbackMemorySnapshot);
