/// Platform-selected constructor for [SystemMemoryMonitor].
library;

export 'memory_monitor_stub.dart' if (dart.library.io) 'memory_monitor_io.dart';
