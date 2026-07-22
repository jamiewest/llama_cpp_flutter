import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:llama_cpp_flutter/src/orchestration/memory/memory_monitor_io.dart';

void main() {
  test('DarwinMemoryMonitor measures real values on macOS', () async {
    if (!Platform.isMacOS) {
      markTestSkipped('Darwin-only smoke test.');
      return;
    }
    final monitor = createSystemMemoryMonitor();
    final snapshot = await monitor.sample();

    // Real hardware: total is at least 1 GiB, available is positive and
    // no larger than total, and the process footprint is plausible.
    expect(snapshot.isEstimated, isFalse);
    expect(snapshot.totalBytes, greaterThan(1024 * 1024 * 1024));
    expect(snapshot.availableBytes, greaterThan(0));
    expect(snapshot.availableBytes, lessThanOrEqualTo(snapshot.totalBytes));
    expect(snapshot.appFootprintBytes, isNotNull);
    expect(snapshot.appFootprintBytes, greaterThan(1024 * 1024));
  });
}
