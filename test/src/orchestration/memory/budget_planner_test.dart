import 'package:flutter_test/flutter_test.dart';

import 'package:llama_cpp_flutter/orchestration.dart';

void main() {
  const estimate = ModelMemoryEstimate(
    weightsBytes: 0,
    kvBytesPerToken: 1000,
    fixedOverheadBytes: 0,
  );
  const policy = MemoryPolicy();

  MemorySnapshot memory(int availableBytes) =>
      MemorySnapshot(totalBytes: 8000000000, availableBytes: availableBytes);

  PlannerDecision plan({
    required int availableBytes,
    int currentContextTokens = 4096,
    int maxContextTokens = 8192,
    bool underPressure = false,
    Duration? sinceLastResize,
  }) => planContextBudget(
    memory: memory(availableBytes),
    estimate: estimate,
    currentContextTokens: currentContextTokens,
    maxContextTokens: maxContextTokens,
    policy: policy,
    underPressure: underPressure,
    sinceLastResize: sinceLastResize,
  );

  test('keeps the context when the target matches', () {
    // 5,120,000 x 0.85 = 4,352,000 -> 4352 tokens... choose an amount that
    // lands exactly on the current size instead.
    final decision = plan(availableBytes: 4819000, currentContextTokens: 4096);
    expect(decision, isA<KeepContext>());
  });

  test('proposes growth when memory frees up', () {
    final decision = plan(availableBytes: 10000000);
    expect(
      decision,
      isA<ResizeContext>()
          .having((d) => d.toTokens, 'toTokens', 8192)
          .having((d) => d.reason, 'reason', ResizeReason.grow),
    );
  });

  test('ignores changes within hysteresis', () {
    // Target 4352 vs current 4096: 6.25% < 10% hysteresis.
    final decision = plan(availableBytes: 5120000);
    expect(decision, isA<KeepContext>());
  });

  test('respects the resize cooldown', () {
    final decision = plan(
      availableBytes: 10000000,
      sinceLastResize: const Duration(seconds: 10),
    );
    expect(decision, isA<KeepContext>());
  });

  test('under pressure, shrinks immediately despite the cooldown', () {
    final decision = plan(
      availableBytes: 2900000,
      currentContextTokens: 8192,
      underPressure: true,
      sinceLastResize: const Duration(seconds: 5),
    );
    expect(
      decision,
      isA<ResizeContext>()
          .having((d) => d.reason, 'reason', ResizeReason.shrink)
          .having((d) => d.toTokens, 'toTokens', 2304),
    );
  });

  test('under pressure, suppresses growth', () {
    final decision = plan(availableBytes: 10000000, underPressure: true);
    expect(decision, isA<KeepContext>());
  });

  test('reports critical when even the minimum context cannot fit', () {
    final decision = plan(availableBytes: 1000000);
    expect(
      decision,
      isA<MemoryCritical>().having(
        (d) => d.requiredBytes,
        'requiredBytes',
        estimate.bytesForContext(policy.minContextTokens),
      ),
    );
  });

  test('counts reclaimable bytes toward the budget', () {
    // Available alone would be critical, but the loaded model's own
    // footprint comes back when resizing.
    final decision = planContextBudget(
      memory: memory(0),
      estimate: estimate,
      currentContextTokens: 8192,
      maxContextTokens: 8192,
      policy: policy,
      reclaimableBytes: estimate.bytesForContext(8192),
    );
    expect(
      decision,
      isA<ResizeContext>()
          .having((d) => d.reason, 'reason', ResizeReason.shrink)
          .having((d) => d.toTokens, 'toTokens', 6912),
    );
  });
}
