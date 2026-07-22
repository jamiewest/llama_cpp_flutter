/// Pure context-budget policy: given memory and a model estimate, decide
/// whether to keep, resize, or declare the context unaffordable.
library;

import 'dart:math' as math;

import 'memory_estimator.dart';
import 'memory_monitor.dart';

/// Context sizes are planned in steps of this many tokens, so measurement
/// wobble cannot produce a stream of one-token resize proposals.
const int contextTokenGranularity = 256;

/// Tuning for the orchestrator's memory behavior.
class MemoryPolicy {
  /// Creates a policy.
  const MemoryPolicy({
    this.headroomFraction = 0.15,
    this.minContextTokens = 2048,
    this.maxContextTokens = 32768,
    this.resizeHysteresisFraction = 0.10,
    this.resizeCooldown = const Duration(seconds: 30),
    this.pollInterval = const Duration(seconds: 5),
    this.pressureAvailableFraction = 0.08,
    this.unloadOnCritical = false,
    this.maxStashBytes = 1024 * 1024 * 1024,
  });

  /// Fraction of available memory deliberately left unused.
  final double headroomFraction;

  /// Below this context size the orchestrator refuses to shrink further
  /// and reports memory-critical instead.
  final int minContextTokens;

  /// Hard ceiling on planned context size, regardless of memory.
  final int maxContextTokens;

  /// Target/current differences within this fraction are ignored —
  /// resizing recreates the context, so thrash costs more than being
  /// slightly off-budget.
  final double resizeHysteresisFraction;

  /// Minimum quiet period between resizes. Bypassed under pressure.
  final Duration resizeCooldown;

  /// How often the orchestrator samples system memory.
  final Duration pollInterval;

  /// Available memory below this fraction of total counts as pressure:
  /// shrinks apply immediately, growth is suppressed.
  final double pressureAvailableFraction;

  /// Whether a memory-critical verdict unloads the model (instead of only
  /// emitting an event and hoping the app reacts).
  final bool unloadOnCritical;

  /// Byte cap on the engine-side in-memory KV stash. When agent swaps push
  /// past it, the oldest stashes are dropped (those agents re-prefill on
  /// their next turn). The most recent stash is always kept, even alone
  /// over the cap.
  final int maxStashBytes;
}

/// Why a resize was proposed.
enum ResizeReason {
  /// Memory freed up; a bigger context fits.
  grow,

  /// Memory tightened; the current context no longer fits the budget.
  shrink,
}

/// Outcome of [planContextBudget].
sealed class PlannerDecision {
  const PlannerDecision();
}

/// The current context still fits the budget (or a change is within
/// hysteresis/cooldown); do nothing.
final class KeepContext extends PlannerDecision {
  /// Creates the decision.
  const KeepContext();
}

/// Recreate the context at [toTokens].
final class ResizeContext extends PlannerDecision {
  /// Creates the decision.
  const ResizeContext({
    required this.fromTokens,
    required this.toTokens,
    required this.reason,
  });

  /// The context size being replaced.
  final int fromTokens;

  /// The proposed context size.
  final int toTokens;

  /// Why the resize is proposed.
  final ResizeReason reason;
}

/// Even the policy's minimum context does not fit the budget.
final class MemoryCritical extends PlannerDecision {
  /// Creates the decision.
  const MemoryCritical({
    required this.availableBytes,
    required this.requiredBytes,
  });

  /// Memory available when the verdict was reached.
  final int availableBytes;

  /// Bytes the minimum acceptable context would need.
  final int requiredBytes;
}

/// Plans the context size for the next planning period.
///
/// [reclaimableBytes] is what the currently loaded model + context occupy
/// (zero when planning an initial load): it is added back to available
/// memory, since a resize releases it before reallocating.
/// [maxContextTokens] is the caller's cap — typically
/// `min(policy.maxContextTokens, spec.contextSize, trained context)`.
/// [sinceLastResize] enables the cooldown; null means no resize happened
/// yet. [underPressure] applies the pressure rules (immediate shrink, no
/// growth).
PlannerDecision planContextBudget({
  required MemorySnapshot memory,
  required ModelMemoryEstimate estimate,
  required int currentContextTokens,
  required int maxContextTokens,
  required MemoryPolicy policy,
  int reclaimableBytes = 0,
  bool underPressure = false,
  Duration? sinceLastResize,
}) {
  final budget =
      ((memory.availableBytes + reclaimableBytes) *
              (1 - policy.headroomFraction))
          .floor();
  var target = math.min(estimate.maxContextForBudget(budget), maxContextTokens);
  target = (target ~/ contextTokenGranularity) * contextTokenGranularity;

  if (target < policy.minContextTokens) {
    return MemoryCritical(
      availableBytes: memory.availableBytes,
      requiredBytes: estimate.bytesForContext(policy.minContextTokens),
    );
  }
  if (target == currentContextTokens) return const KeepContext();

  final growing = target > currentContextTokens;
  if (growing && underPressure) return const KeepContext();

  final delta = (target - currentContextTokens).abs();
  final withinHysteresis =
      delta / currentContextTokens <= policy.resizeHysteresisFraction;
  if (withinHysteresis && !underPressure) return const KeepContext();

  final coolingDown =
      sinceLastResize != null && sinceLastResize < policy.resizeCooldown;
  if (coolingDown && !underPressure) return const KeepContext();

  return ResizeContext(
    fromTokens: currentContextTokens,
    toTokens: target,
    reason: growing ? ResizeReason.grow : ResizeReason.shrink,
  );
}
