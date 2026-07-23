import 'package:llama_cpp_flutter/llama_cpp_flutter.dart';
import 'package:extensions/ai.dart';

/// How reluctantly an agent's context should be sacrificed under memory
/// pressure. Advisory in v1 (residency is LRU-based); becomes an eviction
/// key when priority-aware eviction lands.
enum AgentPriority {
  /// First to lose context.
  low,

  /// Default.
  normal,

  /// Last to lose context.
  high,
}

/// Declares one purpose-specific agent to the orchestrator: identity,
/// persona ([instructions]), [tools], context residency, and sampling.
///
/// This is plain configuration data. The orchestrator uses it to size and
/// schedule the agent's KV-cache residency; building an agent abstraction
/// on top is the host app's job — see [AgentHandle.chatClient].
class AgentProfile {
  /// Creates a profile.
  const AgentProfile({
    required this.id,
    this.name,
    this.description,
    this.instructions,
    this.tools,
    this.priority = AgentPriority.normal,
    this.preferredContextTokens,
    this.sampling,
  });

  /// Stable identifier; keys the agent's registry entry, KV snapshots on
  /// disk, and stash entries.
  final String id;

  /// Human-readable name for this agent; defaults to [id].
  final String? name;

  /// Description of the agent's purpose.
  final String? description;

  /// System instructions (persona) to apply to every run of this agent.
  final String? instructions;

  /// Tools this agent may invoke.
  ///
  /// Carried as configuration only: nothing in this package executes them.
  /// Wire them up where the agent is built — e.g. by passing them in
  /// `ChatOptions.tools` and wrapping [AgentHandle.chatClient] in a
  /// tool-invoking client.
  final List<AITool>? tools;

  /// Eviction priority under memory pressure.
  final AgentPriority priority;

  /// Context-size hint. Advisory in v1: the shared context is sized by
  /// the memory budget, not per agent.
  final int? preferredContextTokens;

  /// Sampling defaults overriding the model spec's, when set.
  final SamplingDefaults? sampling;
}

/// A registered agent's connection to the orchestrator.
class AgentHandle {
  /// Creates a handle. Constructed by the orchestrator.
  AgentHandle({
    required this.profile,
    required ChatClient Function() chatClientFactory,
    required Future<void> Function() invalidateSnapshot,
  }) : _chatClientFactory = chatClientFactory,
       _invalidateSnapshot = invalidateSnapshot;

  /// The profile this handle was registered with.
  final AgentProfile profile;

  final ChatClient Function() _chatClientFactory;
  final Future<void> Function() _invalidateSnapshot;
  ChatClient? _chatClient;

  /// The chat client this agent runs on. Resolving a request through it
  /// activates the agent — the orchestrator makes its KV state resident
  /// first. Created lazily; requires a loaded model.
  ///
  /// This is the seam for an agent framework: build whatever agent
  /// abstraction you use on top of this client, carrying [profile]'s name,
  /// description, instructions, and tools.
  ChatClient get chatClient => _chatClient ??= _chatClientFactory();

  /// Drops the agent's saved KV snapshot and stash; its next turn starts
  /// from a fresh prefill.
  Future<void> invalidateSnapshot() => _invalidateSnapshot();
}
