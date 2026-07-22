import 'package:agents/agents.dart';
import 'package:llama_flutter/llama_flutter.dart';
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
/// The orchestrator builds a full Agents Framework [AIAgent]
/// (`ChatClientAgent`) from this profile — see `AgentHandle.agent`.
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

  /// Human-readable name for the built [AIAgent]; defaults to [id].
  final String? name;

  /// Description of the agent's purpose, surfaced on the built [AIAgent].
  final String? description;

  /// System instructions (persona) applied to every run of the built
  /// [AIAgent].
  final String? instructions;

  /// Tools the built [AIAgent] can invoke (automatic function invocation
  /// is wired by `ChatClientAgent`).
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
    required this._chatClientFactory,
    required this._invalidateSnapshot,
  });

  /// The profile this handle was registered with.
  final AgentProfile profile;

  final ChatClient Function() _chatClientFactory;
  final Future<void> Function() _invalidateSnapshot;
  ChatClient? _chatClient;
  AIAgent? _agent;

  /// The chat client this agent's [agent] runs on. Resolving a request
  /// through it activates the agent — the orchestrator makes its KV state
  /// resident first. Created lazily; requires a loaded model.
  ChatClient get chatClient => _chatClient ??= _chatClientFactory();

  /// The Agents Framework agent built from [profile]: a
  /// `ChatClientAgent` over [chatClient] carrying the profile's name,
  /// description, instructions, and tools. Created lazily; requires a
  /// loaded model.
  AIAgent get agent => _agent ??= ChatClientAgent(
    chatClient,
    options: ChatClientAgentOptions()
      ..id = profile.id
      ..name = profile.name ?? profile.id
      ..description = profile.description
      ..chatOptions = profile.instructions == null && profile.tools == null
          ? null
          : ChatOptions(
              instructions: profile.instructions,
              tools: profile.tools,
            ),
  );

  /// Drops the agent's saved KV snapshot and stash; its next turn starts
  /// from a fresh prefill.
  Future<void> invalidateSnapshot() => _invalidateSnapshot();
}
