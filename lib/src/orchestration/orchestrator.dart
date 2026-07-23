import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:llama_cpp_flutter/llama_cpp_flutter.dart';
import 'package:extensions/logging.dart';

import 'agent_profile.dart';
import 'memory/budget_planner.dart';
import 'memory/memory_monitor.dart';
import 'memory/memory_monitor_factory.dart';
import 'memory/memory_estimator.dart';
import 'orchestrator_events.dart';
import 'staging/artifact_stager.dart';
import 'staging/staged_artifacts.dart';
import 'state/snapshot_store.dart';

/// Owns one loaded model and shares its session between registered agents,
/// sizing the context to the device's memory budget.
///
/// Agents keep their contexts three ways, best-first:
///
/// 1. **Multi-sequence** ([sequenceSlots] > 1, native): each resident agent
///    owns a llama.cpp KV sequence inside one context — switching agents
///    costs nothing. With more agents than slots, the least-recently-used
///    resident is stashed (in-memory) and its slot reassigned.
/// 2. **In-memory stash** (single slot, native): switching agents copies
///    the outgoing agent's KV state into an engine-side RAM stash and
///    restores the incoming agent's — no disk in the swap path.
/// 3. **Re-prefill** (web, or nothing saved): the next generation rebuilds
///    the context from the rendered transcript.
///
/// Disk snapshots are written only at persistence points (unload, resize,
/// dispose) for the active agent, and restored when nothing faster exists.
///
/// The orchestrator is the owner behind `LlamaChatClient`'s
/// `SessionProvider` seam: each agent's chat client resolves its session
/// through [sessionFor], which activates the agent before returning a
/// session bound to the agent's sequence.
class LlamaOrchestrator {
  /// Creates an orchestrator.
  ///
  /// [cacheDirectory] holds downloaded model artifacts and per-agent disk
  /// snapshots. [sequenceSlots] asks for that many independent KV
  /// sequences at load (engines without sequence support fall back to 1;
  /// note it disables speculative decoding when > 1). [memoryMonitor]
  /// defaults to the platform monitor; [downloader] overrides how
  /// artifacts download (auth, fakes). [loggerFactory] supplies loggers for
  /// the orchestrator and the chat clients it builds; null logs nothing.
  LlamaOrchestrator({
    required LlamaRuntime runtime,
    required String cacheDirectory,
    this.policy = const MemoryPolicy(),
    int sequenceSlots = 1,
    SystemMemoryMonitor? memoryMonitor,
    ModelDownloader? downloader,
    LoggerFactory? loggerFactory,
  }) : _runtime = runtime,
       _downloader = downloader,
       _cacheDirectory = cacheDirectory,
       _defaultSlots = math.max(sequenceSlots, 1),
       _requestedSlots = math.max(sequenceSlots, 1),
       _monitor = memoryMonitor ?? createSystemMemoryMonitor(),
       _snapshots = SnapshotStore(cacheDirectory),
       _loggerFactory = loggerFactory,
       _logger = (loggerFactory ?? NullLoggerFactory.instance).createLogger(
         'LlamaOrchestrator',
       );

  /// Memory-behavior tuning.
  final MemoryPolicy policy;

  final LlamaRuntime _runtime;
  final String _cacheDirectory;
  final LoggerFactory? _loggerFactory;
  final Logger _logger;
  final int _defaultSlots;
  int _requestedSlots;
  final SystemMemoryMonitor _monitor;
  final ModelDownloader? _downloader;
  final SnapshotStore _snapshots;
  final Map<String, AgentHandle> _agents = <String, AgentHandle>{};
  final _TaskQueue _tasks = _TaskQueue();
  // Synchronous delivery so an event is observable as soon as the
  // operation that emitted it completes; listeners must not call back
  // into the orchestrator synchronously (queue a microtask instead).
  final StreamController<OrchestratorEvent> _events =
      StreamController<OrchestratorEvent>.broadcast(sync: true);

  ModelSpec? _spec;
  StagedArtifacts? _staged;
  ModelMemoryEstimate? _estimate;
  LlamaSession? _session;
  int _contextTokens = 0;
  int _slots = 1;
  String? _activeAgentId;
  DateTime? _lastResizeAt;
  int? _pendingGrowTokens;
  Timer? _pollTimer;
  bool _disposed = false;

  /// Resident agents in multi-sequence mode: agent id -> sequence id.
  final Map<String, int> _agentSequence = <String, int>{};
  final List<int> _freeSequences = <int>[];
  final Map<String, int> _lastUsed = <String, int>{};
  int _useTick = 0;

  /// Stash bookkeeping (the blobs themselves live engine-side), in
  /// insertion order — oldest first for cap eviction.
  final Map<String, LlamaStashResult> _stashed = <String, LlamaStashResult>{};
  int _stashBytes = 0;

  /// Lifecycle and memory telemetry.
  Stream<OrchestratorEvent> get events => _events.stream;

  /// The TOTAL context size the current session was created with; `0` when
  /// no model is loaded. Split across [sequenceSlots] in multi-sequence
  /// mode — see [contextTokensPerAgent].
  int get contextTokens => _contextTokens;

  /// The context window each agent effectively gets.
  int get contextTokensPerAgent =>
      _slots <= 1 ? _contextTokens : _contextTokens ~/ _slots;

  /// KV sequences the loaded session actually supports (1 until a model
  /// is loaded, or on engines without sequence support).
  int get sequenceSlots => _slots;

  /// Bytes currently held by the engine-side KV stash.
  int get stashedBytes => _stashBytes;

  /// The most recently activated agent, or null.
  String? get activeAgentId => _activeAgentId;

  /// Whether a model is loaded.
  bool get isModelLoaded => _session != null;

  /// The memory estimate for the loaded model, when one could be read.
  ModelMemoryEstimate? get memoryEstimate => _estimate;

  /// Loads [spec], choosing a context size that fits the current memory
  /// budget (clamped to `spec.contextSize`, the policy ceiling, and the
  /// model's trained context — all scaled by [sequenceSlots]).
  ///
  /// On native, artifacts are downloaded into the cache directory unless
  /// [localPath] (and friends) point at existing files; the GGUF header
  /// is then read to build the memory estimate. [estimateOverride] skips
  /// header-based estimation (tests, pre-computed estimates). Replaces any
  /// previously loaded model.
  ///
  /// [sequenceSlots] overrides the constructor default for this load —
  /// e.g. `1` to keep speculative decoding (a draft model) usable, since
  /// multi-sequence sessions drop the drafter.
  Future<void> loadModel(
    ModelSpec spec, {
    String? localPath,
    String? localMmprojPath,
    String? localDraftPath,
    LlamaLoadProgress? onProgress,
    ModelMemoryEstimate? estimateOverride,
    int? sequenceSlots,
  }) => _tasks.run(() async {
    _checkNotDisposed();
    await _unloadLocked();
    _requestedSlots = math.max(sequenceSlots ?? _defaultSlots, 1);
    final staged = await stageModelArtifacts(
      spec,
      cacheDirectory: _cacheDirectory,
      localPath: localPath,
      localMmprojPath: localMmprojPath,
      localDraftPath: localDraftPath,
      downloader: _downloader,
      onProgress: onProgress,
    );
    _spec = spec;
    _staged = staged;
    _estimate = estimateOverride ?? staged.estimate;
    final initial = await _initialContextTokens(spec);
    final session = await _loadSession(spec, initial);
    _session = session;
    _contextTokens = initial;
    _resetResidency(session.capabilities.maxSequences);
    _logger.logInformation(
      'Model "${spec.id}" loaded: $initial context tokens across '
      '$_slots sequence slot${_slots == 1 ? '' : 's'}.',
    );
    _emit(ModelLoadedEvent(contextTokens: initial));
    _startPolling();
  });

  /// The registered agents, in registration order.
  List<AgentHandle> get agents => List.unmodifiable(_agents.values);

  /// The registered agent ids, in registration order.
  List<String> get agentIds => List.unmodifiable(_agents.keys);

  /// The handle registered under [id], or null.
  AgentHandle? agent(String id) => _agents[id];

  /// Removes the agent registered under [id] and cleans up its state: its
  /// live sequence (multi-sequence mode), its RAM stash entry, and its
  /// disk snapshot. Returns false when no such agent was registered.
  ///
  /// The handle (and any [AgentHandle.agent] built from it) stops working:
  /// its next request would re-register nothing and throws through
  /// [sessionFor].
  Future<bool> unregisterAgent(String id) => _tasks.run(() async {
    final handle = _agents.remove(id);
    if (handle == null) return false;
    _lastUsed.remove(id);
    final session = _session;
    if (session != null) {
      final seq = _agentSequence.remove(id);
      if (seq != null) {
        await session.clearSequence(seq);
        _freeSequences.add(seq);
      }
      await _dropStash(session, id);
    } else {
      _stashBytes -= _stashed.remove(id)?.bytes ?? 0;
    }
    final spec = _spec;
    if (spec != null) await _snapshots.invalidate(spec.id, id);
    if (_activeAgentId == id) _activeAgentId = null;
    _logger.logDebug('Agent "$id" unregistered.');
    _emit(AgentUnregisteredEvent(agentId: id));
    return true;
  });

  /// Registers [profile] and returns its handle. Registering an already
  /// known id returns the existing handle.
  ///
  /// New registrations emit an [AgentRegisteredEvent].
  AgentHandle registerAgent(AgentProfile profile) {
    final existing = _agents[profile.id];
    if (existing != null) return existing;
    final handle = _createHandle(profile);
    _agents[profile.id] = handle;
    _logger.logDebug('Agent "${profile.id}" registered.');
    _emit(AgentRegisteredEvent(agentId: profile.id));
    return handle;
  }

  AgentHandle _createHandle(AgentProfile profile) => AgentHandle(
    profile: profile,
    chatClientFactory: () {
      final spec = _spec;
      if (spec == null) {
        throw StateError('Load a model before creating agent chat clients.');
      }
      return createLlamaChatClient(
        spec: spec,
        sessionProvider: () => sessionFor(profile.id),
        contextSizeOverride: contextTokensPerAgent,
        samplingOverride: profile.sampling,
        loggerFactory: _loggerFactory,
      );
    },
    invalidateSnapshot: () async {
      final spec = _spec;
      if (spec == null) return;
      await _snapshots.invalidate(spec.id, profile.id);
      final session = _session;
      if (session != null && _stashed.containsKey(profile.id)) {
        await _dropStash(session, profile.id);
      }
      _emit(SnapshotInvalidatedEvent(agentId: profile.id));
    },
  );

  /// Activates [agentId] and returns a session to generate with (bound to
  /// the agent's sequence in multi-sequence mode).
  ///
  /// Activation is serialized: concurrent agent requests queue rather than
  /// interleave. This is the `SessionProvider` behind every registered
  /// agent's chat client; call it directly only when driving sessions
  /// without the Agents Framework.
  Future<LlamaSession> sessionFor(String agentId) =>
      _tasks.run(() => _activateLocked(agentId));

  /// Re-plans the context size against a fresh memory sample now, instead
  /// of waiting for the next poll tick.
  Future<void> checkMemory() => _tasks.run(_checkMemoryLocked);

  /// Unloads the model (saving the active agent's snapshot to disk first,
  /// where supported). Registered agents survive and reactivate after the
  /// next [loadModel].
  Future<void> unloadModel() => _tasks.run(_unloadLocked);

  /// Unloads and closes the event stream. The instance is unusable
  /// afterwards.
  Future<void> dispose() async {
    if (_disposed) return;
    _disposed = true;
    await _tasks.run(_unloadLocked);
    await _events.close();
  }

  // --- Activation ---

  Future<LlamaSession> _activateLocked(String agentId) async {
    if (!_agents.containsKey(agentId)) {
      throw StateError('No agent is registered under "$agentId".');
    }
    var session = _session;
    if (session == null) {
      throw StateError('No model is loaded; call loadModel first.');
    }
    final pendingGrow = _pendingGrowTokens;
    _pendingGrowTokens = null;
    if (pendingGrow != null && pendingGrow != _contextTokens) {
      session = await _resizeLocked(pendingGrow, ResizeReason.grow);
    }
    _lastUsed[agentId] = ++_useTick;
    final spec = _spec!;
    if (_slots > 1) return _activateMultiSequence(session, spec, agentId);

    if (_activeAgentId == agentId) return session;
    final capabilities = session.capabilities;
    var restored = 0;
    if (capabilities.canStashState || capabilities.canPersistState) {
      final previous = _activeAgentId;
      if (previous != null) {
        await _preserveAgent(session, spec, previous, 0);
      }
      restored = await _restoreAgent(session, spec, agentId, 0);
      // No state to restore needs no explicit clear: the engine's prefix
      // reuse trims to the common prefix on the next prompt anyway.
    }
    _activeAgentId = agentId;
    _logger.logDebug(
      'Agent "$agentId" activated ($restored tokens restored).',
    );
    _emit(AgentActivatedEvent(agentId: agentId, restoredTokens: restored));
    return session;
  }

  Future<LlamaSession> _activateMultiSequence(
    LlamaSession session,
    ModelSpec spec,
    String agentId,
  ) async {
    final existing = _agentSequence[agentId];
    if (existing != null) {
      if (_activeAgentId != agentId) {
        _activeAgentId = agentId;
        _emit(
          AgentActivatedEvent(
            agentId: agentId,
            restoredTokens: 0,
            wasResident: true,
          ),
        );
      }
      return _SequenceBoundSession(session, existing);
    }
    final seq = await _acquireSequence(session, spec);
    _agentSequence[agentId] = seq;
    final restored = await _restoreAgent(session, spec, agentId, seq);
    _activeAgentId = agentId;
    _logger.logDebug(
      'Agent "$agentId" activated on sequence $seq '
      '($restored tokens restored).',
    );
    _emit(AgentActivatedEvent(agentId: agentId, restoredTokens: restored));
    return _SequenceBoundSession(session, seq);
  }

  /// Returns a free sequence, evicting the least-recently-used resident
  /// agent (stash + clear) when all slots are taken.
  Future<int> _acquireSequence(LlamaSession session, ModelSpec spec) async {
    if (_freeSequences.isNotEmpty) return _freeSequences.removeLast();
    final victim = _agentSequence.keys.reduce(
      (a, b) => (_lastUsed[a] ?? 0) < (_lastUsed[b] ?? 0) ? a : b,
    );
    final seq = _agentSequence.remove(victim)!;
    _logger.logDebug(
      'Evicting least-recently-used agent "$victim" from sequence $seq.',
    );
    await _preserveAgent(session, spec, victim, seq);
    await session.clearSequence(seq);
    return seq;
  }

  /// Saves [agentId]'s state out of [sequenceId] — stash when supported,
  /// disk otherwise. Failures degrade to "nothing preserved".
  Future<void> _preserveAgent(
    LlamaSession session,
    ModelSpec spec,
    String agentId,
    int sequenceId,
  ) async {
    if (session.capabilities.canStashState) {
      try {
        final result = await session.stashState(
          agentId,
          sequenceId: sequenceId,
        );
        if (result.tokens > 0) {
          _stashBytes -= _stashed.remove(agentId)?.bytes ?? 0;
          _stashed[agentId] = result;
          _stashBytes += result.bytes;
          _emit(
            StateStashedEvent(
              agentId: agentId,
              tokens: result.tokens,
              bytes: result.bytes,
            ),
          );
          await _enforceStashCap(session);
        } else {
          await _dropStash(session, agentId);
        }
        return;
      } on Object catch (error) {
        _logger.logWarning(
          'Stashing agent "$agentId" state failed; falling back to a disk '
          'snapshot.',
          error: error,
        );
        await _dropStash(session, agentId);
      }
    }
    if (session.capabilities.canPersistState) {
      await _saveDiskSnapshot(session, spec.id, agentId, sequenceId);
    }
  }

  /// Restores [agentId]'s state into [sequenceId] — stash first, disk
  /// snapshot second. Returns tokens restored (`0` = re-prefill).
  Future<int> _restoreAgent(
    LlamaSession session,
    ModelSpec spec,
    String agentId,
    int sequenceId,
  ) async {
    if (_stashed.containsKey(agentId)) {
      try {
        final tokens = await session.restoreStashedState(
          agentId,
          sequenceId: sequenceId,
        );
        await _dropStash(session, agentId);
        return tokens;
      } on Object catch (error) {
        _logger.logWarning(
          'Restoring agent "$agentId" from the stash failed; trying its '
          'disk snapshot.',
          error: error,
        );
        await _dropStash(session, agentId);
      }
    }
    if (session.capabilities.canPersistState) {
      final snapshot = await _snapshots.find(
        spec.id,
        agentId,
        maxTokens: contextTokensPerAgent,
      );
      if (snapshot != null) {
        try {
          return await session.loadState(
            snapshot.statePath,
            sequenceId: sequenceId,
          );
        } on Object catch (error) {
          _logger.logWarning(
            'Loading agent "$agentId" disk snapshot failed; it will '
            're-prefill.',
            error: error,
          );
          await _snapshots.invalidate(spec.id, agentId);
          _emit(SnapshotInvalidatedEvent(agentId: agentId));
        }
      }
    }
    return 0;
  }

  Future<void> _dropStash(LlamaSession session, String agentId) async {
    final entry = _stashed.remove(agentId);
    if (entry == null) return;
    _stashBytes -= entry.bytes;
    try {
      await session.dropStashedState(agentId);
    } on Object {
      // Best-effort; the stash dies with the session regardless.
    }
  }

  /// Evicts oldest stashes until under the policy cap, always keeping the
  /// most recent entry.
  Future<void> _enforceStashCap(LlamaSession session) async {
    while (_stashBytes > policy.maxStashBytes && _stashed.length > 1) {
      final oldest = _stashed.keys.first;
      final bytes = _stashed[oldest]!.bytes;
      await _dropStash(session, oldest);
      _logger.logDebug(
        'Stash cap exceeded: evicted agent "$oldest" ($bytes bytes).',
      );
      _emit(StashEvictedEvent(agentId: oldest, bytes: bytes));
    }
  }

  Future<void> _saveDiskSnapshot(
    LlamaSession session,
    String modelKey,
    String agentId,
    int sequenceId,
  ) async {
    final path = _snapshots.statePathFor(modelKey, agentId);
    try {
      final tokens = await session.saveState(path, sequenceId: sequenceId);
      if (tokens > 0) {
        await _snapshots.record(
          modelKey,
          agentId,
          tokens: tokens,
          contextTokens: contextTokensPerAgent,
        );
        _emit(SnapshotSavedEvent(agentId: agentId, tokens: tokens));
      } else {
        // Nothing cached: any older snapshot is stale relative to the
        // agent's (now empty) live state; drop it.
        await _snapshots.invalidate(modelKey, agentId);
      }
    } on Object catch (error) {
      _logger.logWarning(
        'Saving agent "$agentId" disk snapshot failed; its previous '
        'snapshot was invalidated.',
        error: error,
      );
      await _snapshots.invalidate(modelKey, agentId);
    }
  }

  // --- Memory management ---

  Future<void> _checkMemoryLocked() async {
    final session = _session;
    final estimate = _estimate;
    final spec = _spec;
    if (session == null || estimate == null || spec == null) return;
    final memory = await _monitor.sample();
    final underPressure =
        memory.availableBytes <
        memory.totalBytes * policy.pressureAvailableFraction;
    final decision = planContextBudget(
      memory: memory,
      estimate: estimate,
      currentContextTokens: _contextTokens,
      maxContextTokens: _maxContextTokens(spec, estimate),
      policy: _plannerPolicy,
      reclaimableBytes: estimate.bytesForContext(_contextTokens),
      underPressure: underPressure,
      sinceLastResize: _lastResizeAt == null
          ? null
          : DateTime.now().difference(_lastResizeAt!),
    );
    switch (decision) {
      case KeepContext():
        break;
      case ResizeContext(:final toTokens, :final reason):
        if (reason == ResizeReason.shrink) {
          await _resizeLocked(toTokens, reason);
        } else {
          // Growth is lazy: applied on the next agent activation, when
          // swapping is happening anyway.
          _pendingGrowTokens = toTokens;
        }
      case MemoryCritical(:final requiredBytes):
        _logger.logWarning(
          'Memory critical: $requiredBytes bytes required, '
          '${memory.availableBytes} available'
          '${policy.unloadOnCritical ? '; unloading the model' : ''}.',
        );
        _emit(
          MemoryCriticalEvent(snapshot: memory, requiredBytes: requiredBytes),
        );
        if (policy.unloadOnCritical) await _unloadLocked();
    }
  }

  Future<LlamaSession> _resizeLocked(int toTokens, ResizeReason reason) async {
    final session = _session!;
    final spec = _spec!;
    final from = _contextTokens;
    await session.cancel();
    // The stash dies with the session, so the active agent's state goes to
    // disk; other residents/stashes re-prefill on their next turn.
    final active = _activeAgentId;
    if (active != null && session.capabilities.canPersistState) {
      await _saveDiskSnapshot(
        session,
        spec.id,
        active,
        _agentSequence[active] ?? 0,
      );
    }
    _session = null;
    await session.dispose();
    final next = await _loadSession(spec, toTokens);
    _session = next;
    _contextTokens = toTokens;
    _lastResizeAt = DateTime.now();
    _resetResidency(next.capabilities.maxSequences);
    await _snapshots.invalidateWhere(
      spec.id,
      (snapshot) => snapshot.tokens > contextTokensPerAgent,
    );
    if (active != null && next.capabilities.canPersistState) {
      final seq = _slots > 1 ? await _acquireSequence(next, spec) : 0;
      if (_slots > 1) _agentSequence[active] = seq;
      final snapshot = await _snapshots.find(
        spec.id,
        active,
        maxTokens: contextTokensPerAgent,
      );
      if (snapshot != null) {
        try {
          await next.loadState(snapshot.statePath, sequenceId: seq);
        } on Object {
          await _snapshots.invalidate(spec.id, active);
          _emit(SnapshotInvalidatedEvent(agentId: active));
        }
      }
    }
    _logger.logInformation(
      'Context resized from $from to $toTokens tokens (${reason.name}).',
    );
    _emit(
      ContextResizedEvent(fromTokens: from, toTokens: toTokens, reason: reason),
    );
    return next;
  }

  Future<void> _unloadLocked() async {
    _pollTimer?.cancel();
    _pollTimer = null;
    final session = _session;
    final spec = _spec;
    _session = null;
    if (session != null) {
      await session.cancel();
      final active = _activeAgentId;
      if (active != null &&
          spec != null &&
          session.capabilities.canPersistState) {
        await _saveDiskSnapshot(
          session,
          spec.id,
          active,
          _agentSequence[active] ?? 0,
        );
      }
      await session.dispose();
      _logger.logInformation('Model unloaded.');
    }
    _activeAgentId = null;
    _contextTokens = 0;
    _pendingGrowTokens = null;
    _resetResidency(1);
  }

  /// Resets all residency/stash bookkeeping for a (re)loaded session
  /// supporting [slots] sequences. The previous session's stash blobs died
  /// with it.
  void _resetResidency(int slots) {
    _slots = math.max(slots, 1);
    _agentSequence.clear();
    _freeSequences
      ..clear()
      ..addAll(List<int>.generate(_slots, (index) => _slots - 1 - index));
    _stashed.clear();
    _stashBytes = 0;
  }

  Future<int> _initialContextTokens(ModelSpec spec) async {
    final estimate = _estimate;
    final cap = _maxContextTokens(spec, estimate);
    if (estimate == null) return cap;
    final planner = _plannerPolicy;
    final memory = await _monitor.sample();
    final budget = (memory.availableBytes * (1 - planner.headroomFraction))
        .floor();
    var target = math.min(estimate.maxContextForBudget(budget), cap);
    target = (target ~/ contextTokenGranularity) * contextTokenGranularity;
    if (target < planner.minContextTokens) {
      // Load anyway at the smallest acceptable size and let the host react
      // to the event; refusing to load entirely helps nobody.
      _emit(
        MemoryCriticalEvent(
          snapshot: memory,
          requiredBytes: estimate.bytesForContext(planner.minContextTokens),
        ),
      );
      return math.min(planner.minContextTokens, cap);
    }
    return target;
  }

  /// The planner works in TOTAL context tokens; per-agent policy bounds
  /// scale by the requested slot count.
  MemoryPolicy get _plannerPolicy => _requestedSlots <= 1
      ? policy
      : MemoryPolicy(
          headroomFraction: policy.headroomFraction,
          minContextTokens: policy.minContextTokens * _requestedSlots,
          maxContextTokens: policy.maxContextTokens * _requestedSlots,
          resizeHysteresisFraction: policy.resizeHysteresisFraction,
          resizeCooldown: policy.resizeCooldown,
          pollInterval: policy.pollInterval,
          pressureAvailableFraction: policy.pressureAvailableFraction,
          unloadOnCritical: policy.unloadOnCritical,
          maxStashBytes: policy.maxStashBytes,
        );

  int _maxContextTokens(ModelSpec spec, ModelMemoryEstimate? estimate) {
    var cap = math.min(spec.contextSize, _plannerPolicy.maxContextTokens);
    final trained = estimate?.trainedContextTokens;
    if (trained != null && trained > 0) {
      cap = math.min(cap, trained * _requestedSlots);
    }
    return cap;
  }

  Future<LlamaSession> _loadSession(ModelSpec spec, int contextTokens) {
    final staged = _staged!;
    return _runtime.loadModel(
      spec.copyWith(contextSize: contextTokens, maxSequences: _requestedSlots),
      localPath: staged.modelPath,
      localMmprojPath: staged.mmprojPath,
      localDraftPath: staged.draftPath,
    );
  }

  void _startPolling() {
    _pollTimer?.cancel();
    _pollTimer = Timer.periodic(policy.pollInterval, (_) {
      unawaited(checkMemory());
    });
  }

  void _emit(OrchestratorEvent event) {
    if (!_events.isClosed) _events.add(event);
  }

  void _checkNotDisposed() {
    if (_disposed) throw StateError('The orchestrator has been disposed.');
  }
}

/// A view of the shared session locked to one KV sequence: every
/// generation and state operation targets that sequence, regardless of the
/// `sequenceId` the caller passes. [dispose] is a no-op — the orchestrator
/// owns the underlying session.
final class _SequenceBoundSession implements LlamaSession {
  _SequenceBoundSession(this._inner, this._sequenceId);

  final LlamaSession _inner;
  final int _sequenceId;

  @override
  Stream<String> generate(
    String prompt, {
    int maxTokens = 256,
    double temperature = 0.8,
    int? topK,
    double? topP,
    int? seed,
    List<String> stopSequences = const <String>[],
    List<Uint8List>? media,
    List<LlamaChatTurn>? turns,
    int sequenceId = 0,
    LlamaStatsCallback? onStats,
  }) => _inner.generate(
    prompt,
    maxTokens: maxTokens,
    temperature: temperature,
    topK: topK,
    topP: topP,
    seed: seed,
    stopSequences: stopSequences,
    media: media,
    turns: turns,
    sequenceId: _sequenceId,
    onStats: onStats,
  );

  @override
  Future<void> cancel() => _inner.cancel();

  @override
  LlamaSessionCapabilities get capabilities => _inner.capabilities;

  @override
  Future<int> saveState(String path, {int sequenceId = 0}) =>
      _inner.saveState(path, sequenceId: _sequenceId);

  @override
  Future<int> loadState(String path, {int sequenceId = 0}) =>
      _inner.loadState(path, sequenceId: _sequenceId);

  @override
  Future<int> stateSizeBytes({int sequenceId = 0}) =>
      _inner.stateSizeBytes(sequenceId: _sequenceId);

  @override
  Future<LlamaStashResult> stashState(String key, {int sequenceId = 0}) =>
      _inner.stashState(key, sequenceId: _sequenceId);

  @override
  Future<int> restoreStashedState(String key, {int sequenceId = 0}) =>
      _inner.restoreStashedState(key, sequenceId: _sequenceId);

  @override
  Future<int> dropStashedState(String key) => _inner.dropStashedState(key);

  @override
  Future<void> clearSequence(int sequenceId) =>
      _inner.clearSequence(_sequenceId);

  @override
  Future<void> setImageTokenBudget(int? imageTokenBudget) =>
      _inner.setImageTokenBudget(imageTokenBudget);

  @override
  Future<void> dispose() async {}
}

/// Serializes orchestrator mutations: activation, resize, load/unload.
class _TaskQueue {
  Future<void> _tail = Future<void>.value();

  Future<T> run<T>(Future<T> Function() action) {
    final result = _tail.then((_) => action());
    _tail = result.then<void>((_) {}, onError: (Object _, StackTrace _) {});
    return result;
  }
}
