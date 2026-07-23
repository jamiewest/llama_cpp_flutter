import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:agents/agents.dart' show ChatClientAgent;
import 'package:flutter_test/flutter_test.dart';
import 'package:llama_cpp_flutter/chat.dart';
import 'package:llama_cpp_flutter/llama_cpp_flutter.dart';
import 'package:llama_cpp_flutter/orchestration.dart';

final class _FakeSession implements LlamaSession {
  _FakeSession({
    this.canPersist = true,
    this.canStash = true,
    this.maxSequences = 1,
  });

  final bool canPersist;
  final bool canStash;
  final int maxSequences;

  /// Tokens `saveState`/`stashState` report (and, for saves, write a state
  /// file for when > 0).
  int cachedTokens = 0;

  /// Tokens `loadState`/`restoreStashedState` report restoring.
  int restoreTokens = 120;

  final List<String> savedPaths = <String>[];
  final List<String> loadedPaths = <String>[];
  final List<({String key, int sequenceId})> stashCalls = [];
  final List<({String key, int sequenceId})> restoreCalls = [];
  final List<int> clearedSequences = <int>[];
  final List<int> generateSequences = <int>[];
  final Set<String> stashKeys = <String>{};
  bool disposed = false;
  bool cancelled = false;

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
  }) {
    generateSequences.add(sequenceId);
    return const Stream<String>.empty();
  }

  @override
  Future<void> cancel() async {
    cancelled = true;
  }

  @override
  LlamaSessionCapabilities get capabilities => LlamaSessionCapabilities(
    canPersistState: canPersist,
    reportsStateSize: canPersist,
    canStashState: canStash,
    maxSequences: maxSequences,
  );

  @override
  Future<int> saveState(String path, {int sequenceId = 0}) async {
    savedPaths.add(path);
    if (cachedTokens > 0) File(path).writeAsStringSync('state');
    return cachedTokens;
  }

  @override
  Future<int> loadState(String path, {int sequenceId = 0}) async {
    loadedPaths.add(path);
    return restoreTokens;
  }

  @override
  Future<int> stateSizeBytes({int sequenceId = 0}) async => cachedTokens * 1000;

  @override
  Future<LlamaStashResult> stashState(String key, {int sequenceId = 0}) async {
    stashCalls.add((key: key, sequenceId: sequenceId));
    if (cachedTokens <= 0) return (tokens: 0, bytes: 0);
    stashKeys.add(key);
    return (tokens: cachedTokens, bytes: cachedTokens * 1000);
  }

  @override
  Future<int> restoreStashedState(String key, {int sequenceId = 0}) async {
    restoreCalls.add((key: key, sequenceId: sequenceId));
    if (!stashKeys.contains(key)) {
      throw StateError('No stashed state under $key');
    }
    return restoreTokens;
  }

  @override
  Future<int> dropStashedState(String key) async =>
      stashKeys.remove(key) ? 1 : 0;

  @override
  Future<void> clearSequence(int sequenceId) async {
    clearedSequences.add(sequenceId);
  }

  @override
  Future<void> setImageTokenBudget(int? imageTokenBudget) async {}

  @override
  Future<void> dispose() async {
    disposed = true;
  }
}

final class _FakeRuntime implements LlamaRuntime {
  _FakeRuntime({this.canPersist = true, this.canStash = true});

  final bool canPersist;
  final bool canStash;
  final List<ModelSpec> loads = <ModelSpec>[];
  _FakeSession? lastSession;

  @override
  bool get supportsMultiThreading => true;

  @override
  Future<LlamaSession> loadModel(
    ModelSpec spec, {
    String? localPath,
    String? localMmprojPath,
    String? localDraftPath,
    LlamaLoadProgress? onProgress,
  }) async {
    loads.add(spec);
    return lastSession = _FakeSession(
      canPersist: canPersist,
      canStash: canStash,
      maxSequences: spec.maxSequences,
    );
  }
}

ModelSpec _spec({int contextSize = 8192}) => ModelSpec(
  id: 'test-model',
  displayName: 'Test model',
  modelUrl: Uri.parse('https://example.com/model.gguf'),
  contextSize: contextSize,
  format: const Lfm2ChatFormat(),
);

const _estimate = ModelMemoryEstimate(
  weightsBytes: 400000000,
  kvBytesPerToken: 50000,
  fixedOverheadBytes: 100000000,
);

void main() {
  late Directory cacheDir;
  late _FakeRuntime runtime;
  late FixedMemoryMonitor monitor;
  late LlamaOrchestrator orchestrator;
  late List<OrchestratorEvent> events;

  LlamaOrchestrator create({
    _FakeRuntime? withRuntime,
    int sequenceSlots = 1,
    MemoryPolicy policy = const MemoryPolicy(pollInterval: Duration(hours: 1)),
  }) {
    final created = LlamaOrchestrator(
      runtime: withRuntime ?? runtime,
      cacheDirectory: cacheDir.path,
      policy: policy,
      sequenceSlots: sequenceSlots,
      memoryMonitor: monitor,
    );
    created.events.listen(events.add);
    return created;
  }

  setUp(() async {
    cacheDir = await Directory.systemTemp.createTemp('orchestrator_test');
    runtime = _FakeRuntime();
    monitor = FixedMemoryMonitor(
      const MemorySnapshot(totalBytes: 8000000000, availableBytes: 2000000000),
    );
    events = <OrchestratorEvent>[];
    orchestrator = create();
  });

  tearDown(() async {
    await orchestrator.dispose();
    await cacheDir.delete(recursive: true);
  });

  Future<void> load() => orchestrator.loadModel(
    _spec(),
    localPath: '${cacheDir.path}/model.gguf',
    estimateOverride: _estimate,
  );

  group('loadModel', () {
    test('sizes the context to the memory budget', () async {
      // 2e9 x 0.85 = 1.7e9 budget; minus weights+overhead = 1.2e9 of KV
      // at 50 KB/token = 24000 tokens, capped by the spec's 8192.
      await load();

      expect(orchestrator.contextTokens, 8192);
      expect(runtime.loads.single.contextSize, 8192);
      expect(events.whereType<ModelLoadedEvent>(), hasLength(1));
    });

    test('clamps to the budget when memory is tight', () async {
      monitor.snapshot = const MemorySnapshot(
        totalBytes: 8000000000,
        availableBytes: 1000000000,
      );
      // 1e9 x 0.85 = 850e6; minus 500e6 fixed = 350e6 -> 7000 tokens
      // -> 6912 after granularity rounding.
      await load();

      expect(orchestrator.contextTokens, 6912);
    });

    test(
      'loads at the minimum and reports critical when nothing fits',
      () async {
        monitor.snapshot = const MemorySnapshot(
          totalBytes: 8000000000,
          availableBytes: 550000000,
        );
        await load();

        expect(orchestrator.contextTokens, 2048);
        expect(events.whereType<MemoryCriticalEvent>(), hasLength(1));
      },
    );

    test('requests sequence slots on the loaded spec', () async {
      orchestrator = create(sequenceSlots: 3);
      await load();

      expect(runtime.loads.single.maxSequences, 3);
      expect(orchestrator.sequenceSlots, 3);
      expect(
        orchestrator.contextTokensPerAgent,
        orchestrator.contextTokens ~/ 3,
      );
    });
  });

  group('single-slot stash swaps', () {
    test('first activation restores nothing and preserves nothing', () async {
      await load();
      orchestrator.registerAgent(const AgentProfile(id: 'db'));

      final session = await orchestrator.sessionFor('db');

      expect(session, same(runtime.lastSession));
      expect(runtime.lastSession!.stashCalls, isEmpty);
      expect(runtime.lastSession!.savedPaths, isEmpty);
      expect(orchestrator.activeAgentId, 'db');
      final activated = events.whereType<AgentActivatedEvent>().single;
      expect(activated.restoredTokens, 0);
    });

    test(
      'switching stashes the previous agent and restores on return',
      () async {
        await load();
        orchestrator.registerAgent(const AgentProfile(id: 'db'));
        orchestrator.registerAgent(const AgentProfile(id: 'code'));
        final session = runtime.lastSession!;

        await orchestrator.sessionFor('db');
        session.cachedTokens = 120;
        await orchestrator.sessionFor('code');

        // db went to the RAM stash, not to disk.
        expect(session.stashCalls.single.key, 'db');
        expect(session.savedPaths, isEmpty);
        expect(orchestrator.stashedBytes, 120000);
        final stashed = events.whereType<StateStashedEvent>().single;
        expect(stashed.agentId, 'db');
        expect(stashed.tokens, 120);

        // Switching back stashes code on the way out, restores db from
        // the stash, and releases db's entry — only code's stash remains.
        await orchestrator.sessionFor('db');
        expect(session.restoreCalls.single.key, 'db');
        expect(session.stashCalls.last.key, 'code');
        expect(orchestrator.stashedBytes, 120000);
        final reactivated = events.whereType<AgentActivatedEvent>().last;
        expect(reactivated.restoredTokens, session.restoreTokens);
      },
    );

    test('falls back to disk snapshots when stashing is unsupported', () async {
      final diskRuntime = _FakeRuntime(canStash: false);
      orchestrator = create(withRuntime: diskRuntime);
      await load();
      orchestrator.registerAgent(const AgentProfile(id: 'db'));
      orchestrator.registerAgent(const AgentProfile(id: 'code'));
      final session = diskRuntime.lastSession!;

      await orchestrator.sessionFor('db');
      session.cachedTokens = 120;
      await orchestrator.sessionFor('code');

      expect(session.stashCalls, isEmpty);
      expect(session.savedPaths, hasLength(1));
      expect(events.whereType<SnapshotSavedEvent>().single.tokens, 120);

      await orchestrator.sessionFor('db');
      expect(session.loadedPaths, hasLength(1));
    });

    test('evicts the oldest stash beyond the byte cap', () async {
      orchestrator = create(
        policy: const MemoryPolicy(
          pollInterval: Duration(hours: 1),
          maxStashBytes: 150000,
        ),
      );
      await load();
      for (final id in ['a', 'b', 'c']) {
        orchestrator.registerAgent(AgentProfile(id: id));
      }
      final session = runtime.lastSession!..cachedTokens = 100;

      await orchestrator.sessionFor('a');
      await orchestrator.sessionFor('b'); // stashes a (100k)
      await orchestrator.sessionFor('c'); // stashes b (200k total) -> evict a

      final evicted = events.whereType<StashEvictedEvent>().single;
      expect(evicted.agentId, 'a');
      expect(orchestrator.stashedBytes, 100000);
      expect(session.stashKeys, {'b'});

      // a lost its stash (and has no disk snapshot): fresh prefill.
      await orchestrator.sessionFor('a');
      final reactivated = events.whereType<AgentActivatedEvent>().last;
      expect(reactivated.restoredTokens, 0);
    });

    test('re-activating the current agent is a no-op', () async {
      await load();
      orchestrator.registerAgent(const AgentProfile(id: 'db'));

      await orchestrator.sessionFor('db');
      await orchestrator.sessionFor('db');

      expect(events.whereType<AgentActivatedEvent>(), hasLength(1));
      expect(runtime.lastSession!.stashCalls, isEmpty);
    });

    test('sessions without any state support switch bare', () async {
      final bareRuntime = _FakeRuntime(canPersist: false, canStash: false);
      orchestrator = create(withRuntime: bareRuntime);
      await load();
      orchestrator.registerAgent(const AgentProfile(id: 'db'));
      orchestrator.registerAgent(const AgentProfile(id: 'code'));
      final session = bareRuntime.lastSession!..cachedTokens = 120;

      await orchestrator.sessionFor('db');
      await orchestrator.sessionFor('code');

      expect(session.stashCalls, isEmpty);
      expect(session.savedPaths, isEmpty);
      expect(orchestrator.activeAgentId, 'code');
    });

    test('throws without a loaded model', () async {
      orchestrator.registerAgent(const AgentProfile(id: 'db'));
      expect(orchestrator.sessionFor('db'), throwsStateError);
    });
  });

  group('multi-sequence residency', () {
    test('agents get their own sequences; switching swaps nothing', () async {
      orchestrator = create(sequenceSlots: 2);
      await load();
      orchestrator.registerAgent(const AgentProfile(id: 'db'));
      orchestrator.registerAgent(const AgentProfile(id: 'code'));
      final session = runtime.lastSession!..cachedTokens = 100;

      final dbSession = await orchestrator.sessionFor('db');
      final codeSession = await orchestrator.sessionFor('code');

      expect(session.stashCalls, isEmpty);
      expect(session.savedPaths, isEmpty);

      // The bound sessions route generations to their own sequences.
      await dbSession.generate('hi').drain<void>();
      await codeSession.generate('hi').drain<void>();
      expect(session.generateSequences, [0, 1]);

      // Switching back to a resident agent is free.
      await orchestrator.sessionFor('db');
      final back = events.whereType<AgentActivatedEvent>().last;
      expect(back.agentId, 'db');
      expect(back.wasResident, isTrue);
    });

    test('over-subscription evicts the least recently used agent', () async {
      orchestrator = create(sequenceSlots: 2);
      await load();
      for (final id in ['a', 'b', 'c']) {
        orchestrator.registerAgent(AgentProfile(id: id));
      }
      final session = runtime.lastSession!..cachedTokens = 100;

      await orchestrator.sessionFor('a'); // seq 0
      await orchestrator.sessionFor('b'); // seq 1
      await orchestrator.sessionFor('c'); // evicts a (LRU) -> stash + clear

      expect(session.stashCalls.single.key, 'a');
      expect(session.stashCalls.single.sequenceId, 0);
      expect(session.clearedSequences, [0]);

      // Reactivating a: b is now LRU -> evicted; a restored from stash
      // into b's freed sequence.
      await orchestrator.sessionFor('a');
      expect(session.stashCalls.last.key, 'b');
      expect(session.restoreCalls.single.key, 'a');
      expect(session.restoreCalls.single.sequenceId, 1);
      final reactivated = events.whereType<AgentActivatedEvent>().last;
      expect(reactivated.agentId, 'a');
      expect(reactivated.restoredTokens, session.restoreTokens);
    });

    test(
      'resize drops residency and restores the active agent from disk',
      () async {
        orchestrator = create(sequenceSlots: 2);
        await load();
        orchestrator.registerAgent(const AgentProfile(id: 'db'));
        await orchestrator.sessionFor('db');
        final first = runtime.lastSession!..cachedTokens = 100;

        monitor.snapshot = const MemorySnapshot(
          totalBytes: 8000000000,
          availableBytes: 50000000,
        );
        await orchestrator.checkMemory();

        expect(first.disposed, isTrue);
        expect(runtime.loads, hasLength(2));
        // Active agent's state went to disk (the stash dies with the
        // session) and came back into a sequence of the new session.
        expect(first.savedPaths, hasLength(1));
        expect(runtime.lastSession!.loadedPaths, hasLength(1));
        expect(orchestrator.stashedBytes, 0);
        final resized = events.whereType<ContextResizedEvent>().single;
        expect(resized.reason, ResizeReason.shrink);
      },
    );
  });

  group('memory-driven resize', () {
    test('shrinks immediately under pressure and reloads the model', () async {
      await load();
      orchestrator.registerAgent(const AgentProfile(id: 'db'));
      await orchestrator.sessionFor('db');
      final first = runtime.lastSession!..cachedTokens = 100;

      // 8e9 x 0.08 = 640e6 pressure threshold; 50e6 is well under it.
      monitor.snapshot = const MemorySnapshot(
        totalBytes: 8000000000,
        availableBytes: 50000000,
      );
      await orchestrator.checkMemory();

      expect(first.disposed, isTrue);
      expect(runtime.loads, hasLength(2));
      final resized = events.whereType<ContextResizedEvent>().single;
      expect(resized.reason, ResizeReason.shrink);
      expect(resized.toTokens, lessThan(8192));
      expect(orchestrator.contextTokens, resized.toTokens);
      expect(runtime.loads.last.contextSize, resized.toTokens);
      expect(first.savedPaths, hasLength(1));
      expect(runtime.lastSession!.loadedPaths, hasLength(1));
    });

    test('growth is deferred to the next activation', () async {
      monitor.snapshot = const MemorySnapshot(
        totalBytes: 8000000000,
        availableBytes: 1000000000,
      );
      await load();
      expect(orchestrator.contextTokens, 6912);
      orchestrator.registerAgent(const AgentProfile(id: 'db'));
      orchestrator.registerAgent(const AgentProfile(id: 'code'));
      await orchestrator.sessionFor('db');

      monitor.snapshot = const MemorySnapshot(
        totalBytes: 8000000000,
        availableBytes: 2000000000,
      );
      await orchestrator.checkMemory();
      // No resize yet: growth waits for a swap.
      expect(runtime.loads, hasLength(1));

      await orchestrator.sessionFor('code');
      expect(runtime.loads, hasLength(2));
      final resized = events.whereType<ContextResizedEvent>().single;
      expect(resized.reason, ResizeReason.grow);
      expect(orchestrator.contextTokens, greaterThan(6912));
    });

    test('shrinking invalidates snapshots that no longer fit', () async {
      await load();
      final store = SnapshotStore(cacheDir.path);
      await File(
        store.statePathFor('test-model', 'big'),
      ).writeAsString('state');
      await store.record(
        'test-model',
        'big',
        tokens: 7000,
        contextTokens: 8192,
      );

      monitor.snapshot = const MemorySnapshot(
        totalBytes: 8000000000,
        availableBytes: 50000000,
      );
      await orchestrator.checkMemory();

      expect(orchestrator.contextTokens, lessThan(7000));
      expect(await store.find('test-model', 'big'), isNull);
    });
  });

  group('agent registry', () {
    test('lists, looks up, and deduplicates registrations', () async {
      final db = orchestrator.registerAgent(const AgentProfile(id: 'db'));
      orchestrator.registerAgent(const AgentProfile(id: 'code'));

      expect(orchestrator.agentIds, ['db', 'code']);
      expect(orchestrator.agents.map((h) => h.profile.id), ['db', 'code']);
      expect(orchestrator.agent('db'), same(db));
      expect(orchestrator.agent('missing'), isNull);
      // Re-registering the same id returns the existing handle.
      expect(
        orchestrator.registerAgent(const AgentProfile(id: 'db')),
        same(db),
      );
      expect(orchestrator.agents, hasLength(2));
    });

    test('builds a ChatClientAgent from the profile', () async {
      await load();
      final handle = orchestrator.registerAgent(
        const AgentProfile(
          id: 'db',
          name: 'Database agent',
          description: 'Designs schemas.',
          instructions: 'You are a database engineer.',
        ),
      );

      final agent = handle.agent;
      expect(agent, isA<ChatClientAgent>());
      expect(agent.name, 'Database agent');
      expect(agent.description, 'Designs schemas.');
      final chatAgent = agent as ChatClientAgent;
      expect(chatAgent.instructions, 'You are a database engineer.');
      expect(chatAgent.idCore, 'db');
      // The same instance comes back on every access.
      expect(handle.agent, same(agent));
    });

    test('activating an unregistered agent throws', () async {
      await load();
      expect(orchestrator.sessionFor('ghost'), throwsStateError);
    });

    test('unregister cleans up stash and disk state', () async {
      await load();
      orchestrator.registerAgent(const AgentProfile(id: 'db'));
      orchestrator.registerAgent(const AgentProfile(id: 'code'));
      final session = runtime.lastSession!;

      await orchestrator.sessionFor('db');
      session.cachedTokens = 120;
      await orchestrator.sessionFor('code'); // db stashed

      expect(await orchestrator.unregisterAgent('db'), isTrue);
      expect(orchestrator.agent('db'), isNull);
      expect(orchestrator.agentIds, ['code']);
      expect(orchestrator.stashedBytes, 0);
      expect(session.stashKeys, isEmpty);
      expect(events.whereType<AgentUnregisteredEvent>().single.agentId, 'db');
      // Re-activating the removed id now fails.
      expect(orchestrator.sessionFor('db'), throwsStateError);
      // Unknown ids report false.
      expect(await orchestrator.unregisterAgent('ghost'), isFalse);
    });

    test('unregistering a resident agent frees its sequence slot', () async {
      orchestrator = create(sequenceSlots: 2);
      await load();
      for (final id in ['a', 'b', 'c']) {
        orchestrator.registerAgent(AgentProfile(id: id));
      }
      final session = runtime.lastSession!..cachedTokens = 100;

      await orchestrator.sessionFor('a'); // seq 0
      await orchestrator.sessionFor('b'); // seq 1

      expect(await orchestrator.unregisterAgent('a'), isTrue);
      expect(session.clearedSequences, [0]);
      expect(orchestrator.activeAgentId, 'b');

      // c takes a's freed slot without evicting b.
      await orchestrator.sessionFor('c');
      expect(session.stashCalls, isEmpty);
      final activated = events.whereType<AgentActivatedEvent>().last;
      expect(activated.agentId, 'c');
    });
  });

  group('unload and dispose', () {
    test('unload writes the active agent to disk and disposes', () async {
      await load();
      orchestrator.registerAgent(const AgentProfile(id: 'db'));
      await orchestrator.sessionFor('db');
      final session = runtime.lastSession!..cachedTokens = 80;

      await orchestrator.unloadModel();

      expect(session.disposed, isTrue);
      expect(session.savedPaths, hasLength(1));
      expect(orchestrator.isModelLoaded, isFalse);
      expect(orchestrator.activeAgentId, isNull);
      expect(orchestrator.contextTokens, 0);
      expect(orchestrator.stashedBytes, 0);
    });

    test('chat clients require a loaded model', () {
      final handle = orchestrator.registerAgent(const AgentProfile(id: 'db'));
      expect(() => handle.chatClient, throwsStateError);
    });
  });
}
