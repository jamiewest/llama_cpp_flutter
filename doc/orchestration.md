# Orchestration in `llama_flutter`

Memory-aware orchestration of purpose-specific agents over a single
loaded llama.cpp model. One model, many agents (a database agent, a code
agent, …), each with its own KV-cache context that is swapped in when the
agent takes a turn — sized to the device's actual memory and resized as
conditions change.

See [SPEC.md](SPEC.md) for the full design.

## Usage

```dart
import 'package:llama_flutter/llama_flutter.dart';

final orchestrator = LlamaOrchestrator(
  runtime: createLlamaRuntime(),
  cacheDirectory: appSupportDir.path,
);

// Downloads (native), reads the GGUF header for a memory estimate, and
// loads with a context size that fits the current memory budget.
await orchestrator.loadModel(gemmaSpec);

final db = orchestrator.registerAgent(const AgentProfile(
  id: 'database',
  name: 'Database agent',
  instructions: 'You are a database engineer…',
  tools: [createTableTool],
));
final code = orchestrator.registerAgent(const AgentProfile(id: 'code'));

// Each handle exposes a full Agents Framework AIAgent (a
// ChatClientAgent built from the profile's name/description/
// instructions/tools). Running it activates the agent: its KV state is
// made resident first (own sequence, stash restore, or re-prefill).
final session = await db.agent.createSession();
final response = await db.agent.run(session, null, message: 'Design a schema.');

// Registry management:
orchestrator.agents;                     // all handles
orchestrator.agent('database');          // lookup by id
await orchestrator.unregisterAgent('code'); // remove + clean up KV state

orchestrator.events.listen((event) {
  // ModelLoadedEvent, AgentActivatedEvent, SnapshotSaved/Restored,
  // ContextResizedEvent, MemoryCriticalEvent — drive UI/logging here.
});
```

## What it does

- **Multi-sequence residency** — with `sequenceSlots: N`, each resident
  agent owns a llama.cpp KV sequence inside the one loaded context;
  switching agents costs nothing. Over-subscription evicts the
  least-recently-used resident into the RAM stash and reuses its slot.
  (Disables speculative decoding; `contextSize` becomes the total KV
  budget split across slots.)
- **In-memory KV stash** — single-slot swaps (and multi-sequence
  evictions) copy KV state into an engine-side RAM stash
  (`llama_state_seq_get_data`/`set_data`) — no disk and no bytes over the
  platform channel. Capped by `MemoryPolicy.maxStashBytes` (oldest
  evicted first). Disk snapshots are written only at unload/resize/
  dispose. A missing/invalid state degrades to a re-prefill, never an
  error. On web (wllama) there is no state API, so switches always
  re-prefill; the orchestrator API is identical.
- **Memory estimation** — reads layer count, GQA head counts, and head
  dimensions from the GGUF header to compute KV bytes/token, and combines
  that with the file size and a fixed overhead into a
  `ModelMemoryEstimate`.
- **Budgeted context sizing** — samples system memory (sysctl /
  host_statistics64 / os_proc_available_memory via FFI on Apple
  platforms) and picks the largest context that fits under a headroom
  policy, clamped to the spec, policy, and trained context.
- **Dynamic resizing** — a polling loop re-plans continuously. Shrinks
  apply immediately under pressure (llama.cpp contexts cannot resize in
  place, so this is a model reload + snapshot restore); growth waits for
  the next agent switch. Hysteresis and a cooldown prevent thrash.

## Sampling parameters are not memory

Temperature, top-k, and top-p do not affect memory use — they are pure
sampling knobs. The memory levers are the model file (quantization),
context size, and batch size; the planner keys off those only.
