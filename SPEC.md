# llama_cpp_flutter orchestration — design spec (draft 1)

> Historical note: this layer began life as the standalone
> `llama_orchestrator` package, was merged into `agents_llama` (briefly
> `llama_cpp_flutter`), and now lives in `llama_cpp_flutter` as
> `lib/src/orchestration/`.

Status: **v1 + phase 2 implemented** (M0a, M0b, M1–M4 core, and both §9
phase-2 items: in-memory state blobs and multi-sequence contexts).

Phase-2, as built:

- **In-memory stash** (`llama_state_seq_get_data`/`set_data`): blobs live
  in a native-side keyed stash (`LlamaSession.swift`) and never cross the
  platform channel — `stashState`/`restoreStashedState` on the neutral
  interface. Single-slot agent swaps go RAM-to-RAM; disk snapshots are
  written only at persistence points (unload/resize/dispose) for the
  active agent. `MemoryPolicy.maxStashBytes` caps the stash (oldest-first
  eviction; evicted agents re-prefill).
- **Multi-sequence contexts** (`LlamaOrchestrator(sequenceSlots: N)`):
  the context is created with `n_seq_max = N` and `spec.contextSize`
  becomes the TOTAL KV budget (`llama_n_ctx_seq` per agent). Each
  resident agent owns a sequence — switching costs nothing; with more
  agents than slots the LRU resident is stashed and its slot reused.
  Sequence 0 keeps the historical auto-position decode path byte-for-byte;
  other sequences use explicit-position batches. Speculative decoding is
  sequence-0-only, so multi-sequence sessions drop the drafter at load
  (logged). Generation is still one-at-a-time (a session-wide serial
  queue); concurrent multi-sequence *batched* decode remains future work.
- The planner still works in total tokens; per-agent policy bounds
  (min/max/trained context) scale by the slot count.

v1 deviations from this spec, as built:

- The neutral `LlamaSession` interface also gained `stateSizeBytes()`
  (backed by the plugin's new `getSessionStateSize` / llama.cpp
  `llama_state_seq_get_size`), so estimator calibration is possible now
  rather than deferred.
- `SystemMemoryMonitor` is sample-only (`sample()`); there is no pressure
  event stream yet. Pressure is inferred from polled samples
  (`MemoryPolicy.pressureAvailableFraction`), and the
  `DISPATCH_SOURCE_TYPE_MEMORYPRESSURE` hook remains future plugin work.
- On memory-critical at load time the orchestrator loads at
  `minContextTokens` and emits `MemoryCriticalEvent` instead of refusing
  to load; `MemoryPolicy.unloadOnCritical` (default false) governs the
  running case.
- Open question 1 resolved as designed: agent switches await in-flight
  generation (the plugin's per-session serial queue provides this
  naturally); resizes cancel it.

## 1. Problem statement

One model (e.g. Gemma), many purpose-specific agents (database agent, code
agent, …), each with its own system prompt, tools, and conversation history —
running on a device whose memory budget is unknown ahead of time and changes
while the app runs.

Today the stack has the right seams but no owner:

- `LlamaChatClient` binds to a session lazily through
  `SessionProvider` (`Future<LlamaSession> Function()`,
  `llama_chat_client.dart:21`) and deliberately does **not** own the session
  lifecycle. Something external is supposed to own load/dispose/state — that
  something doesn't exist yet.
- `llama_cpp_flutter` already has KV-cache persistence
  (`LlamaSession.saveState(path)` / `loadState(path)`, backed by
  `llama_state_seq_save_file`/`load_file` in `LlamaSession.swift`), but it is
  **not surfaced** through the neutral `LlamaSession` interface in
  `llama_cpp_flutter`.
- Nothing anywhere estimates memory or watches system pressure. Context size
  is whatever `ModelSpec.contextSize` says, and if it doesn't fit, the OS
  jetsams the app (iOS) or swaps (macOS).

The orchestration layer owns all of this: model
residency, per-agent context state, memory budgeting, and dynamic context
resizing.

### A note on which parameters cost memory

The prompt for this design mentioned "model parameters like temp, top k,
context size". Only some of these are memory levers:

| parameter | memory impact |
|---|---|
| model file / quant | dominant fixed cost (weights) |
| `contextSize` (n_ctx) | linear KV-cache cost — **the** runtime lever |
| `n_batch` / `n_ubatch` | compute-buffer size (plugin caps at 2048) |
| draft model | second set of weights + small context |
| mmproj | projector weights, loaded lazily |
| `temperature`, `topK`, `topP`, `seed` | **none** — pure sampling params |

So the estimator keys off model metadata + context size; sampling defaults
stay in `SamplingDefaults` and are irrelevant to the planner.

## 2. Goals / non-goals

**Goals (v1)**

1. Load a model once; serve N registered agents from it.
2. Per-agent KV-cache snapshots so switching agents restores their context
   instead of re-prefilling ("swap contexts as the agent changes").
3. Estimate required memory from GGUF metadata + requested context size
   *before* loading; pick the largest context that fits the budget.
4. Continuously monitor system memory; shrink (or grow) contexts in response,
   accepting a re-prefill pause when a shrink invalidates a snapshot.
5. Same public API on native and web, with documented degraded behavior on
   web (no state save/load in wllama → switch = re-prefill).

**Non-goals (v1)**

- Multiple models resident at once (design leaves room; see §9).
- Multi-sequence batching (several agents decoding concurrently in one
  context). Phase-2 candidate; see §9.
- Any change to the Agents Framework side — agents keep consuming a plain
  `ChatClient`.

## 3. Package layout

```
packages/llama_cpp_flutter/lib/src/orchestration/
  orchestrator.dart            # LlamaOrchestrator (core)
  agent_profile.dart           # AgentProfile, AgentHandle
  memory/memory_monitor.dart   # SystemMemoryMonitor (conditional io/web)
  memory/memory_estimator.dart # ModelMemoryEstimator
  memory/budget_planner.dart   # ContextBudgetPlanner + MemoryPolicy
  state/snapshot_store.dart    # per-agent KV snapshot files
```

Dependencies: `llama_cpp_flutter` (runtime, chat clients, GGUF reader,
`ModelSpec`), `extensions` (ChatClient types). It must **not** depend on
`llama_cpp_flutter` directly — everything platform-specific goes through the
neutral `LlamaSession` interface, which we extend (§7).

## 4. Public API sketch

```dart
final orchestrator = LlamaOrchestrator(
  runtime: createLlamaRuntime(),
  cacheDirectory: appSupportDir.path,          // snapshots + model artifacts
  policy: MemoryPolicy(
    headroomFraction: 0.15,   // keep 15% of available memory free
    minContextTokens: 2048,   // below this, refuse to shrink; evict instead
    maxContextTokens: 32768,
    resizeHysteresisFraction: 0.10, // ignore <10% budget wobble (§6.3)
  ),
);

// Loads the model with an orchestrator-chosen n_ctx (clamped to
// ModelSpec.contextSize). Downloads via ModelDownloader if needed.
await orchestrator.loadModel(gemmaSpec, onProgress: ...);

final dbAgent = orchestrator.registerAgent(AgentProfile(
  id: 'database',
  instructions: 'You create and modify SQLite databases…',
  tools: [createTableTool, runSqlTool],
  priority: AgentPriority.high,      // last to lose context under pressure
  preferredContextTokens: 8192,      // hint; planner may give less
));

// ChatClient for the Agents Framework. SessionProvider inside it routes
// through the orchestrator: resolving it activates the agent (KV swap).
final AIAgent agent = AIAgent(chatClient: dbAgent.chatClient, ...);

orchestrator.events.listen((e) { ... }); // resize/swap/pressure telemetry
await orchestrator.dispose();
```

`AgentProfile` fields: `id`, `instructions`, `tools`, `priority`,
`preferredContextTokens`, optional `sampling` override, optional
`format` override. `AgentHandle` exposes `chatClient`, `snapshotTokens`
(how much context a resume restores), and `invalidate()` (drop snapshot,
next turn starts fresh).

## 5. Core mechanism: one session, swapped KV state

### 5.1 Why one session

In `llama_cpp_flutter`, one session = one model **copy** + one context; loading
the same model twice duplicates the weights (no model sharing across
sessions). So for a single set of weights the orchestrator keeps exactly one
active `LlamaSession` and time-slices it between agents. Weight duplication
is exactly what a memory orchestrator must not do.

### 5.2 Activation protocol

The `SessionProvider` handed to each agent's `LlamaChatClient` is:

```dart
Future<LlamaSession> _provideFor(String agentId) async {
  await _activate(agentId);   // serialized on an internal mutex
  return _session;
}
```

`_activate(next)`:

1. If `next` is already active → return.
2. If a generation is in flight for the current agent → await it (or cancel,
   policy flag; default **await** — never silently kill an agent's turn).
3. `saveState(snapshotPath(current))` — persists current agent's KV cache.
   Returns 0 if nothing cached (fine — no file written, drop stale snapshot).
4. If `next` has a valid snapshot **and** it was taken at the current n_ctx
   (or smaller): `loadState(snapshotPath(next))`. On any load error: clear
   and fall through to (5).
5. Otherwise: cleared cache; the next `generate` call re-prefills from the
   rendered transcript. This is the "wait that bit for it to reload" path.

Snapshot bookkeeping lives in `SnapshotStore`: one file per agent under
`<cacheDirectory>/orchestrator/<modelCacheKey>/<agentId>.llstate` plus a
small JSON sidecar (`tokens`, `nCtx` at save time, model cache key, mtime).
Snapshots are invalidated when the model changes or when n_ctx shrinks below
the snapshot's token count.

### 5.3 Swap cost and why snapshots beat re-prefill

KV bytes/token ≈ `2 (K+V) × n_layer × n_embd_gqa × bytes_per_element`
(f16 = 2). For a Gemma-class 4B model (~34 layers, GQA), a full 8k context
is on the order of a few hundred MB → a save+load round trip is disk-bound,
roughly 1–3 s on NVMe-class storage. Re-prefilling 8k tokens at typical
on-device prefill rates is many times that. So: swap by default, re-prefill
only when the snapshot is invalid. (Phase 2 upgrades to in-memory blobs,
§9.)

Note `messagesWithRuntimeContext` in `llama_chat_client.dart` already
relocates volatile context messages to keep the KV prefix stable — the
orchestrator inherits that benefit per agent for free.

### 5.4 Web degradation

wllama exposes no state save/load. On web the orchestrator keeps the same
API: `saveState` is a no-op returning 0, activation of a different agent
just means the next generation re-prefills (wllama's own prefix cache helps
when the same agent takes consecutive turns). `OrchestratorCapabilities`
(queried from the session, §7) tells callers which behavior they're getting.

## 6. Memory budgeting

### 6.1 SystemMemoryMonitor

Interface:

```dart
abstract interface class SystemMemoryMonitor {
  Future<MemorySnapshot> sample();          // {totalBytes, availableBytes, appFootprintBytes?}
  Stream<MemoryPressureEvent> get pressure; // OS pressure signals where available
}
```

- **macOS/iOS (v1, pure Dart FFI — no plugin change needed):**
  `sysctlbyname("hw.memsize")` for total; `host_statistics64`
  (`vm_statistics64`: free + inactive + purgeable) for available;
  `task_info(TASK_VM_INFO).phys_footprint` for own footprint. On iOS also
  `os_proc_available_memory()` — the actual jetsam headroom, far more
  honest than vm stats. Pressure stream via
  `DISPATCH_SOURCE_TYPE_MEMORYPRESSURE`; that one *does* need a small
  platform hook, so v1 may ship polling-only (default 5 s interval,
  configurable) with the dispatch source added to `llama_cpp_flutter` later
  (§7, optional).
- **Web:** `navigator.deviceMemory` (coarse, capped at 8) and, where
  present, `performance.measureUserAgentSpecificMemory()`. Honest answer:
  web budgeting is heuristic — the wasm32 heap cap (≤4 GiB) is the real
  ceiling, and the existing 2 GiB-per-file OPFS split machinery already
  acknowledges that. The planner uses fixed conservative defaults on web.

### 6.2 ModelMemoryEstimator

Input: GGUF header (via the existing `readGgufMetadata` /
`GgufReader` in `llama_cpp_flutter`, with the key set extended — §7) + file size.

```dart
class MemoryEstimate {
  int weightsBytes;        // ≈ file size (mmapped, but resident when offloaded)
  int kvBytesPerToken;     // 2 × n_layer_kv × n_embd_k_gqa+v_gqa × elemSize
  int fixedOverheadBytes;  // compute graph + output buffers, per formula below
  int bytesForContext(int nCtx);
  int maxContextForBudget(int budgetBytes);
}
```

Keys read: `<arch>.block_count`, `<arch>.embedding_length`,
`<arch>.attention.head_count`, `<arch>.attention.head_count_kv`,
`<arch>.attention.key_length` / `value_length`, `<arch>.context_length`,
plus SWA keys where present (`<arch>.attention.sliding_window`) — Gemma-3
style interleaved SWA layers make the true KV cost lower than the naive
formula; v1 estimates **conservatively** (assume all layers full-attention)
and treats accuracy as a calibration item. Compute-buffer overhead is
estimated as `c₁ × n_ubatch × n_embd + c₂ × vocab` with constants
calibrated against real loads (the plugin already caps `n_outputs_max`, so
logits are cheap); a `calibration.md` records measured vs. predicted for a
few reference models.

Ground truth hook: once §7's `getStateSize`/`getMemoryInfo` land, the
estimator self-corrects — after load, compare predicted vs. actual footprint
and scale the constants.

### 6.3 ContextBudgetPlanner

Pure function, unit-testable:

```dart
PlannerDecision plan({
  required MemorySnapshot memory,
  required MemoryEstimate model,
  required List<AgentProfile> agents,
  required int currentNCtx,
  required MemoryPolicy policy,
});
```

- Budget = `availableBytes × (1 − headroomFraction)` minus weights and fixed
  overhead → KV budget → `maxContextForBudget`, clamped to
  `[minContextTokens, min(policy.max, spec.contextSize, model trained ctx)]`.
- **Hysteresis:** only emit a resize if the target differs from current by
  more than `resizeHysteresisFraction` *and* the last resize was more than
  `resizeCooldown` (default 30 s) ago — resizing means killing and reloading
  the context, so thrash is worse than being 10% off.
- Shrink is immediate on an OS pressure event (bypasses cooldown); growth is
  lazy (only at the next agent switch, when the context is being swapped
  anyway).
- If even `minContextTokens` doesn't fit → emit
  `OrchestratorEvent.memoryCritical` and (policy option) unload the model
  rather than get jetsammed.

### 6.4 Resize protocol

llama.cpp contexts can't change n_ctx in place, so resize = recreate:

1. Wait for in-flight generation (or cancel under critical pressure).
2. `saveState` active agent (only useful when growing).
3. Dispose session; `loadModel` again with new `contextSize` (model file is
   in the artifact cache — reload is weight-load time, no download).
4. Grow: `loadState` restores the active agent fully. Shrink: snapshots with
   `tokens > newNCtx` are invalidated; the active agent re-prefills its
   (format-rendered, possibly history-trimmed) transcript on next turn —
   the accepted "wait a bit" path. Snapshots that still fit stay valid.

## 7. Required changes in the other layers (prerequisites)

**`llama_cpp_flutter` (M0a):**

1. Surface state ops on the neutral interface — additive:
   ```dart
   abstract interface class LlamaSession {
     ...
     LlamaSessionCapabilities get capabilities; // {canPersistState, ...}
     Future<int> saveState(String path);   // web: returns 0
     Future<int> loadState(String path);   // web: throws UnsupportedError
   }
   ```
   `_NativeLlamaSession` forwards to the existing plugin methods; the web
   session implements the degraded contract.
2. Export a way to read GGUF metadata for a *local* file conveniently
   (today's reader takes a header prefix; add a small
   `readGgufMetadataFromFile(path, keys)` io helper).
3. No changes to `LlamaChatClient` — `SessionProvider` is already the right
   seam.

**`llama_cpp_flutter` (M0b, small Pigeon additions):**

1. `@async int getStateSize(int sessionId)` → `llama_state_seq_get_size`
   (exact KV+state byte size — feeds estimator calibration).
2. `@async MemoryInfo getMemoryInfo()` → phys_footprint +
   `os_proc_available_memory` (iOS) — optional if the FFI monitor proves
   sufficient, but the plugin is the natural home for
   `DISPATCH_SOURCE_TYPE_MEMORYPRESSURE` → an event on the existing
   `LlamaTokenStream`-style channel (or a second channel).

Both are additive; Pigeon regen per the existing workflow. Nothing here
touches the staging `llama_ext_` ABI.

## 8. Concurrency & failure semantics

- The orchestrator serializes **activation and resize** on one internal
  mutex; generation itself streams as today. One agent generates at a time
  (v1) — a second agent's request queues at the `SessionProvider` await.
- Every snapshot load failure degrades to clear-and-re-prefill, never to an
  error surfaced to the agent. `loadSessionState` on the plugin already
  guarantees "throws ⇒ empty cache", which makes this safe.
- Model reload failures during resize retry once at the previous n_ctx; if
  that also fails → `memoryCritical` event, orchestrator enters `unloaded`
  state, next agent request triggers a fresh load attempt.
- All transitions emit `OrchestratorEvent`s (agentActivated, snapshotSaved /
  Restored / Invalidated, contextResized{from,to,reason}, memoryPressure,
  memoryCritical) so the app can show "reloading context…" UI during the
  re-prefill wait.

## 9. Phase 2+ (explicitly out of v1)

- **In-memory state blobs:** `llama_state_seq_get_data`/`set_data` instead
  of files — removes disk I/O from the swap path; orchestrator keeps hot
  snapshots in RAM (they then count against the budget — planner learns a
  second knob: spill snapshot to disk vs. keep hot).
- **Multi-sequence single context:** each agent = a llama.cpp seq id in one
  context (`llama_memory_seq_*` are all unused today). Removes swap cost
  entirely and enables concurrent agents, but is a much larger
  `LlamaSession.swift` change (the whole file assumes seq 0) and interacts
  with SWA prefix-reuse guards. The spec's activation protocol is designed
  so this can replace §5.2 behind the same `SessionProvider` seam.
- **Multi-model:** `engineId` on `ModelSpec` is the documented dispatch
  seam; orchestrator grows a model table and the planner an eviction policy
  (priority × recency) across models.
- **Draft-model awareness:** speculative decoding doubles some costs; v1
  simply adds draft file size + a small fixed context to the estimate.

## 10. Open questions

1. **Cancel vs. await on switch** (§5.2 step 2): default is await; is there
   an agent UX where preemption matters enough to expose per-call?
2. **Snapshot retention:** unbounded per-agent files, LRU cap, or
   policy-driven (`maxSnapshotBytes`)? Proposal: LRU with a byte cap
   defaulting to 2 × (KV size at current n_ctx).
3. **Per-agent n_ctx:** v1 gives every agent the same n_ctx (one context).
   Worth planning per-agent sizes now, or defer to the multi-sequence phase
   where it becomes natural?
4. **iOS memory entitlement:** should the planner read
   `com.apple.developer.kernel.increased-memory-limit` availability into its
   budget, or stay entitlement-agnostic via `os_proc_available_memory`?

## 11. Milestones

| # | deliverable | depends on |
|---|---|---|
| M0a | `llama_cpp_flutter`: state ops + capabilities on neutral interface; GGUF file helper | — |
| M0b | `llama_cpp_flutter`: `getStateSize`, memory info/pressure (optional) | — |
| M1 | package scaffold; estimator + planner (pure, fully unit-tested); FFI memory monitor | M0a |
| M2 | orchestrator core: registry, activation/swap, snapshot store; integration test with a real model swapping two agents | M1 |
| M3 | monitoring loop + dynamic resize with hysteresis; pressure-event fast path | M2, (M0b) |
| M4 | web degradation pass + capability reporting; example app: DB agent + second agent sharing one Gemma | M2 |
