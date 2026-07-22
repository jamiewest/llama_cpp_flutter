#if os(iOS)
  import Flutter
#elseif os(macOS)
  import FlutterMacOS
#endif
import Foundation
import llama

/// Outcome callbacks for a single generation run.
///
/// `onDone` receives the Pigeon `GenerationStats` (Messages.g.swift) so the
/// counts, finish reason, and timings cross the bridge unchanged.
struct GenerationCallbacks {
  let onToken: (String) -> Void
  let onDone: (GenerationStats) -> Void
  let onError: (String) -> Void
}

/// A prompt-ingestion failure carrying a human-readable message.
///
/// Needed because `Result`'s failure type must conform to `Error`.
struct GenerationError: Error {
  let message: String
  init(_ message: String) { self.message = message }
}

/// Writes a diagnostic line to stderr.
///
/// stderr rather than `print`: llama.cpp's own logs go to stderr, and when the
/// app's stdout is a pipe (e.g. under `flutter run`) it is block-buffered, so
/// `print` output can sit unflushed and be lost if the process aborts.
private func logNative(_ message: String) {
  fputs("[llama_cpp_flutter] \(message)\n", stderr)
}

/// Microseconds elapsed since `start`, for the stats reported to Dart.
private func microsecondsSince(_ start: DispatchTime) -> Int64 {
  Int64((DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1_000)
}

/// Returns a regex matching every tensor in the GGUF at `modelPath` whose
/// quant type has no Metal kernels, plus the match count — or nil when all
/// tensor types are Metal-capable.
///
/// The pinned llama.cpp (see LLAMA_REF) ships no Metal kernels for the
/// ternary types tq1_0/tq2_0, but `ggml_metal_device_supports_op` still
/// claims MUL_MAT support for them, so such weights get placed in Metal
/// buffers and the first matmul dereferences a nil pipeline ("Function
/// kernel_mul_mm_tq2_0_f32 was not found in the library"). Re-check this
/// type set whenever the pin moves, like the LlamaExtShim signatures.
private func metalUnsupportedTensorPattern(
  modelPath: String
) -> (pattern: String, count: Int)? {
  let params = gguf_init_params(no_alloc: true, ctx: nil)
  guard let gguf = modelPath.withCString({ gguf_init_from_file($0, params) })
  else { return nil }
  defer { gguf_free(gguf) }

  var names: [String] = []
  for i in 0..<gguf_get_n_tensors(gguf) {
    let type = gguf_get_tensor_type(gguf, i)
    if type.rawValue == GGML_TYPE_TQ1_0.rawValue
      || type.rawValue == GGML_TYPE_TQ2_0.rawValue
    {
      names.append(String(cString: gguf_get_tensor_name(gguf, i)))
    }
  }
  guard !names.isEmpty else { return nil }
  // One anchored alternation of the exact names: a single override entry
  // sidesteps llama_max_tensor_buft_overrides, and llama.cpp matches the
  // pattern with std::regex_search, so anchoring keeps it exact.
  let escaped = names.map { $0.replacingOccurrences(of: ".", with: "\\.") }
  return ("^(" + escaped.joined(separator: "|") + ")$", names.count)
}

/// Loads the model at `path`, pinning any Metal-unsupported tensors into CPU
/// buffers via `tensor_buft_overrides` so the scheduler runs their ops on the
/// CPU backend instead of crashing on a missing Metal kernel.
///
/// The scan is skipped on the pure-CPU path (`n_gpu_layers == 0`, e.g. the
/// iOS simulator), where every type is supported anyway.
private func loadModelPinningUnsupportedTensorsToCpu(
  path: String, params: llama_model_params
) -> OpaquePointer? {
  var params = params
  guard params.n_gpu_layers != 0,
    let (pattern, count) = metalUnsupportedTensorPattern(modelPath: path)
  else {
    return path.withCString { llama_model_load_from_file($0, params) }
  }
  logNative(
    "keeping \(count) ternary-quant tensor(s) on CPU (no Metal kernels for "
      + "tq1_0/tq2_0 at this llama.cpp pin): \(path)")
  let cPattern = strdup(pattern)
  defer { free(cPattern) }
  let overrides = [
    llama_model_tensor_buft_override(
      pattern: cPattern, buft: ggml_backend_cpu_buffer_type()),
    llama_model_tensor_buft_override(pattern: nil, buft: nil),
  ]
  return overrides.withUnsafeBufferPointer { buffer in
    params.tensor_buft_overrides = buffer.baseAddress
    return path.withCString { llama_model_load_from_file($0, params) }
  }
}

/// Number of physical performance cores, matching llama.cpp's own thread
/// default (`cpu_get_num_math`).
///
/// ggml's threadpool spin-synchronises all workers per op, so including
/// efficiency cores makes CPU-resident work *slower* — the P-cores stall
/// waiting for E-core stragglers on every matmul. This matters wherever ops
/// actually run on CPU: the ternary-quant tensors
/// `loadModelPinningUnsupportedTensorsToCpu` pins (every E2B decode pays
/// those late-block FFNs per token) and the fully-CPU simulator path.
private func performanceCoreCount() -> Int32 {
  var count: Int32 = 0
  var size = MemoryLayout<Int32>.size
  if sysctlbyname("hw.perflevel0.physicalcpu", &count, &size, nil, 0) == 0,
    count > 0
  {
    return count
  }
  return Int32(ProcessInfo.processInfo.activeProcessorCount)
}

/// Wraps a loaded llama.cpp model + context and runs generation on a dedicated
/// serial queue so the platform thread is never blocked.
///
/// All llama.cpp pointers are owned here and freed in `dispose()`.
final class LlamaSession {
  private let model: OpaquePointer
  private let context: OpaquePointer
  private let vocab: OpaquePointer
  /// `mtmd` (multimodal) context, created lazily by `ensureMtmd()` on the
  /// first image turn. Only touched on `queue`.
  private var mtmd: OpaquePointer?
  /// Projector path the session was loaded with, or nil when the session
  /// cannot accept image input.
  private let mmprojPath: String?
  private let mtmdUseGpu: Bool
  /// Per-image vision-encoder token budget (mtmd image_min/max_tokens), or
  /// nil to keep the projector's metadata default. Mutable via
  /// `setImageTokenBudget`; only touched on `queue` after init.
  private var imageTokenBudget: Int32?
  /// Drafter model/context, non-nil when the session was loaded with a
  /// `DraftModelOptions`, enabling speculative decoding for text generations.
  private let draft: DraftModel?
  private let queue: DispatchQueue
  /// Sliding-window size of the model's SWA layers (0 = none); bounds how far
  /// the KV cache can be rewound when reusing a prompt prefix.
  private let nSwa: Int32

  /// Per-sequence ledgers of the tokens known to occupy KV positions
  /// `[0, count)` of the main context (and, for sequence 0 when speculation
  /// ran, the draft context) after the last cleanly finished generation. A
  /// missing/empty entry means that sequence holds nothing reusable and
  /// must be cleared before the next decode. Only touched on `queue`.
  private var cachedTokensBySeq: [llama_seq_id: [llama_token]] = [:]

  /// Sequence-0 view of the ledger, kept because the speculative decoding
  /// paths (standalone and MTP) are sequence-0 only.
  private var cachedTokens: [llama_token] {
    get { cachedTokensBySeq[0] ?? [] }
    set { cachedTokensBySeq[0] = newValue }
  }

  /// In-memory KV-state stash (`llama_state_seq_get_data` blobs plus token
  /// ledgers), keyed by caller-chosen names. Lives and dies with the
  /// session. Only touched on `queue`.
  private var stash: [String: StashEntry] = [:]

  private struct StashEntry {
    let tokens: [llama_token]
    let data: [UInt8]
  }

  /// True when the standalone drafter's context no longer mirrors
  /// `cachedTokens` (set by `loadState`, which only restores the target).
  /// The next speculative run then re-feeds the drafter the full prompt
  /// instead of trusting the shared ledger. MTP drafters are unaffected —
  /// they share the target's memory and never hold prompt state. Only
  /// touched on `queue`.
  private var draftCacheStale = false

  /// Set from any thread to abort an in-flight generation.
  private var cancelled = false
  /// Set by `dispose()` so generations still queued behind it bail out
  /// instead of resetting the cancel flag and running against a session
  /// whose pointers are about to be freed.
  private var disposed = false
  private let lock = NSLock()

  private init(
    model: OpaquePointer, context: OpaquePointer, mmprojPath: String?,
    mtmdUseGpu: Bool, imageTokenBudget: Int32?, draft: DraftModel?,
    sessionId: Int64
  ) {
    self.model = model
    self.context = context
    self.vocab = llama_model_get_vocab(model)!
    self.nSwa = llama_model_n_swa(model)
    self.mmprojPath = mmprojPath
    self.mtmdUseGpu = mtmdUseGpu
    self.imageTokenBudget = imageTokenBudget
    self.draft = draft
    self.queue = DispatchQueue(label: "dev.llama_cpp_flutter.session.\(sessionId)")
  }

  /// Loads `request.modelPath` into a new session, or returns `nil` on failure.
  static func load(request: ModelLoadRequest, sessionId: Int64) -> LlamaSession? {
    var modelParams = llama_model_default_params()
    modelParams.n_gpu_layers = Int32(request.gpuLayers)

    let model = loadModelPinningUnsupportedTensorsToCpu(
      path: request.modelPath, params: modelParams)
    guard let model else { return nil }

    var ctxParams = llama_context_default_params()
    ctxParams.n_ctx = UInt32(request.contextSize)
    // n_ubatch caps the physical batch at 2048 anyway, so a larger n_batch
    // only lets one llama_decode carry more tokens (decodeChunked splits to
    // n_batch regardless) while inflating n_batch-sized output buffers — the
    // unmasked nextn buffer the MTP drafter path enables is n_batch × n_embd
    // floats (~170 MB at a 16k n_batch, ~20 MB at 2048).
    // Floored at the image token budget: vision encoders with non-causal
    // attention (Gemma) need the whole image chunk inside one ubatch, or
    // llama_decode trips "non-causal attention requires n_ubatch >= n_tokens".
    let physicalBatch = max(
      min(request.contextSize, 2048), request.imageTokenBudget ?? 0)
    ctxParams.n_batch = UInt32(physicalBatch)
    // Larger physical batches speed up prefill (default 512); threads matter
    // wherever ops run on CPU — gpuLayers 0 (iOS simulator) and the
    // ternary-quant tensors pinned to CPU buffers at load. P-cores only: see
    // performanceCoreCount.
    ctxParams.n_ubatch = UInt32(physicalBatch)
    let threads = performanceCoreCount()
    ctxParams.n_threads = threads
    ctxParams.n_threads_batch = threads
    // No decode ever flags more than draftBudget+1 logits rows (a
    // speculative verification batch; everything else flags one), but the
    // reserve-time logits buffer is n_outputs_max × n_vocab per ubatch graph
    // and n_outputs_max defaults to n_batch — ~2 GB of Metal compute buffer
    // at Gemma's 262k vocab. Upstream's server caps it the same way.
    let draftBudget = Int(request.draftModel.map { min(max($0.maxDraftTokens, 0), 64) } ?? 0)
    // Floor of maxSequences: the context constructor pre-reserves one
    // output row per sequence and output_reserve asserts
    // n_outputs_max >= n_seq_max (a few MB per row at a 262k vocab).
    let maxSequences = max(Int(request.maxSequences ?? 1), 1)
    ctxParams.n_outputs_max = UInt32(max(1 + draftBudget, maxSequences))
    // Rollback snapshots for recurrent/hybrid models (e.g. LFM2's conv
    // layers), whose state cannot be partially erased otherwise:
    // llama_memory_seq_rm on them succeeds only for trailing removals of at
    // most n_rs_seq positions. Speculation trims up to draftBudget rejected
    // tokens per step and prefix reuse trims at least one token at every
    // turn boundary — with 0 the first fails mid-generation and the second
    // falls back to a full-prompt re-decode every turn. Mirrors upstream's
    // need_n_rs_seq() (= draft n_max) plus a floor of 1 for the prefix-reuse
    // trim; pure-attention models (Gemma) clamp it back to 0 internally.
    // Multiple independent KV sequences (one per agent/conversation). The
    // draft is dropped for multi-sequence sessions: the speculative paths
    // are sequence-0 only and their buffer/rollback sizing has never been
    // exercised against a partitioned KV cache.
    ctxParams.n_seq_max = UInt32(maxSequences)
    ctxParams.n_rs_seq = UInt32(max(draftBudget, 1) * maxSequences)

    guard let context = llama_init_from_model(model, ctxParams) else {
      llama_model_free(model)
      return nil
    }

    // The multimodal (mtmd) context is NOT created here: the projector
    // weights are hundreds of MB of memory that text-only sessions never
    // touch — phones especially. ensureMtmd() stands it up on the first
    // image turn; a bad projector therefore surfaces as a generation error
    // on that turn instead of failing the whole load.

    // Optionally load the drafter for speculative decoding. Failure here
    // (logged by DraftModel.load) degrades to single-model decoding instead
    // of aborting: the drafter is purely a speed optimization and must not
    // take the main model down with it.
    var draft: DraftModel?
    if request.draftModel != nil, maxSequences > 1 {
      logNative(
        "draft model ignored: speculative decoding is sequence-0 only and "
          + "this session was loaded with maxSequences=\(maxSequences)")
    } else if let options = request.draftModel {
      draft = DraftModel.load(
        options: options, contextSize: Int(request.contextSize),
        mainModel: model, mainContext: context)
      if draft == nil {
        logNative("continuing without speculative decoding")
      }
    }
    if let draft, draft.isMtp {
      // The MTP drafter consumes the target's nextn rows (the hidden state
      // before the final output norm), so the main context must output them.
      // Unmasked, exactly like upstream's target context: in masked mode the
      // gemma4 graph skips the output-ids gather entirely (see gemma4.cpp in
      // llama.cpp), which misaligns logits and nextn rows for any batch that
      // doesn't flag every token — masked is only valid for the drafter's
      // fully-flagged batches. Unmasked rows are indexed by raw batch
      // position and sized by n_batch (capped at load for this reason).
      llama_ext_set_embeddings_nextn(
        UnsafeMutableRawPointer(context), true, false)
    }

    return LlamaSession(
      model: model, context: context, mmprojPath: request.mmprojPath,
      mtmdUseGpu: request.gpuLayers != 0,
      imageTokenBudget: request.imageTokenBudget.map { Int32($0) },
      draft: draft, sessionId: sessionId)
  }

  /// Returns the mtmd context, creating it from the projector on first use.
  ///
  /// Must be called on `queue` (generation and dispose both run there).
  /// Returns nil when the session has no projector or its load fails; a
  /// failed load is logged and retried on the next image turn.
  private func ensureMtmd() -> OpaquePointer? {
    if let mtmd { return mtmd }
    guard let mmprojPath else { return nil }
    var mtmdParams = mtmd_context_params_default()
    mtmdParams.use_gpu = mtmdUseGpu
    if let imageTokenBudget {
      // Pin the dynamic-resolution budget to one value (Gemma accepts 70,
      // 140, 280, 560, or 1120); load() already floored n_ubatch at it.
      mtmdParams.image_min_tokens = imageTokenBudget
      mtmdParams.image_max_tokens = imageTokenBudget
    }
    mtmd = mmprojPath.withCString { cMmproj in
      mtmd_init_from_file(cMmproj, model, mtmdParams)
    }
    if mtmd == nil {
      logNative("mmproj projector failed to load: \(mmprojPath)")
    }
    return mtmd
  }

  /// Changes the per-image vision token budget for subsequent image turns;
  /// nil restores the projector's metadata default.
  ///
  /// Frees the mtmd context so the next image turn re-creates it with the
  /// new budget (`ensureMtmd` reloads the projector from disk); the LLM
  /// context and every KV cache are untouched. Queue placement serializes
  /// this behind any in-flight generation. The budget cannot exceed the
  /// `n_ubatch` fixed at load: the vision encoder's non-causal image chunk
  /// must fit one micro-batch, and growing that requires a model reload.
  ///
  /// Completion runs on the session queue with an error message, or nil on
  /// success.
  func setImageTokenBudget(_ budget: Int64?, completion: @escaping (String?) -> Void) {
    queue.async { [self] in
      lock.lock()
      let disposed = self.disposed
      lock.unlock()
      if disposed {
        completion("session is disposed")
        return
      }
      let ubatch = Int64(llama_n_ubatch(context))
      if let budget, budget < 1 || budget > ubatch {
        completion(
          "imageTokenBudget \(budget) is outside [1, \(ubatch)]; n_ubatch is "
            + "fixed at load, so a larger budget needs the model reloaded "
            + "with that imageTokenBudget")
        return
      }
      let newBudget = budget.map { Int32($0) }
      guard newBudget != imageTokenBudget else {
        completion(nil)
        return
      }
      imageTokenBudget = newBudget
      if let mtmd {
        mtmd_free(mtmd)
        self.mtmd = nil
      }
      completion(nil)
    }
  }

  func cancel() {
    lock.lock()
    cancelled = true
    lock.unlock()
  }

  private var isCancelled: Bool {
    lock.lock()
    defer { lock.unlock() }
    return cancelled
  }

  /// Saves the KV cache + token ledger to `path` (llama_state_seq format).
  ///
  /// Completion runs on the session queue with `(tokenCount, error)`:
  /// a positive count on success, `0` when there is nothing cached (no file
  /// is written). Queue placement means a snapshot requested between
  /// generations captures the finished turn, never a partial one.
  func saveState(
    toPath path: String, sequenceId seq: llama_seq_id,
    completion: @escaping (Int64, String?) -> Void
  ) {
    queue.async { [weak self] in
      guard let self else { return }
      let tokens = self.cachedTokensBySeq[seq] ?? []
      guard !tokens.isEmpty else {
        completion(0, nil)
        return
      }
      let written = tokens.withUnsafeBufferPointer { buffer in
        llama_state_seq_save_file(
          self.context, path, seq, buffer.baseAddress, buffer.count)
      }
      guard written > 0 else {
        completion(-1, "Failed to save session state to \(path)")
        return
      }
      logNative(
        "saved KV state (seq \(seq)): \(tokens.count) tokens, "
          + "\(written) bytes -> \(path)")
      completion(Int64(tokens.count), nil)
    }
  }

  /// Restores KV cache + token ledger previously written by `saveState`.
  ///
  /// The current cache is cleared first, so on any failure (missing file,
  /// incompatible model, corrupt data) the session degrades to an empty
  /// cache and the next generation prefills from scratch.
  func loadState(
    fromPath path: String, sequenceId seq: llama_seq_id,
    completion: @escaping (Int64, String?) -> Void
  ) {
    queue.async { [weak self] in
      guard let self else { return }
      self.cachedTokensBySeq[seq] = []
      self.eraseSequence(seq, in: self.context)
      // A restored target cache says nothing about the standalone drafter's
      // context; force the next speculative run to re-feed it the prompt.
      if seq == 0 { self.draftCacheStale = true }

      let capacity = Int(llama_n_ctx_seq(self.context))
      var tokens = [llama_token](repeating: 0, count: capacity)
      var count = 0
      let read = tokens.withUnsafeMutableBufferPointer { buffer in
        llama_state_seq_load_file(
          self.context, path, seq, buffer.baseAddress, capacity, &count)
      }
      guard read > 0, count > 0 else {
        self.eraseSequence(seq, in: self.context)
        completion(-1, "Failed to restore session state from \(path)")
        return
      }
      self.cachedTokensBySeq[seq] = Array(tokens.prefix(count))
      logNative("restored KV state (seq \(seq)): \(count) tokens from \(path)")
      completion(Int64(count), nil)
    }
  }

  /// Measures the byte size of the current single-sequence state (KV cache
  /// plus bookkeeping) without writing it anywhere — the exact size
  /// `saveState` would produce for the current cache.
  ///
  /// Completion runs on the session queue with `(sizeBytes, error)`. Queue
  /// placement means a measurement requested between generations sizes the
  /// finished turn, never a partial one.
  func stateSize(
    sequenceId seq: llama_seq_id,
    completion: @escaping (Int64, String?) -> Void
  ) {
    queue.async { [weak self] in
      guard let self else { return }
      guard !(self.cachedTokensBySeq[seq] ?? []).isEmpty else {
        completion(0, nil)
        return
      }
      let size = llama_state_seq_get_size(self.context, seq)
      completion(Int64(size), nil)
    }
  }

  /// Copies one sequence's KV state into the in-memory stash under `key`
  /// (replacing any existing entry) without touching the sequence itself.
  ///
  /// Completion runs on the session queue with `(tokens, bytes, error)`;
  /// `(0, 0, nil)` means the sequence held nothing reusable and nothing was
  /// stashed. Restoring later is a RAM-to-RAM copy — no disk, no channel.
  func stashState(
    sequenceId seq: llama_seq_id, key: String,
    completion: @escaping (Int64, Int64, String?) -> Void
  ) {
    queue.async { [weak self] in
      guard let self else { return }
      let tokens = self.cachedTokensBySeq[seq] ?? []
      guard !tokens.isEmpty else {
        completion(0, 0, nil)
        return
      }
      let size = llama_state_seq_get_size(self.context, seq)
      guard size > 0 else {
        completion(-1, 0, "Failed to size sequence \(seq) state")
        return
      }
      var data = [UInt8](repeating: 0, count: size)
      let written = data.withUnsafeMutableBufferPointer { buffer in
        llama_state_seq_get_data(self.context, buffer.baseAddress, size, seq)
      }
      guard written > 0 else {
        completion(-1, 0, "Failed to copy sequence \(seq) state")
        return
      }
      if written < size { data.removeLast(size - written) }
      self.stash[key] = StashEntry(tokens: tokens, data: data)
      logNative(
        "stashed KV state '\(key)' (seq \(seq)): \(tokens.count) tokens, "
          + "\(written) bytes")
      completion(Int64(tokens.count), Int64(written), nil)
    }
  }

  /// Restores the stash entry under `key` into `seq`, replacing that
  /// sequence's cache. The entry is kept until dropped explicitly.
  ///
  /// Completion runs on the session queue with `(tokens, error)`; on any
  /// failure the sequence is left empty so the next generation prefills.
  func restoreStashedState(
    sequenceId seq: llama_seq_id, key: String,
    completion: @escaping (Int64, String?) -> Void
  ) {
    queue.async { [weak self] in
      guard let self else { return }
      guard let entry = self.stash[key] else {
        completion(-1, "No stashed state under '\(key)'")
        return
      }
      self.cachedTokensBySeq[seq] = []
      self.eraseSequence(seq, in: self.context)
      if seq == 0 { self.draftCacheStale = true }
      let read = entry.data.withUnsafeBufferPointer { buffer in
        llama_state_seq_set_data(
          self.context, buffer.baseAddress, entry.data.count, seq)
      }
      guard read > 0 else {
        self.eraseSequence(seq, in: self.context)
        completion(-1, "Failed to restore stashed state '\(key)'")
        return
      }
      self.cachedTokensBySeq[seq] = entry.tokens
      logNative(
        "restored stashed KV state '\(key)' -> seq \(seq): "
          + "\(entry.tokens.count) tokens")
      completion(Int64(entry.tokens.count), nil)
    }
  }

  /// Drops the stash entry under `key`. Completion gets the bytes freed
  /// (`0` when no entry existed).
  func dropStashedState(
    key: String, completion: @escaping (Int64, String?) -> Void
  ) {
    queue.async { [weak self] in
      guard let self else { return }
      let bytes = self.stash.removeValue(forKey: key)?.data.count ?? 0
      completion(Int64(bytes), nil)
    }
  }

  /// Erases one sequence's KV cache and ledger.
  func clearSequence(
    _ seq: llama_seq_id, completion: @escaping (String?) -> Void
  ) {
    queue.async { [weak self] in
      guard let self else { return }
      self.cachedTokensBySeq[seq] = []
      self.eraseSequence(seq, in: self.context)
      if seq == 0 { self.draftCacheStale = true }
      completion(nil)
    }
  }

  /// Removes every KV position of `seq` from `target`'s memory. Single-
  /// sequence contexts keep the historical full-clear (identical effect,
  /// also resets recurrent state); multi-sequence contexts remove just the
  /// one sequence, falling back to a full clear (and a full ledger wipe)
  /// only if the memory backend cannot do partial removal.
  private func eraseSequence(_ seq: llama_seq_id, in target: OpaquePointer) {
    let memory = llama_get_memory(target)
    if llama_n_seq_max(target) == 1 {
      llama_memory_clear(memory, true)
    } else if !llama_memory_seq_rm(memory, seq, -1, -1) {
      logNative("sequence \(seq) removal unsupported; clearing all sequences")
      llama_memory_clear(memory, true)
      if target == context { cachedTokensBySeq = [:] }
    }
  }

  /// Runs generation asynchronously, streaming token text through `callbacks`.
  func generate(request: GenerationRequest, callbacks: GenerationCallbacks) {
    queue.async { [weak self] in
      guard let self else { return }
      // Reset on the serial queue, not at submit time: a previous run may
      // still be executing ahead of this block, and clearing the flag early
      // would clobber a cancel() aimed at it.
      self.lock.lock()
      let disposed = self.disposed
      self.cancelled = false
      self.lock.unlock()
      guard !disposed else {
        callbacks.onError("Session was disposed")
        return
      }
      self.runGeneration(request: request, callbacks: callbacks)
    }
  }

  private func runGeneration(
    request: GenerationRequest, callbacks: GenerationCallbacks
  ) {
    let sampler = makeSampler(request: request)
    defer { llama_sampler_free(sampler) }

    let seq = llama_seq_id(truncatingIfNeeded: request.sequenceId ?? 0)

    // Each prompt re-renders the full conversation, so consecutive turns
    // share a long prefix; reuse the part of the KV cache that still matches
    // instead of re-decoding it. The sequence's ledger is invalidated here
    // and committed only when a run finishes cleanly, so error paths leave
    // it empty and the next call falls back to a full clear + decode.
    let previous = cachedTokensBySeq[seq] ?? []
    cachedTokensBySeq[seq] = []

    // Ingest the prompt into the context. With images this runs through mtmd,
    // interleaving image embeddings with text; otherwise it is a plain text
    // decode. Either way `nPast` is the position to continue sampling from.
    let images = request.images ?? []
    let nPast: Int
    var promptTokens: [llama_token] = []
    var reused = 0
    let prefillStarted = DispatchTime.now()
    if images.isEmpty {
      promptTokens = tokenize(request.prompt, addBos: true)
      switch decodeTextPrompt(promptTokens, previous: previous, seq: seq) {
      case .success(let n): reused = n
      case .failure(let error): callbacks.onError(error.message); return
      }
      nPast = promptTokens.count
    } else {
      guard let mtmd = ensureMtmd() else {
        callbacks.onError(
          "No multimodal projector is available; cannot accept images")
        return
      }
      // Prefix reuse covers text turns only: mtmd evaluates from position 0
      // and image chunks have no token representation to match against, so
      // start from an empty sequence (its ledger stays invalidated).
      eraseSequence(seq, in: context)
      switch evalMultimodalPrompt(
        mtmd: mtmd, prompt: request.prompt, images: images, seq: seq)
      {
      case .success(let n): nPast = n
      case .failure(let error): callbacks.onError(error.message); return
      }
    }

    let prefillMicroseconds = microsecondsSince(prefillStarted)
    let prefillSeconds = Double(prefillMicroseconds) / 1_000_000
    let prefillDecoded = nPast - reused
    logNative(
      String(
        format:
          "prefill: reused %d of %d tokens; decoded %d in %.2fs (%.1f tok/s)",
        reused, nPast, prefillDecoded, prefillSeconds,
        prefillSeconds > 0 ? Double(prefillDecoded) / prefillSeconds : 0))

    // Speculate only for sequence-0 text generations: mtmd evaluates image
    // embeddings into the main context alone, so the drafter could never
    // see them, and the speculative paths are sequence-0 only (multi-
    // sequence sessions drop the drafter at load).
    if let draft, draft.maxDraftTokens > 0, images.isEmpty, seq == 0 {
      if draft.isMtp {
        // An MTP drafter never decodes the prompt: it shares the target's
        // memory via ctx_other and is seeded from the target's hidden states
        // instead. Re-decode the prompt's final token as its own single-token
        // batch first: unmasked nextn rows are indexed per decode batch, so
        // this pins the seed row (and the first sample's logits) to index 0
        // regardless of how the prompt prefill was chunked.
        let lastPosition = llama_pos(promptTokens.count - 1)
        guard
          llama_memory_seq_rm(
            llama_get_memory(context), 0, lastPosition, -1),
          decodeOne(promptTokens[promptTokens.count - 1], into: context)
        else {
          callbacks.onError("Failed to re-decode the prompt tail for MTP")
          return
        }
        runSpeculativeMtp(
          request: request, draft: draft, sampler: sampler,
          promptTokens: promptTokens, reusedPromptTokens: reused,
          prefillMicroseconds: prefillMicroseconds, callbacks: callbacks)
        return
      }
      // The draft cache mirrors the main one at every commit point (the
      // speculative loop trims both back to the accepted sequence each step),
      // so `previous` describes its contents too — unless a state restore
      // repopulated only the target (`draftCacheStale`).
      let draftPrevious = draftCacheStale ? [] : previous
      let draftReused = reuseCachedPrefix(
        promptTokens, previous: draftPrevious, in: draft.context,
        nSwa: draft.nSwa)
      guard
        decodeChunked(
          Array(promptTokens[draftReused...]), into: draft.context)
      else {
        callbacks.onError("Failed to decode prompt in the draft context")
        return
      }
      draftCacheStale = false
      runSpeculative(
        request: request, draft: draft, sampler: sampler,
        promptTokens: promptTokens, reusedPromptTokens: reused,
        prefillMicroseconds: prefillMicroseconds, callbacks: callbacks)
      return
    }

    let nCtx = Int(llama_n_ctx_seq(context))
    var emitter = TokenEmitter(
      stops: request.stopSequences, onToken: callbacks.onToken)
    var generated = 0
    var position = nPast

    // Ledger of the tokens actually decoded into the KV cache (the emitter's
    // stop-sequence holdback means emitted text can lag what the cache
    // holds). Committed on every clean finish so the next turn can reuse the
    // prefix; image turns are excluded because mtmd positions aren't tokens.
    var decoded = promptTokens
    let started = DispatchTime.now()
    func finish(_ reason: FinishReason) {
      if images.isEmpty {
        cachedTokensBySeq[seq] = decoded
      }
      callbacks.onDone(
        GenerationStats(
          promptTokenCount: Int64(nPast),
          cachedTokenCount: Int64(reused),
          generatedTokenCount: Int64(generated),
          finishReason: reason,
          prefillMicroseconds: prefillMicroseconds,
          decodeMicroseconds: microsecondsSince(started)))
    }

    var endReason = FinishReason.maxTokens
    while generated < request.maxTokens {
      if isCancelled {
        finish(.cancelled)
        return
      }
      if position >= nCtx {
        endReason = .contextFull
        break
      }

      let newToken = llama_sampler_sample(sampler, context, -1)
      if llama_vocab_is_eog(vocab, newToken) {
        endReason = .eogToken
        break
      }

      if emitter.push(pieceBytes(for: newToken)) {
        finish(.stopSequence)
        return
      }
      let tokenPosition = position
      generated += 1
      position += 1

      // Sequence 0 keeps llama_batch_get_one's auto-assigned positions
      // (the historical path); other sequences need explicit positions and
      // an explicit seq id, which llama_batch_get_one cannot express.
      let ok =
        seq == 0
        ? decodeOne(newToken, into: context)
        : decodeOneInSequence(newToken, seq: seq, position: tokenPosition)
      if !ok {
        callbacks.onError("Failed to decode token")
        return
      }
      decoded.append(newToken)
    }

    emitter.finish()
    let elapsed =
      Double(DispatchTime.now().uptimeNanoseconds - started.uptimeNanoseconds)
      / 1_000_000_000
    logNative(
      String(
        format: "decode: %d tokens in %.2fs (%.1f tok/s)",
        generated, elapsed, elapsed > 0 ? Double(generated) / elapsed : 0))
    finish(endReason)
  }

  /// Generic draft-and-verify (speculative) decode loop.
  ///
  /// Per step: the drafter greedily proposes up to `maxDraftTokens` tokens,
  /// the main model verifies them in a single batched decode, and the longest
  /// prefix whose tokens match what the main sampler produces is accepted.
  /// Because every emitted token was sampled from the main model, the output
  /// distribution is identical to single-model decoding; the drafter only
  /// changes speed. Both contexts hold the prompt at entry, and `position`
  /// (the next write index) is kept in lockstep between them by rolling back
  /// rejected speculation after each step.
  private func runSpeculative(
    request: GenerationRequest, draft: DraftModel,
    sampler: UnsafeMutablePointer<llama_sampler>,
    promptTokens: [llama_token], reusedPromptTokens: Int,
    prefillMicroseconds: Int64, callbacks: GenerationCallbacks
  ) {
    let nCtx = Int(llama_n_ctx(context))
    var emitter = TokenEmitter(
      stops: request.stopSequences, onToken: callbacks.onToken)
    var generated = 0
    var position = promptTokens.count

    // Ledger of the tokens occupying both contexts' KV caches at every
    // commit point (i.e. after each step's rollback trim); committed on
    // clean finishes so the next turn can reuse the prefix. A stop firing
    // mid acceptance-walk returns before that step's trim, leaving drafted
    // tokens in the caches beyond the ledger — the next call's unconditional
    // seq_rm in `reuseCachedPrefix` wipes them.
    var decoded = promptTokens
    var draftedTotal = 0
    var acceptedTotal = 0
    let started = DispatchTime.now()
    func finish(_ reason: FinishReason) {
      cachedTokens = decoded
      callbacks.onDone(
        GenerationStats(
          promptTokenCount: Int64(promptTokens.count),
          cachedTokenCount: Int64(reusedPromptTokens),
          generatedTokenCount: Int64(generated),
          finishReason: reason,
          prefillMicroseconds: prefillMicroseconds,
          decodeMicroseconds: microsecondsSince(started),
          draftedTokenCount: Int64(draftedTotal),
          acceptedTokenCount: Int64(acceptedTotal)))
    }

    // The drafter always drafts greedily; the requested temperature applies
    // to the main model's sampler, which has the final say on every token.
    let draftSampler = llama_sampler_chain_init(
      llama_sampler_chain_default_params())!
    llama_sampler_chain_add(draftSampler, llama_sampler_init_greedy())
    defer { llama_sampler_free(draftSampler) }

    var batch = llama_batch_init(Int32(draft.maxDraftTokens + 1), 0, 1)
    defer { llama_batch_free(batch) }

    var current = llama_sampler_sample(sampler, context, -1)

    var endReason = FinishReason.maxTokens
    while generated < request.maxTokens {
      if isCancelled {
        finish(.cancelled)
        return
      }
      if position >= nCtx {
        endReason = .contextFull
        break
      }
      if llama_vocab_is_eog(vocab, current) {
        endReason = .eogToken
        break
      }
      if emitter.push(pieceBytes(for: current)) {
        finish(.stopSequence)
        return
      }
      generated += 1

      // This step's verification batch occupies positions
      // `position ..< position + 1 + budget`, so cap drafting to what the
      // context and the remaining token allowance can actually use. A zero
      // budget means the loop is about to terminate anyway, so skipping the
      // draft feed below cannot desynchronise the contexts.
      let budget = min(
        draft.maxDraftTokens, nCtx - position - 1,
        Int(request.maxTokens) - generated)

      // Draft phase: feed `current` to the drafter, then let it propose
      // tokens one at a time. Every proposal is fed back so the draft context
      // stays one step ahead, ready for the next proposal.
      var drafted: [llama_token] = []
      if budget > 0 {
        guard decodeOne(current, into: draft.context) else {
          callbacks.onError("Failed to decode token in the draft context")
          return
        }
        while drafted.count < budget {
          let proposal = llama_sampler_sample(draftSampler, draft.context, -1)
          if llama_vocab_is_eog(vocab, proposal) {
            break
          }
          guard decodeOne(proposal, into: draft.context) else {
            callbacks.onError("Failed to decode token in the draft context")
            return
          }
          drafted.append(proposal)
        }
      }

      // Verification: one decode of `current` plus the proposals, with logits
      // at every index so the main sampler can be consulted per position.
      fill(batch: &batch, tokens: [current] + drafted, startPosition: position)
      guard llama_decode(context, batch) == 0 else {
        callbacks.onError("Failed to decode verification batch")
        return
      }

      // Acceptance walk: the sample at index i is the main model's token
      // following `current` and the first i proposals. A proposal is accepted
      // iff it equals that sample; the first divergence (or the sample after
      // a fully accepted draft) becomes the next `current`.
      var accepted = 0
      var next: llama_token = 0
      for index in 0...drafted.count {
        let token = llama_sampler_sample(sampler, context, Int32(index))
        if index < drafted.count && token == drafted[index] {
          if emitter.push(pieceBytes(for: token)) {
            finish(.stopSequence)
            return
          }
          generated += 1
          accepted += 1
        } else {
          next = token
          break
        }
      }

      // Drop the rejected tail of the speculation from both contexts so the
      // next step continues from the accepted sequence.
      position += 1 + accepted
      llama_memory_seq_rm(
        llama_get_memory(context), 0, llama_pos(position), -1)
      llama_memory_seq_rm(
        llama_get_memory(draft.context), 0, llama_pos(position), -1)
      decoded.append(current)
      decoded.append(contentsOf: drafted.prefix(accepted))

      draftedTotal += drafted.count
      acceptedTotal += accepted
      current = next
    }

    emitter.finish()
    let elapsed =
      Double(DispatchTime.now().uptimeNanoseconds - started.uptimeNanoseconds)
      / 1_000_000_000
    logNative(
      String(
        format:
          "speculative: %d tokens in %.2fs (%.1f tok/s), "
          + "draft acceptance %d/%d",
        generated, elapsed, elapsed > 0 ? Double(generated) / elapsed : 0,
        acceptedTotal, draftedTotal))
    finish(endReason)
  }

  /// MTP (multi-token-prediction) draft-and-verify decode loop, a
  /// single-sequence port of llama.cpp's `common_speculative_impl_draft_mtp`
  /// (common/speculative.cpp at the pinned LLAMA_REF).
  ///
  /// Differences from the standalone loop in `runSpeculative`: the drafter
  /// shares the target's memory (ctx_other), so it never decodes the prompt
  /// and rejected speculation is trimmed from the target only. Each drafter
  /// step consumes a (token, hidden-state) pair — the token just produced
  /// plus the *nextn* row (hidden state before the final output norm) of the
  /// position before it: the target's row for the first proposal of a step,
  /// the drafter's own row for the rest. All drafted tokens occupy the same
  /// position (shared-memory MTP semantics; the verification decode then
  /// writes the real rows). Verification is identical to the standalone
  /// loop, so the output distribution still exactly matches single-model
  /// decoding.
  private func runSpeculativeMtp(
    request: GenerationRequest, draft: DraftModel,
    sampler: UnsafeMutablePointer<llama_sampler>,
    promptTokens: [llama_token], reusedPromptTokens: Int,
    prefillMicroseconds: Int64, callbacks: GenerationCallbacks
  ) {
    let nCtx = Int(llama_n_ctx(context))
    var emitter = TokenEmitter(
      stops: request.stopSequences, onToken: callbacks.onToken)
    var generated = 0
    var position = promptTokens.count

    // Ledger of the tokens occupying the target's KV cache at every commit
    // point; same stray-token caveat as `runSpeculative` applies to early
    // returns mid acceptance-walk.
    var decoded = promptTokens
    var draftedTotal = 0
    var acceptedTotal = 0
    let started = DispatchTime.now()
    func finish(_ reason: FinishReason) {
      cachedTokens = decoded
      callbacks.onDone(
        GenerationStats(
          promptTokenCount: Int64(promptTokens.count),
          cachedTokenCount: Int64(reusedPromptTokens),
          generatedTokenCount: Int64(generated),
          finishReason: reason,
          prefillMicroseconds: prefillMicroseconds,
          decodeMicroseconds: microsecondsSince(started),
          draftedTokenCount: Int64(draftedTotal),
          acceptedTokenCount: Int64(acceptedTotal)))
    }

    let nEmbd = draft.nEmbd
    let rowBytes = nEmbd * MemoryLayout<Float>.size

    // The hidden row pairing with the *next* token fed to the drafter
    // (upstream's pending_h): h of position p-1 when the next drafter input
    // token sits at position p. Seeded from the prompt's last token, which
    // `runGeneration` just re-decoded as a single-token batch — unmasked
    // nextn rows are batch-position indexed, so its row is index 0.
    var pendingH = [Float](repeating: 0, count: nEmbd)
    guard
      let promptH = llama_ext_get_embeddings_nextn_ith(
        UnsafeMutableRawPointer(context), 0)
    else {
      callbacks.onError("MTP drafter: no nextn row after the prompt decode")
      return
    }
    memcpy(&pendingH, promptH, rowBytes)

    // The drafter always drafts greedily; the requested temperature applies
    // to the main model's sampler, which has the final say on every token.
    let draftSampler = llama_sampler_chain_init(
      llama_sampler_chain_default_params())!
    llama_sampler_chain_add(draftSampler, llama_sampler_init_greedy())
    defer { llama_sampler_free(draftSampler) }

    var verifyBatch = llama_batch_init(Int32(draft.maxDraftTokens + 1), 0, 1)
    defer { llama_batch_free(verifyBatch) }

    // Single-token draft batch carrying BOTH a token id and an embedding row.
    // llama_batch_init allocates only one of the two (embd here), so the
    // token array is allocated manually — and detached again before
    // llama_batch_free, which would otherwise free() Swift-allocated memory.
    var draftBatch = llama_batch_init(1, Int32(nEmbd), 1)
    draftBatch.token = UnsafeMutablePointer<llama_token>.allocate(capacity: 1)
    defer {
      draftBatch.token.deallocate()
      draftBatch.token = nil
      llama_batch_free(draftBatch)
    }

    // Feeds one (token, hidden-row) pair to the drafter. Every draft token of
    // a step is decoded at the current commit position, not sequential ones.
    func decodeDraft(_ token: llama_token, h: UnsafePointer<Float>) -> Bool {
      draftBatch.n_tokens = 1
      draftBatch.token[0] = token
      draftBatch.pos[0] = llama_pos(position)
      draftBatch.n_seq_id[0] = 1
      draftBatch.seq_id[0]![0] = 0
      draftBatch.logits[0] = 1
      memcpy(draftBatch.embd, h, rowBytes)
      return llama_decode(draft.context, draftBatch) == 0
    }

    var current = llama_sampler_sample(sampler, context, -1)

    var endReason = FinishReason.maxTokens
    while generated < request.maxTokens {
      if isCancelled {
        finish(.cancelled)
        return
      }
      if position >= nCtx {
        endReason = .contextFull
        break
      }
      if llama_vocab_is_eog(vocab, current) {
        endReason = .eogToken
        break
      }
      if emitter.push(pieceBytes(for: current)) {
        finish(.stopSequence)
        return
      }
      generated += 1

      // Same budget cap as the standalone loop: the verification batch
      // occupies positions `position ..< position + 1 + budget`.
      let budget = min(
        draft.maxDraftTokens, nCtx - position - 1,
        Int(request.maxTokens) - generated)

      // Draft phase: seed with (current, pendingH), then chain proposals,
      // each paired with the drafter's own nextn row from the decode that
      // produced it.
      var drafted: [llama_token] = []
      if budget > 0 {
        let seeded = pendingH.withUnsafeBufferPointer {
          decodeDraft(current, h: $0.baseAddress!)
        }
        guard seeded else {
          callbacks.onError("Failed to decode token in the MTP draft context")
          return
        }
        while true {
          let proposal = llama_sampler_sample(draftSampler, draft.context, -1)
          if llama_vocab_is_eog(vocab, proposal) {
            break
          }
          drafted.append(proposal)
          if drafted.count >= budget {
            break
          }
          guard
            let hRow = llama_ext_get_embeddings_nextn_ith(
              UnsafeMutableRawPointer(draft.context), -1)
          else {
            break
          }
          guard decodeDraft(proposal, h: hRow) else {
            callbacks.onError(
              "Failed to decode token in the MTP draft context")
            return
          }
        }
      }

      // Verification: identical to the standalone loop.
      fill(
        batch: &verifyBatch, tokens: [current] + drafted,
        startPosition: position)
      guard llama_decode(context, verifyBatch) == 0 else {
        callbacks.onError("Failed to decode verification batch")
        return
      }

      var accepted = 0
      var next: llama_token = 0
      for index in 0...drafted.count {
        let token = llama_sampler_sample(sampler, context, Int32(index))
        if index < drafted.count && token == drafted[index] {
          if emitter.push(pieceBytes(for: token)) {
            finish(.stopSequence)
            return
          }
          generated += 1
          accepted += 1
        } else {
          next = token
          break
        }
      }

      // Re-seed the drafter input from the verification decode: `next` was
      // sampled at batch row `accepted`, so the hidden row of the position
      // before it is that same row (upstream's accept(): pending_h =
      // verify_h[min(n_accepted, n_rows-1)]).
      guard
        let verifiedH = llama_ext_get_embeddings_nextn_ith(
          UnsafeMutableRawPointer(context), Int32(accepted))
      else {
        callbacks.onError("MTP drafter: no nextn row after verification")
        return
      }
      memcpy(&pendingH, verifiedH, rowBytes)

      // Drop the rejected tail from the target; the drafter shares this
      // memory, so there is no second cache to trim.
      position += 1 + accepted
      guard
        llama_memory_seq_rm(
          llama_get_memory(context), 0, llama_pos(position), -1)
      else {
        // A recurrent/hybrid cache that ran out of rollback snapshots: the
        // cache now holds the rejected tokens' state and no ledger describes
        // it, so stop rather than keep decoding against corrupt state.
        // cachedTokens stays invalidated (finish() is deliberately skipped),
        // forcing a clean re-decode on the next call.
        logNative(
          "speculative (mtp): rejected-tail rollback failed, "
            + "ending generation")
        emitter.finish()
        callbacks.onDone(
          GenerationStats(
            promptTokenCount: Int64(promptTokens.count),
            cachedTokenCount: Int64(reusedPromptTokens),
            generatedTokenCount: Int64(generated),
            finishReason: .aborted,
            prefillMicroseconds: prefillMicroseconds,
            decodeMicroseconds: microsecondsSince(started),
            draftedTokenCount: Int64(draftedTotal),
            acceptedTokenCount: Int64(acceptedTotal)))
        return
      }
      decoded.append(current)
      decoded.append(contentsOf: drafted.prefix(accepted))

      draftedTotal += drafted.count
      acceptedTotal += accepted
      current = next
    }

    emitter.finish()
    let elapsed =
      Double(DispatchTime.now().uptimeNanoseconds - started.uptimeNanoseconds)
      / 1_000_000_000
    logNative(
      String(
        format:
          "speculative (mtp): %d tokens in %.2fs (%.1f tok/s), "
          + "draft acceptance %d/%d",
        generated, elapsed, elapsed > 0 ? Double(generated) / elapsed : 0,
        acceptedTotal, draftedTotal))
    finish(endReason)
  }

  /// Frees the underlying context and model. Not reusable afterwards.
  ///
  /// Any in-flight generation is cancelled first (it observes the flag
  /// within one decode step) and the free itself runs asynchronously on the
  /// session queue, so the caller — the platform thread — never blocks
  /// behind a running generation. The block holds `self` strongly, keeping
  /// the session alive until its pointers are freed.
  func dispose() {
    lock.lock()
    disposed = true
    cancelled = true
    lock.unlock()
    queue.async {
      if let mtmd = self.mtmd {
        mtmd_free(mtmd)
      }
      self.draft?.dispose()
      llama_free(self.context)
      llama_model_free(self.model)
    }
  }

  // MARK: - Prompt ingestion

  /// Decodes a tokenized text-only prompt into the main context, reusing the
  /// longest KV-cache prefix shared with `previous` (the tokens the cache
  /// held after the last generation). Returns the number of reused tokens;
  /// the position to continue sampling from is always `tokens.count`.
  private func decodeTextPrompt(
    _ tokens: [llama_token], previous: [llama_token], seq: llama_seq_id
  ) -> Result<Int, GenerationError> {
    if tokens.isEmpty {
      return .failure(GenerationError("Prompt tokenized to zero tokens"))
    }

    // Refuse a prompt that cannot fit (leaving room for at least one sampled
    // token). Without this guard an over-long prompt reaches llama_decode and
    // trips `GGML_ASSERT(n_tokens_all <= n_batch)`, which calls ggml_abort and
    // kills the whole process — a hard crash, not a recoverable error.
    let nCtx = Int(llama_n_ctx_seq(context))
    if tokens.count >= nCtx {
      return .failure(GenerationError(
        "Prompt is \(tokens.count) tokens but the context window is \(nCtx). "
          + "Shorten the input or load the model with a larger contextSize."))
    }

    let reused = reuseCachedPrefix(
      tokens, previous: previous, in: context, nSwa: nSwa, seq: seq)
    let suffix = Array(tokens[reused...])
    let decoded =
      seq == 0
      ? decodeChunked(suffix, into: context)
      : decodeChunkedInSequence(suffix, seq: seq, startPosition: reused)
    return decoded
      ? .success(reused)
      : .failure(GenerationError("Failed to decode prompt"))
  }

  /// Rewinds `target`'s KV cache to the longest usable common prefix of
  /// `previous` (the tokens the cache currently holds, possibly followed by
  /// stray speculation the last run never trimmed) and `tokens` (the new
  /// prompt), returning how many positions survive. 0 means the cache was
  /// cleared and the whole prompt must be decoded. At least one suffix token
  /// is always left for the caller to decode so sampling gets fresh logits,
  /// even when the prompts are identical (regeneration).
  private func reuseCachedPrefix(
    _ tokens: [llama_token], previous: [llama_token],
    in target: OpaquePointer, nSwa: Int32, seq: llama_seq_id = 0
  ) -> Int {
    let memory = llama_get_memory(target)
    var matched = 0
    let limit = min(tokens.count, previous.count)
    while matched < limit && tokens[matched] == previous[matched] {
      matched += 1
    }
    matched = min(matched, tokens.count - 1)
    if matched <= 0 {
      eraseSequence(seq, in: target)
      return 0
    }

    // Sliding-window attention layers prune keys older than their window;
    // decoding at a position whose window the cache has already dropped
    // would attend to missing keys and corrupt output. Mirrors llama-server's
    // pos_min check.
    if nSwa > 0,
      llama_memory_seq_pos_min(memory, seq) > max(0, Int32(matched) - nSwa)
    {
      logNative("prefix reuse skipped: SWA cache pruned past the rewind point")
      eraseSequence(seq, in: target)
      return 0
    }

    // Trim unconditionally — even when `matched == previous.count` the cache
    // can hold stray tokens past the ledger (a stop that fired mid
    // acceptance-walk skips that step's rollback trim).
    guard llama_memory_seq_rm(memory, seq, llama_pos(matched), -1) else {
      logNative("prefix reuse skipped: partial KV-cache removal unsupported")
      eraseSequence(seq, in: target)
      return 0
    }
    return matched
  }

  /// Feeds [tokens] into [target] in chunks of at most its n_batch tokens.
  ///
  /// A single oversized llama_decode batch trips
  /// `GGML_ASSERT(n_tokens_all <= n_batch)` (ggml_abort, a hard process
  /// crash); chunking keeps every batch within bounds even if n_batch is ever
  /// made smaller than n_ctx. llama_batch_get_one leaves pos = nullptr, so
  /// llama_decode auto-assigns sequential KV positions continuing across
  /// successive chunks.
  private func decodeChunked(
    _ tokens: [llama_token], into target: OpaquePointer
  ) -> Bool {
    var tokens = tokens
    let nBatch = max(Int(llama_n_batch(target)), 1)
    return tokens.withUnsafeMutableBufferPointer { buffer -> Bool in
      guard let base = buffer.baseAddress else { return false }
      var offset = 0
      while offset < buffer.count {
        let chunk = min(nBatch, buffer.count - offset)
        let batch = llama_batch_get_one(base + offset, Int32(chunk))
        if llama_decode(target, batch) != 0 { return false }
        offset += chunk
      }
      return true
    }
  }

  /// Decodes a single token into [target] at the next auto-assigned position,
  /// computing logits for it.
  private func decodeOne(_ token: llama_token, into target: OpaquePointer) -> Bool {
    var token = token
    return withUnsafeMutablePointer(to: &token) { ptr in
      llama_decode(target, llama_batch_get_one(ptr, 1)) == 0
    }
  }

  /// Feeds [tokens] into the main context's sequence [seq] at explicit
  /// positions starting at [startPosition], chunked to n_batch.
  ///
  /// `llama_batch_get_one` cannot express a sequence id (it defaults to
  /// sequence 0 with auto-assigned positions), so non-zero sequences build
  /// explicit batches. Mirroring get_one, logits are requested only for the
  /// last token of each chunk, respecting the load-time n_outputs_max cap.
  private func decodeChunkedInSequence(
    _ tokens: [llama_token], seq: llama_seq_id, startPosition: Int
  ) -> Bool {
    guard !tokens.isEmpty else { return true }
    let nBatch = max(Int(llama_n_batch(context)), 1)
    var batch = llama_batch_init(Int32(min(nBatch, tokens.count)), 0, 1)
    defer { llama_batch_free(batch) }
    var offset = 0
    while offset < tokens.count {
      let chunk = min(nBatch, tokens.count - offset)
      batch.n_tokens = Int32(chunk)
      for index in 0..<chunk {
        batch.token[index] = tokens[offset + index]
        batch.pos[index] = llama_pos(startPosition + offset + index)
        batch.n_seq_id[index] = 1
        batch.seq_id[index]![0] = seq
        batch.logits[index] = index == chunk - 1 ? 1 : 0
      }
      if llama_decode(context, batch) != 0 { return false }
      offset += chunk
    }
    return true
  }

  /// Decodes a single token into the main context's sequence [seq] at the
  /// explicit [position], computing logits for it.
  private func decodeOneInSequence(
    _ token: llama_token, seq: llama_seq_id, position: Int
  ) -> Bool {
    var batch = llama_batch_init(1, 0, 1)
    defer { llama_batch_free(batch) }
    batch.n_tokens = 1
    batch.token[0] = token
    batch.pos[0] = llama_pos(position)
    batch.n_seq_id[0] = 1
    batch.seq_id[0]![0] = seq
    batch.logits[0] = 1
    return llama_decode(context, batch) == 0
  }

  /// Fills a reusable `llama_batch_init` batch with [tokens] at explicit
  /// sequential positions starting at [startPosition], requesting logits at
  /// every index (unlike `llama_batch_get_one`, which only computes them for
  /// the last token). The batch must have been allocated with capacity for
  /// `tokens.count`.
  private func fill(
    batch: inout llama_batch, tokens: [llama_token], startPosition: Int
  ) {
    batch.n_tokens = Int32(tokens.count)
    for (index, token) in tokens.enumerated() {
      batch.token[index] = token
      batch.pos[index] = llama_pos(startPosition + index)
      batch.n_seq_id[index] = 1
      batch.seq_id[index]![0] = 0
      batch.logits[index] = 1
    }
  }

  /// Runs the prompt (containing one media marker per image) plus the decoded
  /// images through mtmd, evaluating the interleaved text/image chunks into the
  /// context. Returns the new position to continue sampling from.
  private func evalMultimodalPrompt(
    mtmd: OpaquePointer, prompt: String, images: [FlutterStandardTypedData],
    seq: llama_seq_id = 0
  ) -> Result<Int, GenerationError> {
    var bitmaps: [OpaquePointer?] = []
    defer {
      for bitmap in bitmaps where bitmap != nil {
        mtmd_bitmap_free(bitmap)
      }
    }
    for image in images {
      let data = image.data
      let bitmap = data.withUnsafeBytes {
        (raw: UnsafeRawBufferPointer) -> OpaquePointer? in
        guard let base = raw.bindMemory(to: UInt8.self).baseAddress else {
          return nil
        }
        // As of llama.cpp b9585 this returns a wrapper struct (carrying an
        // optional video context we don't use) and takes a `placeholder` flag;
        // `false` decodes a real bitmap rather than a token-counting stub.
        return mtmd_helper_bitmap_init_from_buf(mtmd, base, data.count, false)
          .bitmap
      }
      guard let bitmap else {
        return .failure(GenerationError("Failed to decode media bytes"))
      }
      bitmaps.append(bitmap)
    }

    // mtmd auto-detects audio vs image from each blob's magic bytes. If the
    // caller sent audio but this model's projector has no audio encoder, fail
    // with a clear message rather than an opaque tokenize/encode error.
    if bitmaps.contains(where: { $0 != nil && mtmd_bitmap_is_audio($0) }),
      !mtmd_support_audio(mtmd)
    {
      return .failure(
        GenerationError(
          "This model has no audio projector; cannot accept audio input"))
    }

    guard let chunks = mtmd_input_chunks_init() else {
      return .failure(GenerationError("Failed to allocate mtmd input chunks"))
    }
    defer { mtmd_input_chunks_free(chunks) }

    let tokenizeStatus = prompt.withCString { cPrompt -> Int32 in
      var text = mtmd_input_text()
      text.text = cPrompt
      // Since llama.cpp b100xx the tokenizer reads exactly `text_len` bytes
      // (the field was added to mtmd_input_text); leaving it at the struct's
      // zero-init makes mtmd see an empty prompt and fail every image turn
      // with "media markers (0) != bitmaps (N)" surfaced as status 2.
      text.text_len = strlen(cPrompt)
      text.add_special = true
      text.parse_special = true
      return bitmaps.withUnsafeMutableBufferPointer { buffer -> Int32 in
        // The importer maps `const mtmd_bitmap **` to a *mutable* pointer
        // (the pointee's constness is lost on the opaque type), so a mutable
        // buffer is required here even though mtmd does not mutate it.
        mtmd_tokenize(mtmd, chunks, &text, buffer.baseAddress, buffer.count)
      }
    }
    if tokenizeStatus != 0 {
      return .failure(
        GenerationError("mtmd_tokenize failed (status \(tokenizeStatus))"))
    }

    var newPosition: llama_pos = 0
    let evalStatus = mtmd_helper_eval_chunks(
      mtmd, context, chunks,
      0,  // n_past
      seq,
      Int32(llama_n_batch(context)),
      true,  // logits_last: compute logits for the final token so we can sample
      &newPosition)
    if evalStatus != 0 {
      return .failure(
        GenerationError("mtmd_helper_eval_chunks failed (status \(evalStatus))"))
    }
    return .success(Int(newPosition))
  }

  // MARK: - llama.cpp helpers

  /// Builds the sampler chain top-k → top-p → temperature → dist, omitting
  /// the stages the request leaves unset. Temperature `<= 0` is greedy and
  /// ignores the other stages. A nil seed uses llama.cpp's default
  /// (0xFFFFFFFF), which draws a random seed per chain.
  private func makeSampler(request: GenerationRequest) -> UnsafeMutablePointer<llama_sampler> {
    let sampler = llama_sampler_chain_init(llama_sampler_chain_default_params())!
    if request.temperature <= 0 {
      llama_sampler_chain_add(sampler, llama_sampler_init_greedy())
      return sampler
    }
    if let topK = request.topK, topK > 0 {
      llama_sampler_chain_add(
        sampler, llama_sampler_init_top_k(Int32(truncatingIfNeeded: topK)))
    }
    if let topP = request.topP, topP < 1.0 {
      llama_sampler_chain_add(sampler, llama_sampler_init_top_p(Float(topP), 1))
    }
    llama_sampler_chain_add(
      sampler, llama_sampler_init_temp(Float(request.temperature)))
    let seed = request.seed.map { UInt32(truncatingIfNeeded: $0) } ?? 0xFFFF_FFFF
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(seed))
    return sampler
  }

  private func tokenize(_ text: String, addBos: Bool) -> [llama_token] {
    let byteCount = Int32(text.utf8.count)
    let capacity = Int(byteCount) + (addBos ? 1 : 0) + 1
    var tokens = [llama_token](repeating: 0, count: capacity)
    let count = text.withCString { cText in
      llama_tokenize(
        vocab, cText, byteCount, &tokens, Int32(capacity), addBos, true)
    }
    guard count > 0 else { return [] }
    return Array(tokens.prefix(Int(count)))
  }

  private func pieceBytes(for token: llama_token) -> [UInt8] {
    var buffer = [CChar](repeating: 0, count: 64)
    var count = llama_token_to_piece(
      vocab, token, &buffer, Int32(buffer.count), 0, true)
    if count < 0 {
      buffer = [CChar](repeating: 0, count: Int(-count))
      count = llama_token_to_piece(
        vocab, token, &buffer, Int32(buffer.count), 0, true)
    }
    guard count > 0 else { return [] }
    return buffer.prefix(Int(count)).map { UInt8(bitPattern: $0) }
  }
}

/// A loaded drafter model/context pair backing speculative decoding.
///
/// Two kinds of drafter are supported. A *standalone* drafter is any GGUF
/// whose vocabulary is interchangeable with the main model's (which `load`
/// verifies); it runs its own context in lockstep with the main one. An
/// *MTP* drafter (multi-token-prediction head, `*-assistant` architectures
/// like `gemma4-assistant`) is not a standalone LM: its context is created
/// with `ctx_other` pointing at the main context so it can tap the target's
/// memory and hidden states, and it is driven by the dedicated MTP decode
/// loop. All pointers are owned here and freed in `dispose()`.
private struct DraftModel {
  let model: OpaquePointer
  let context: OpaquePointer
  let vocab: OpaquePointer
  let maxDraftTokens: Int
  /// Sliding-window size of the drafter's SWA layers (0 = none); see
  /// `LlamaSession.nSwa`.
  let nSwa: Int32
  /// Whether this drafter is an MTP head driven through a shared-memory
  /// context (`runSpeculativeMtp`) rather than the standalone loop.
  let isMtp: Bool
  /// Width of the hidden-state rows an MTP drafter consumes; equals the main
  /// model's embedding size (0 for standalone drafters, which never use it).
  let nEmbd: Int

  /// Loads the drafter described by [options], or returns nil (logging the
  /// reason) on failure.
  ///
  /// The drafter's context mirrors the main model's [contextSize] so the two
  /// can stay position-aligned for a whole generation. For standalone
  /// drafters vocabulary compatibility is checked via token count plus
  /// BOS/EOS ids — the practical invariants speculation relies on, since
  /// draft proposals are decoded by both models. MTP drafters share the
  /// target's head and memory instead, so the check there is that the
  /// drafter's input row width matches the target's hidden size (mirroring
  /// upstream's assert in common/speculative.cpp).
  static func load(
    options: DraftModelOptions, contextSize: Int,
    mainModel: OpaquePointer, mainContext: OpaquePointer
  ) -> DraftModel? {
    var modelParams = llama_model_default_params()
    modelParams.n_gpu_layers = Int32(options.gpuLayers)
    let model = loadModelPinningUnsupportedTensorsToCpu(
      path: options.modelPath, params: modelParams)
    guard let model else {
      logNative("drafter failed to load: \(options.modelPath)")
      return nil
    }

    let arch = metaString(model, key: "general.architecture") ?? ""
    let isMtp = arch.hasSuffix("-assistant")

    let vocab = llama_model_get_vocab(model)!
    if isMtp {
      if llama_model_n_embd_out(model) != llama_model_n_embd(mainModel) {
        logNative(
          "MTP drafter row width \(llama_model_n_embd_out(model)) does not "
            + "match the main model's hidden size "
            + "\(llama_model_n_embd(mainModel)): " + options.modelPath)
        llama_model_free(model)
        return nil
      }
    } else {
      let mainVocab = llama_model_get_vocab(mainModel)!
      if llama_vocab_n_tokens(vocab) != llama_vocab_n_tokens(mainVocab)
        || llama_vocab_bos(vocab) != llama_vocab_bos(mainVocab)
        || llama_vocab_eos(vocab) != llama_vocab_eos(mainVocab)
      {
        logNative(
          "drafter vocabulary is incompatible with the main model: "
            + options.modelPath)
        llama_model_free(model)
        return nil
      }
    }

    var ctxParams = llama_context_default_params()
    ctxParams.n_ctx = UInt32(contextSize)
    if isMtp {
      // An MTP drafter never sees the prompt (it shares the target's memory)
      // and only ever decodes single-token batches; verification runs on the
      // target. Reserve-time compute buffers are sized by n_ubatch ×
      // n_outputs_max, so target-sized batches here reserved a ~2 GB logits
      // buffer (262k vocab) for a context whose real batches are one token.
      ctxParams.n_batch = 64
      ctxParams.n_ubatch = 64
      ctxParams.n_outputs_max = 1
    } else {
      // Standalone drafters ingest the full prompt via decodeChunked, so
      // they keep target-sized batches.
      ctxParams.n_batch = UInt32(contextSize)
      ctxParams.n_ubatch = UInt32(min(contextSize, 2048))
    }
    let threads = performanceCoreCount()
    ctxParams.n_threads = threads
    ctxParams.n_threads_batch = threads
    if isMtp {
      // The MTP context taps the target's memory and hidden states; without
      // ctx_other its init fails ("requires ctx_other to be set"). n_rs_seq=0
      // mirrors llama-server's MTP draft-context setup. The flash-attention
      // setting must MATCH the target context's (upstream drives both from
      // the same flag): the drafter attends over the target's shared KV
      // cache, and mismatched FA settings read it with the wrong layout.
      ctxParams.ctx_type = LLAMA_CONTEXT_TYPE_MTP
      ctxParams.ctx_other = mainContext
      ctxParams.n_rs_seq = 0
    }
    guard let context = llama_init_from_model(model, ctxParams) else {
      logNative("drafter context init failed: \(options.modelPath)")
      llama_model_free(model)
      return nil
    }
    if isMtp {
      // The drafter outputs its own nextn rows, which seed the next draft
      // step; masked sizes the buffer by logits-flagged rows only.
      llama_ext_set_embeddings_nextn(
        UnsafeMutableRawPointer(context), true, true)
    }

    // Cap the per-step draft length defensively: it sizes a llama_batch and
    // anything beyond ~24 yields no practical speedup.
    let maxDraftTokens = Int(min(max(options.maxDraftTokens, 0), 64))
    return DraftModel(
      model: model, context: context, vocab: vocab,
      maxDraftTokens: maxDraftTokens, nSwa: llama_model_n_swa(model),
      isMtp: isMtp, nEmbd: isMtp ? Int(llama_model_n_embd(mainModel)) : 0)
  }

  /// Reads a string metadata value from [model], or nil when absent.
  private static func metaString(
    _ model: OpaquePointer, key: String
  ) -> String? {
    var buffer = [CChar](repeating: 0, count: 256)
    let length = llama_model_meta_val_str(model, key, &buffer, buffer.count)
    return length >= 0 ? String(cString: buffer) : nil
  }

  func dispose() {
    llama_free(context)
    llama_model_free(model)
  }
}

/// Streams one token at a time through UTF-8 reassembly and stop-sequence
/// matching, forwarding emitted text to a callback.
///
/// Shared by the plain and speculative decode loops so both have identical
/// emission semantics.
private struct TokenEmitter {
  private var decoder = Utf8Decoder()
  private var stopMatcher: StopMatcher
  private let onToken: (String) -> Void

  init(stops: [String], onToken: @escaping (String) -> Void) {
    self.stopMatcher = StopMatcher(stops: stops)
    self.onToken = onToken
  }

  /// Pushes one token's raw bytes; returns true when a stop sequence fired
  /// (the stop itself is stripped, and generation should end).
  mutating func push(_ bytes: [UInt8]) -> Bool {
    guard let piece = decoder.append(bytes) else { return false }
    let (emit, stopped) = stopMatcher.push(piece)
    if !emit.isEmpty {
      onToken(emit)
    }
    return stopped
  }

  /// Flushes incomplete UTF-8 bytes and held-back stop-candidate text. Call
  /// once when generation ends without a stop match.
  mutating func finish() {
    if let flushed = decoder.flush() {
      let (emit, stopped) = stopMatcher.push(flushed)
      if !emit.isEmpty {
        onToken(emit)
      }
      if stopped { return }
    }
    let tail = stopMatcher.flush()
    if !tail.isEmpty {
      onToken(tail)
    }
  }
}

/// Accumulates raw token bytes and emits only complete UTF-8 sequences.
///
/// A single llama token can be a partial multi-byte character, so bytes are
/// buffered until they decode cleanly.
private struct Utf8Decoder {
  private var pending: [UInt8] = []

  mutating func append(_ bytes: [UInt8]) -> String? {
    pending.append(contentsOf: bytes)
    if let text = String(bytes: pending, encoding: .utf8) {
      pending.removeAll(keepingCapacity: true)
      return text.isEmpty ? nil : text
    }
    return nil
  }

  mutating func flush() -> String? {
    guard !pending.isEmpty else { return nil }
    let text = String(decoding: pending, as: UTF8.self)
    pending.removeAll(keepingCapacity: true)
    return text.isEmpty ? nil : text
  }
}

/// Streaming stop-sequence detector.
///
/// Decoded text is pushed in as it arrives. Once any stop string appears, the
/// text up to the match is returned and `stopped` is set; everything from the
/// match onward is discarded. Because a stop sequence can straddle two pushes,
/// the trailing `maxStopLength - 1` characters are held back until enough
/// follows to rule out a match, so a partial stop is never leaked.
private struct StopMatcher {
  private let stops: [String]
  private let holdback: Int
  private var buffer: String = ""

  init(stops: [String]) {
    self.stops = stops.filter { !$0.isEmpty }
    let maxLen = self.stops.map(\.count).max() ?? 0
    self.holdback = max(maxLen - 1, 0)
  }

  /// Appends [text] and returns the portion safe to emit, plus whether a stop
  /// sequence was hit. Stops are matched on grapheme clusters; since stop
  /// strings are ASCII control markers, holdback boundaries never split a
  /// character.
  mutating func push(_ text: String) -> (emit: String, stopped: Bool) {
    if stops.isEmpty {
      return (text, false)
    }
    buffer += text

    var earliest: Range<String.Index>?
    for stop in stops {
      if let range = buffer.range(of: stop),
        earliest == nil || range.lowerBound < earliest!.lowerBound
      {
        earliest = range
      }
    }
    if let range = earliest {
      let emit = String(buffer[..<range.lowerBound])
      buffer = ""
      return (emit, true)
    }

    if buffer.count <= holdback {
      return ("", false)
    }
    let splitIndex = buffer.index(buffer.endIndex, offsetBy: -holdback)
    let emit = String(buffer[..<splitIndex])
    buffer = String(buffer[splitIndex...])
    return (emit, false)
  }

  /// Returns any remaining held-back text. Call once generation ends without a
  /// stop match.
  mutating func flush() -> String {
    let out = buffer
    buffer = ""
    return out
  }
}
