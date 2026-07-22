#if os(iOS)
  import Flutter
#elseif os(macOS)
  import FlutterMacOS
#endif
import Foundation
import llama

/// Holds the active token `EventChannel` sink and forwards `TokenEvent`s to
/// Dart on the main thread (Flutter sinks must be invoked there).
final class LlamaTokenStreamHandler: StreamTokensStreamHandler {
  private var sink: PigeonEventSink<TokenEvent>?
  private let lock = NSLock()

  override func onListen(withArguments arguments: Any?, sink: PigeonEventSink<TokenEvent>) {
    lock.lock()
    self.sink = sink
    lock.unlock()
  }

  override func onCancel(withArguments arguments: Any?) {
    lock.lock()
    sink = nil
    lock.unlock()
  }

  func send(_ event: TokenEvent) {
    lock.lock()
    let target = sink
    lock.unlock()
    guard let target else { return }
    DispatchQueue.main.async { target.success(event) }
  }
}

public final class LlamaFlutterPlugin: NSObject, FlutterPlugin, LlamaHostApi {
  private let streamHandler = LlamaTokenStreamHandler()
  private var sessions: [Int64: LlamaSession] = [:]
  private var nextSessionId: Int64 = 1
  private let lock = NSLock()
  private let loadQueue = DispatchQueue(
    label: "dev.llama_flutter.load", qos: .userInitiated)

  public static func register(with registrar: FlutterPluginRegistrar) {
    #if os(iOS)
      let messenger = registrar.messenger()
    #elseif os(macOS)
      let messenger = registrar.messenger
    #endif
    let instance = LlamaFlutterPlugin()
    llama_backend_init()
    LlamaHostApiSetup.setUp(binaryMessenger: messenger, api: instance)
    StreamTokensStreamHandler.register(
      with: messenger, streamHandler: instance.streamHandler)
  }

  // MARK: - LlamaHostApi

  func loadModel(
    request: ModelLoadRequest, completion: @escaping (Result<Int64, Error>) -> Void
  ) {
    loadQueue.async { [weak self] in
      guard let self else { return }
      let sessionId = self.reserveSessionId()
      guard let session = LlamaSession.load(request: request, sessionId: sessionId) else {
        DispatchQueue.main.async {
          completion(
            .failure(
              LlamaError(
                code: "load_failed",
                message: "Could not load model at \(request.modelPath)",
                details: nil)))
        }
        return
      }
      self.lock.lock()
      self.sessions[sessionId] = session
      self.lock.unlock()
      DispatchQueue.main.async { completion(.success(sessionId)) }
    }
  }

  func startGeneration(request: GenerationRequest) throws {
    guard let session = session(for: request.sessionId) else {
      streamHandler.send(
        TokenEvent(
          sessionId: request.sessionId, done: true,
          error: "Unknown session \(request.sessionId)"))
      return
    }
    let sessionId = request.sessionId
    session.generate(
      request: request,
      callbacks: GenerationCallbacks(
        onToken: { [weak self] text in
          self?.streamHandler.send(
            TokenEvent(sessionId: sessionId, text: text, done: false))
        },
        onDone: { [weak self] stats in
          self?.streamHandler.send(
            TokenEvent(sessionId: sessionId, done: true, stats: stats))
        },
        onError: { [weak self] message in
          self?.streamHandler.send(
            TokenEvent(sessionId: sessionId, done: true, error: message))
        }))
  }

  func cancelGeneration(sessionId: Int64) throws {
    session(for: sessionId)?.cancel()
  }

  func saveSessionState(
    sessionId: Int64, path: String, sequenceId: Int64?,
    completion: @escaping (Result<Int64, Error>) -> Void
  ) {
    let seq = llama_seq_id(truncatingIfNeeded: sequenceId ?? 0)
    stateOp(sessionId: sessionId, completion: completion) { session, finish in
      session.saveState(toPath: path, sequenceId: seq, completion: finish)
    }
  }

  func loadSessionState(
    sessionId: Int64, path: String, sequenceId: Int64?,
    completion: @escaping (Result<Int64, Error>) -> Void
  ) {
    let seq = llama_seq_id(truncatingIfNeeded: sequenceId ?? 0)
    stateOp(sessionId: sessionId, completion: completion) { session, finish in
      session.loadState(fromPath: path, sequenceId: seq, completion: finish)
    }
  }

  func getSessionStateSize(
    sessionId: Int64, sequenceId: Int64?,
    completion: @escaping (Result<Int64, Error>) -> Void
  ) {
    let seq = llama_seq_id(truncatingIfNeeded: sequenceId ?? 0)
    stateOp(sessionId: sessionId, completion: completion) { session, finish in
      session.stateSize(sequenceId: seq, completion: finish)
    }
  }

  func stashSessionState(
    sessionId: Int64, sequenceId: Int64, key: String,
    completion: @escaping (Result<StashResult, Error>) -> Void
  ) {
    guard let session = session(for: sessionId) else {
      completion(.failure(unknownSession(sessionId)))
      return
    }
    session.stashState(
      sequenceId: llama_seq_id(truncatingIfNeeded: sequenceId), key: key
    ) { tokens, bytes, error in
      DispatchQueue.main.async {
        if let error {
          completion(
            .failure(
              LlamaError(code: "state_failed", message: error, details: nil)))
        } else {
          completion(.success(StashResult(tokens: tokens, bytes: bytes)))
        }
      }
    }
  }

  func restoreStashedState(
    sessionId: Int64, sequenceId: Int64, key: String,
    completion: @escaping (Result<Int64, Error>) -> Void
  ) {
    let seq = llama_seq_id(truncatingIfNeeded: sequenceId)
    stateOp(sessionId: sessionId, completion: completion) { session, finish in
      session.restoreStashedState(sequenceId: seq, key: key, completion: finish)
    }
  }

  func dropStashedState(
    sessionId: Int64, key: String,
    completion: @escaping (Result<Int64, Error>) -> Void
  ) {
    stateOp(sessionId: sessionId, completion: completion) { session, finish in
      session.dropStashedState(key: key, completion: finish)
    }
  }

  func clearSequence(
    sessionId: Int64, sequenceId: Int64,
    completion: @escaping (Result<Void, Error>) -> Void
  ) {
    guard let session = session(for: sessionId) else {
      completion(.failure(unknownSession(sessionId)))
      return
    }
    session.clearSequence(
      llama_seq_id(truncatingIfNeeded: sequenceId)
    ) { error in
      DispatchQueue.main.async {
        if let error {
          completion(
            .failure(
              LlamaError(code: "state_failed", message: error, details: nil)))
        } else {
          completion(.success(()))
        }
      }
    }
  }

  func setImageTokenBudget(
    sessionId: Int64, imageTokenBudget: Int64?,
    completion: @escaping (Result<Void, Error>) -> Void
  ) {
    guard let session = session(for: sessionId) else {
      completion(.failure(unknownSession(sessionId)))
      return
    }
    session.setImageTokenBudget(imageTokenBudget) { error in
      DispatchQueue.main.async {
        if let error {
          completion(
            .failure(
              LlamaError(
                code: "image_budget_failed", message: error, details: nil)))
        } else {
          completion(.success(()))
        }
      }
    }
  }

  private func unknownSession(_ sessionId: Int64) -> LlamaError {
    LlamaError(
      code: "unknown_session",
      message: "Unknown session \(sessionId)", details: nil)
  }

  /// Shared session-lookup + threading adapter for the state calls: the
  /// session completes on its own serial queue, Pigeon completions must run
  /// on the main thread.
  private func stateOp(
    sessionId: Int64,
    completion: @escaping (Result<Int64, Error>) -> Void,
    run: (LlamaSession, @escaping (Int64, String?) -> Void) -> Void
  ) {
    guard let session = session(for: sessionId) else {
      completion(
        .failure(
          LlamaError(
            code: "unknown_session",
            message: "Unknown session \(sessionId)", details: nil)))
      return
    }
    run(session) { count, error in
      DispatchQueue.main.async {
        if let error {
          completion(
            .failure(
              LlamaError(code: "state_failed", message: error, details: nil)))
        } else {
          completion(.success(count))
        }
      }
    }
  }

  func disposeSession(sessionId: Int64) throws {
    lock.lock()
    let session = sessions.removeValue(forKey: sessionId)
    lock.unlock()
    session?.dispose()
  }

  // MARK: - Helpers

  private func reserveSessionId() -> Int64 {
    lock.lock()
    defer { lock.unlock() }
    let id = nextSessionId
    nextSessionId += 1
    return id
  }

  private func session(for id: Int64) -> LlamaSession? {
    lock.lock()
    defer { lock.unlock() }
    return sessions[id]
  }
}
