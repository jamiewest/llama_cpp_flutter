/// A Microsoft.Extensions.AI [ChatClient] backed by on-device llama.cpp.
library;

import 'dart:async';
import 'dart:typed_data';

import 'package:agents/agents.dart'
    show AgentRequestMessageSourceType, ChatMessageExtensions;
import 'package:extensions/ai.dart';
import 'package:extensions/logging.dart';
import 'package:extensions/system.dart';

import '../../diagnostics/prompt_inspector.dart';
import '../../models/model_spec.dart';
import '../../runtime/llama_runtime_api.dart';
import '../chat_format.dart';
import '../image_tiling.dart';

/// Resolves the loaded [LlamaSession] this client generates against.
///
/// The session is owned and loaded elsewhere (the model-loading hosted
/// service); the returned future completes once the model is ready.
typedef SessionProvider = Future<LlamaSession> Function();

/// Returns [messages] with [instructions] materialized as the leading system
/// message.
///
/// M.E.AI carries system instructions out-of-band in
/// [ChatOptions.instructions]; converting them into an in-band system message
/// is the chat client's job (the `extensions` OpenAI client does the same).
/// [ChatFormat]s only render a system turn from an actual system-role
/// message, so without this step the harness instructions and every
/// `AIContextProvider`'s contributed instructions are silently dropped from
/// the prompt.
///
/// If the conversation already starts with a system message the two are
/// merged into one (instructions first): Gemma's wire format has a single
/// system turn, and a second system-role message mid-prompt would render as
/// its own bogus turn.
List<ChatMessage> messagesWithInstructions(
  Iterable<ChatMessage> messages,
  String? instructions,
) {
  final prepared = messagesWithRuntimeContext(messages, instructions);
  final trimmed = prepared.instructions?.trim();
  final list = prepared.messages.toList();
  if (trimmed == null || trimmed.isEmpty) return list;

  ChatMessage system(String text) => ChatMessage(
    role: ChatRole.system,
    contents: <AIContent>[TextContent(text)],
  );

  if (list.isNotEmpty && list.first.role == ChatRole.system) {
    return <ChatMessage>[
      system('$trimmed\n\n${list.first.text}'),
      ...list.skip(1),
    ];
  }
  return <ChatMessage>[system(trimmed), ...list];
}

/// Repositions text-only AI-context-provider messages as one runtime-context
/// turn just before the latest external user message.
///
/// Harness context providers sometimes contribute transient status messages
/// using a `user` role, for example the todo provider's current todo list.
/// Rendering those messages as ordinary trailing user turns changes the
/// perceived latest user request for small local chat models.
///
/// They must not be merged into the system instructions either: that text
/// sits at the very top of the rendered prompt, so any change to it (a todo
/// update, per-turn memory recall) invalidates the entire llama.cpp KV-cache
/// prefix and forces a full re-prefill of the whole conversation every turn.
/// Placing the volatile context after the stable history keeps the prefix
/// reusable while the model still reads the context right before the actual
/// request.
({Iterable<ChatMessage> messages, String? instructions})
messagesWithRuntimeContext(
  Iterable<ChatMessage> messages,
  String? instructions,
) {
  final runtimeContext = <String>[];
  final retained = <ChatMessage>[];

  for (final message in messages) {
    if (_isTextOnlyProviderMessage(message)) {
      final text = message.text.trim();
      if (text.isNotEmpty) {
        final sourceId = message.getAgentRequestMessageSourceId();
        runtimeContext.add(
          sourceId == null || sourceId.isEmpty ? text : '[$sourceId]\n$text',
        );
      }
    } else {
      retained.add(message);
    }
  }

  if (runtimeContext.isEmpty) {
    return (messages: retained, instructions: instructions);
  }

  final contextMessage = ChatMessage(
    role: ChatRole.user,
    contents: <AIContent>[
      TextContent('Runtime context:\n${runtimeContext.join('\n\n')}'),
    ],
  );

  var insertAt = retained.length;
  for (var i = retained.length - 1; i >= 0; i--) {
    if (retained[i].role == ChatRole.user) {
      insertAt = i;
      break;
    }
  }
  final result = [...retained]..insert(insertAt, contextMessage);
  return (messages: result, instructions: instructions);
}

/// Builds the engine-neutral structured view of [messages] passed to
/// [LlamaSession.generate] alongside the rendered prompt for image turns.
///
/// Roles map to the wire names wllama's chat-completion API accepts
/// (`system`/`user`/`assistant`); tool-role results fold into user-role text
/// turns. Media is extracted with the same rule the [ChatFormat] templates
/// use — [DataContent] with non-null bytes — split by kind so the message-level
/// multimodal path can label each content part (`image` vs `audio`).
List<LlamaChatTurn> chatTurnsFromMessages(Iterable<ChatMessage> messages) =>
    <LlamaChatTurn>[
      for (final message in messages)
        LlamaChatTurn(
          role: message.role == ChatRole.system
              ? 'system'
              : message.role == ChatRole.assistant
              ? 'assistant'
              : 'user',
          text: message.text.trim(),
          images: <Uint8List>[
            for (final data in message.contents.whereType<DataContent>())
              if (data.data != null && data.hasTopLevelMediaType('image'))
                data.data!,
          ],
          audio: <Uint8List>[
            for (final data in message.contents.whereType<DataContent>())
              if (data.data != null && data.hasTopLevelMediaType('audio'))
                data.data!,
          ],
        ),
    ];

/// Whether any message carries audio-typed [DataContent] with bytes.
bool _hasAudioContent(Iterable<ChatMessage> messages) => messages.any(
  (message) => message.contents.whereType<DataContent>().any(
    (data) => data.data != null && data.hasTopLevelMediaType('audio'),
  ),
);

bool _isTextOnlyProviderMessage(ChatMessage message) {
  if (message.getAgentRequestMessageSourceType() !=
      AgentRequestMessageSourceType.aiContextProvider) {
    return false;
  }
  if (message.contents.isEmpty) return false;
  return message.contents.every((content) => content is TextContent);
}

/// Bridges the M.E.AI chat abstractions to a model running through
/// `LlamaFlutter`.
///
/// This is the **inner** client: wrap it with `FunctionInvokingChatClient` so
/// tool calls it surfaces (as [FunctionCallContent]) are executed and fed back.
/// All model-family specifics — prompt rendering and the prose/reasoning/
/// tool-call stream split — live in the injected [ChatFormat]. The model is
/// loaded by a hosted service and supplied through [sessionProvider]; this
/// client does not own its lifecycle.
class LlamaChatClient extends ChatClient {
  LlamaChatClient({
    required this.sessionProvider,
    required this.format,
    required this.contextSize,
    this.formatResolver,
    this.sampling = const SamplingDefaults(),
    this.imageTiling,
    this.inspector,
    this.isThinkingEnabled,
    LoggerFactory? loggerFactory,
  }) : _logger = (loggerFactory ?? NullLoggerFactory.instance).createLogger(
         'LlamaChatClient',
       );

  final Logger _logger;

  /// Resolves the ready session; the caller owns its lifecycle.
  final SessionProvider sessionProvider;

  /// The model family's prompt rendering and output decoding, used when
  /// [formatResolver] is absent or returns null.
  final ChatFormat format;

  /// Optional late-bound format, read per request **after** the session
  /// resolves. This lets a host defer the choice until the model file is
  /// actually on disk — e.g. picking the format from the GGUF's embedded
  /// chat template during load — instead of guessing from the file name
  /// at construction time. Returning null falls back to [format].
  final ChatFormat? Function()? formatResolver;

  /// The model's context window in tokens, recorded on each [PromptSnapshot]
  /// so the UI can gauge how full the context is.
  final int contextSize;

  /// Generation defaults used when the per-request [ChatOptions] doesn't
  /// override them.
  final SamplingDefaults sampling;

  /// When set, each attached image is split into overlapping crops before
  /// prompt rendering (see [ImageTiling]), boosting vision fidelity for
  /// OCR and dense documents at the cost of one prefill per crop. Null
  /// passes images through whole.
  final ImageTiling? imageTiling;

  /// Optional sink that records each rendered prompt and its resolved sampling
  /// config so the UI can show exactly what was sent to the model.
  final PromptInspector? inspector;

  /// Reads whether to request the family's reasoning channel, evaluated per
  /// request so a runtime toggle takes effect on the next turn. Null means
  /// thinking is always off. The result is still gated on
  /// [ChatFormat.supportsThinking].
  final bool Function()? isThinkingEnabled;

  @override
  Stream<ChatResponseUpdate> getStreamingResponse({
    required Iterable<ChatMessage> messages,
    ChatOptions? options,
    CancellationToken? cancellationToken,
  }) async* {
    final session = await sessionProvider();
    // Resolved after the session so a formatResolver populated during model
    // load (e.g. from the GGUF's embedded chat template) applies to the very
    // first request.
    final activeFormat = formatResolver?.call() ?? format;
    final tools = (options?.tools ?? const <AITool>[])
        .whereType<AIFunctionDeclaration>();
    final thinking =
        (isThinkingEnabled?.call() ?? false) && activeFormat.supportsThinking;
    // Tiling happens before rendering so each crop gets its own media
    // marker, and before chatTurnsFromMessages so the message-level (web)
    // path sees the same crops. Deterministic per image, so re-tiling the
    // history each turn leaves the rendered prefix stable for KV reuse.
    var prepared = messagesWithInstructions(messages, options?.instructions);
    final tiling = imageTiling;
    if (tiling != null) {
      prepared = await tiledMessages(prepared, tiling);
    }
    final prompt = activeFormat.render(
      prepared,
      tools: tools,
      enableThinking: thinking,
    );

    // A model whose format did not emit any media marker for audio the caller
    // attached cannot "hear" it: the audio would be silently dropped and the
    // model asked to transcribe nothing. Fail loudly instead. (Gemma 4 does
    // collect audio, so this fires only for text/vision-only families; the
    // absent-audio-projector case surfaces as a clear runtime error deeper in.)
    if (prompt.media.isEmpty && _hasAudioContent(prepared)) {
      throw UnsupportedError(
        'This local model cannot accept audio input. Load an audio-capable '
        'model (Gemma 4 with an audio projector) to transcribe speech.',
      );
    }

    final maxTokens = options?.maxOutputTokens ?? sampling.maxTokens;
    final temperature = options?.temperature ?? sampling.temperature;
    final topK = options?.topK ?? sampling.topK;
    final topP = options?.topP ?? sampling.topP;
    final seed = options?.seed ?? sampling.seed;

    inspector?.record(
      PromptSnapshot(
        text: prompt.text,
        stopSequences: prompt.stopSequences,
        maxTokens: maxTokens,
        temperature: temperature,
        topK: topK,
        topP: topP,
        seed: seed,
        imageCount: prompt.media.length,
        contextSize: contextSize,
        capturedAt: DateTime.now(),
      ),
    );

    if (cancellationToken?.isCancellationRequested ?? false) return;

    _logger.logDebug(
      'Generating with ${activeFormat.runtimeType}: '
      '${prompt.text.length} prompt chars, ${prompt.media.length} media, '
      '${tools.length} tools, maxTokens: $maxTokens, thinking: $thinking.',
    );

    LlamaGenerationStats? stats;
    var tokens = session.generate(
      prompt.text,
      maxTokens: maxTokens,
      temperature: temperature,
      topK: topK,
      topP: topP,
      seed: seed,
      stopSequences: prompt.stopSequences,
      media: prompt.media.isEmpty ? null : prompt.media,
      turns: prompt.media.isEmpty ? null : chatTurnsFromMessages(prepared),
      onStats: (reported) => stats = reported,
    );
    // Cancellation is forwarded to the engine (which stops decoding within a
    // step and ends the stream) rather than only filtered here: takeWhile
    // alone would leave the engine generating until its next token arrived.
    CancellationTokenRegistration? cancelRegistration;
    if (cancellationToken != null) {
      cancelRegistration = cancellationToken.register(
        (_) => unawaited(session.cancel()),
      );
      tokens = tokens.takeWhile(
        (_) => !cancellationToken.isCancellationRequested,
      );
    }

    try {
      yield* activeFormat.decode(tokens);
    } finally {
      cancelRegistration?.dispose();
    }

    // The runtime reports token accounting on the done event, after the
    // token stream has drained; surface it the way cloud clients do, as a
    // trailing usage-only update.
    final reported = stats;
    if (reported != null) {
      _logger.logDebug(
        'Generation finished: ${reported.promptTokenCount} in '
        '(${reported.cachedTokenCount} cached), '
        '${reported.generatedTokenCount} out, '
        '${reported.finishReason?.name ?? 'unknown'}.',
      );
      final extraCounts = <String, int>{
        'prefillMicroseconds': ?reported.prefillDuration?.inMicroseconds,
        'decodeMicroseconds': ?reported.decodeDuration?.inMicroseconds,
        'draftedTokenCount': ?reported.draftedTokenCount,
        'acceptedTokenCount': ?reported.acceptedTokenCount,
      };
      yield ChatResponseUpdate(
        role: ChatRole.assistant,
        finishReason: _chatFinishReason(reported.finishReason),
        usage: UsageDetails(
          inputTokenCount: reported.promptTokenCount,
          outputTokenCount: reported.generatedTokenCount,
          totalTokenCount:
              reported.promptTokenCount + reported.generatedTokenCount,
          cachedInputTokenCount: reported.cachedTokenCount,
          additionalCounts: extraCounts.isEmpty ? null : extraCounts,
        ),
      );
    }
  }

  @override
  Future<ChatResponse> getResponse({
    required Iterable<ChatMessage> messages,
    ChatOptions? options,
    CancellationToken? cancellationToken,
  }) async {
    final text = StringBuffer();
    final extras = <AIContent>[];
    UsageDetails? usage;
    ChatFinishReason? finishReason;
    await for (final update in getStreamingResponse(
      messages: messages,
      options: options,
      cancellationToken: cancellationToken,
    )) {
      if (update.usage != null) usage = update.usage;
      if (update.finishReason != null) finishReason = update.finishReason;
      for (final content in update.contents) {
        if (content is TextContent) {
          text.write(content.text);
        } else {
          extras.add(content);
        }
      }
    }
    return ChatResponse(
      messages: <ChatMessage>[
        ChatMessage(
          role: ChatRole.assistant,
          contents: <AIContent>[
            if (text.isNotEmpty) TextContent(text.toString()),
            ...extras,
          ],
        ),
      ],
      finishReason: finishReason,
      usage: usage,
    );
  }

  @override
  void dispose() {
    // The session is owned by the hosted service that loaded it, not here.
  }
}

/// Maps the engine's finish reason onto the M.E.AI [ChatFinishReason]
/// vocabulary: natural ends (EOG token, stop sequence) become
/// [ChatFinishReason.stop] and truncation (max tokens, full context) becomes
/// [ChatFinishReason.length]. Cancellation and engine aborts have no standard
/// value and stay null, matching how a response cut off mid-stream reports no
/// finish reason.
ChatFinishReason? _chatFinishReason(LlamaFinishReason? reason) =>
    switch (reason) {
      LlamaFinishReason.eogToken ||
      LlamaFinishReason.stopSequence => ChatFinishReason.stop,
      LlamaFinishReason.maxTokens ||
      LlamaFinishReason.contextFull => ChatFinishReason.length,
      LlamaFinishReason.cancelled || LlamaFinishReason.aborted || null => null,
    };
