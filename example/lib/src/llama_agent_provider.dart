import 'package:agents/agents.dart';
import 'package:extensions/ai.dart' as ai;
import 'package:flutter/foundation.dart';
import 'package:flutter_ai_toolkit/flutter_ai_toolkit.dart';

/// Bridges flutter_ai_toolkit's [LlmProvider] to an agents-framework
/// [AIAgent] backed by a local llama.cpp model.
///
/// The provider owns the transcript (as the toolkit expects) and replays the
/// full mapped history on every turn with a null agent session, so the agent
/// side stays stateless and the toolkit's `history` setter (used for chat
/// serialization and message editing) just works.
class LlamaAgentProvider extends LlmProvider with ChangeNotifier {
  LlamaAgentProvider({required AIAgent? Function() agentResolver})
    : _agentResolver = agentResolver;

  /// Resolves the current agent so model/agent reconfiguration doesn't
  /// require swapping the provider (and losing the chat view's state).
  final AIAgent? Function() _agentResolver;

  final List<ChatMessage> _history = [];

  @override
  Stream<String> generateStream(
    String prompt, {
    Iterable<Attachment> attachments = const [],
  }) {
    final agent = _requireAgent();
    return _run(agent, [_toAiMessage(ChatMessage.user(prompt, attachments))]);
  }

  @override
  Stream<String> sendMessageStream(
    String prompt, {
    Iterable<Attachment> attachments = const [],
  }) async* {
    final agent = _requireAgent();
    final userMessage = ChatMessage.user(prompt, attachments);
    final llmMessage = ChatMessage.llm();
    final messages = [..._history, userMessage].map(_toAiMessage).toList();
    _history.addAll([userMessage, llmMessage]);
    yield* _run(agent, messages).map((chunk) {
      llmMessage.append(chunk);
      return chunk;
    });
    notifyListeners();
  }

  Stream<String> _run(AIAgent agent, List<ai.ChatMessage> messages) async* {
    await for (final update in agent.runStreaming(
      null,
      null,
      messages: messages,
    )) {
      // Updates without text (tool calls, tool results) are progress, not
      // transcript content.
      final text = update.text;
      if (text.isNotEmpty) yield text;
    }
  }

  AIAgent _requireAgent() {
    final agent = _agentResolver();
    if (agent == null) {
      throw const LlmFailureException(
        'No model is loaded. Pick one in the model settings first.',
      );
    }
    return agent;
  }

  ai.ChatMessage _toAiMessage(ChatMessage message) {
    final contents = <ai.AIContent>[
      if (message.text case final text? when text.isNotEmpty)
        ai.TextContent(text),
      for (final attachment in message.attachments)
        switch (attachment) {
          FileAttachment() => ai.DataContent(
            attachment.bytes,
            mediaType: attachment.mimeType,
            name: attachment.name,
          ),
          LinkAttachment() => ai.TextContent('[link: ${attachment.url}]'),
        },
    ];
    return ai.ChatMessage(
      role: message.origin.isUser ? ai.ChatRole.user : ai.ChatRole.assistant,
      contents: contents,
    );
  }

  /// Starts a fresh conversation (e.g. after loading a different model).
  void clear() {
    _history.clear();
    notifyListeners();
  }

  @override
  Iterable<ChatMessage> get history => List.unmodifiable(_history);

  @override
  set history(Iterable<ChatMessage> history) {
    _history
      ..clear()
      ..addAll(history);
    notifyListeners();
  }
}
