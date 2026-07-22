/// Qwen2.5 / Qwen3 prompt rendering for on-device llama.cpp inference.
///
/// Qwen builds on ChatML (`<|im_start|>role\n…<|im_end|>\n`) and uses the
/// Hermes tool convention (see [hermesToolsSection]) with two Qwen specifics:
///   * tool results are wrapped in `<tool_response>…</tool_response>` inside a
///     `user` turn (consecutive results are grouped into one turn);
///   * Qwen3 emits an optional `<think>…</think>` reasoning channel, surfaced
///     as [TextReasoningContent] by [QwenStreamDecoder].
///
/// No BOS is emitted — the native tokenizer adds it.
library;

import 'dart:typed_data';

import 'package:extensions/ai.dart';

import '../chat_format.dart';
import '../common/hermes_tools.dart';
import '../common/parsed_turn.dart';
import '../gemma/gemma_chat_template.dart' show GemmaChatTemplate;

/// Renders Qwen prompts from M.E.AI chat messages and tool declarations.
class QwenChatTemplate {
  /// Creates a [QwenChatTemplate].
  const QwenChatTemplate();

  /// Turn boundary markers.
  static const String imStart = '<|im_start|>';
  static const String imEnd = '<|im_end|>';

  /// Reasoning-channel markers (Qwen3).
  static const String thinkOpen = '<think>';
  static const String thinkClose = '</think>';

  /// Tool-response wrapper inside a `tool` turn.
  static const String toolResponseOpen = '<tool_response>';
  static const String toolResponseClose = '</tool_response>';

  /// The shared mtmd media marker; one is emitted per attached image.
  static const String mediaMarker = GemmaChatTemplate.mediaMarker;

  /// Stop sequence: `<|im_end|>` ends every turn.
  static const List<String> stopSequences = <String>[imEnd];

  /// Renders [messages] (with optional [tools]) into a Qwen prompt.
  ///
  /// Qwen3 thinks by default; [enableThinking] false pre-fills the
  /// generation prompt with an empty `<think></think>` block — exactly what
  /// the upstream template's `enable_thinking=false` emits — so the model
  /// skips the reasoning pass instead of burning decode tokens on it.
  RenderedPrompt render(
    Iterable<ChatMessage> messages, {
    Iterable<AIFunctionDeclaration> tools = const <AIFunctionDeclaration>[],
    bool enableThinking = true,
    bool addGenerationPrompt = true,
  }) {
    final all = messages.toList();
    final out = StringBuffer();
    final images = <Uint8List>[];

    var loopStart = 0;
    final systemParts = <String>[];
    if (all.isNotEmpty && all.first.role == ChatRole.system) {
      final text = all.first.text;
      if (text.isNotEmpty) systemParts.add(text);
      loopStart = 1;
    }
    final toolsSection = hermesToolsSection(tools);
    if (toolsSection.isNotEmpty) systemParts.add(toolsSection);
    if (systemParts.isNotEmpty) {
      out
        ..write(imStart)
        ..write('system\n')
        ..write(systemParts.join('\n\n'))
        ..write(imEnd)
        ..write('\n');
    }

    final body = all.sublist(loopStart);
    var i = 0;
    while (i < body.length) {
      if (body[i].role == ChatRole.tool) {
        // Qwen groups a run of consecutive tool results into one `user` turn,
        // each wrapped in `<tool_response>` (matching the upstream template).
        final responses = <String>[];
        while (i < body.length && body[i].role == ChatRole.tool) {
          for (final r in body[i].contents.whereType<FunctionResultContent>()) {
            responses.add(
              '$toolResponseOpen\n${_resultText(r.result)}\n$toolResponseClose',
            );
          }
          i++;
        }
        out
          ..write(imStart)
          ..write('user\n')
          ..write(responses.join('\n'))
          ..write(imEnd)
          ..write('\n');
        continue;
      }
      out
        ..write(imStart)
        ..write(body[i].role.value)
        ..write('\n')
        ..write(_contentFor(body[i], images))
        ..write(imEnd)
        ..write('\n');
      i++;
    }

    if (addGenerationPrompt) {
      out
        ..write(imStart)
        ..write('assistant\n');
      if (!enableThinking) {
        out.write('$thinkOpen\n\n$thinkClose\n\n');
      }
    }

    return RenderedPrompt(
      text: out.toString(),
      stopSequences: stopSequences,
      media: images,
    );
  }

  /// Parses one raw generated turn into prose plus any tool calls.
  ParsedTurn parse(String generated) => parseHermesTurn(generated);

  static String _resultText(Object? result) =>
      result is String ? result : result.toString();

  String _contentFor(ChatMessage msg, List<Uint8List> images) {
    final buf = StringBuffer();
    for (final content in msg.contents) {
      if (content is TextContent) {
        buf.write(content.text);
      } else if (content is DataContent &&
          content.data != null &&
          content.hasTopLevelMediaType('image')) {
        images.add(content.data!);
        buf.write(mediaMarker);
      }
    }
    for (final call in msg.contents.whereType<FunctionCallContent>()) {
      if (buf.isNotEmpty) buf.write('\n');
      buf.write(hermesToolCallBlock(call));
    }
    return buf.toString();
  }
}
