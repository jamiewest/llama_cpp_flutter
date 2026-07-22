/// Generic ChatML prompt rendering for on-device llama.cpp inference.
///
/// ChatML (`<|im_start|>role\n…<|im_end|>\n`) is the lingua-franca format of
/// many open fine-tunes (Nous Hermes, OpenHermes, dolphin, and the base for
/// Qwen). Tool calling follows the Hermes/Qwen convention (see
/// [hermesToolsSection]). This is a model-agnostic default for any ChatML
/// checkpoint; a model with its own dedicated template (Gemma, LFM2, Qwen)
/// should use that instead.
///
/// No `<|startoftext|>`/BOS is emitted — the native tokenizer adds it.
library;

import 'dart:typed_data';

import 'package:extensions/ai.dart';

import '../chat_format.dart';
import '../common/hermes_tools.dart';
import '../common/parsed_turn.dart';
import '../gemma/gemma_chat_template.dart' show GemmaChatTemplate;

/// Renders ChatML prompts from M.E.AI chat messages and tool declarations.
class ChatmlChatTemplate {
  /// Creates a [ChatmlChatTemplate].
  const ChatmlChatTemplate();

  /// Turn boundary markers.
  static const String imStart = '<|im_start|>';
  static const String imEnd = '<|im_end|>';

  /// The shared mtmd media marker; one is emitted per attached image.
  static const String mediaMarker = GemmaChatTemplate.mediaMarker;

  /// Stop sequence: `<|im_end|>` ends every turn.
  static const List<String> stopSequences = <String>[imEnd];

  /// Renders [messages] (with optional [tools]) into a ChatML prompt.
  RenderedPrompt render(
    Iterable<ChatMessage> messages, {
    Iterable<AIFunctionDeclaration> tools = const <AIFunctionDeclaration>[],
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

    for (final msg in all.sublist(loopStart)) {
      out
        ..write(imStart)
        ..write(msg.role.value)
        ..write('\n')
        ..write(_contentFor(msg, images))
        ..write(imEnd)
        ..write('\n');
    }

    if (addGenerationPrompt) {
      out
        ..write(imStart)
        ..write('assistant\n');
    }

    return RenderedPrompt(
      text: out.toString(),
      stopSequences: stopSequences,
      media: images,
    );
  }

  /// Parses one raw generated turn into prose plus any tool calls.
  ParsedTurn parse(String generated) => parseHermesTurn(generated);

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
      } else if (content is FunctionResultContent) {
        buf.write(
          content.result is String
              ? content.result as String
              : content.result.toString(),
        );
      }
    }
    for (final call in msg.contents.whereType<FunctionCallContent>()) {
      if (buf.isNotEmpty) buf.write('\n');
      buf.write(hermesToolCallBlock(call));
    }
    return buf.toString();
  }
}
