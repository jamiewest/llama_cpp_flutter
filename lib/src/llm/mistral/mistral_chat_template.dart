/// Mistral / Mixtral (v3 tokenizer) prompt rendering for on-device llama.cpp
/// inference.
///
/// Mistral does not use ChatML. User turns are wrapped in `[INST] … [/INST]`,
/// assistant turns end with `</s>`, and there is no dedicated system role — the
/// system prompt is prepended to the first user turn. Tool calling follows the
/// v3 convention: available tools are advertised in an
/// `[AVAILABLE_TOOLS][…][/AVAILABLE_TOOLS]` block immediately before the last
/// user turn, the model replies with `[TOOL_CALLS][{…}]`, and results are fed
/// back in `[TOOL_RESULTS]…[/TOOL_RESULTS]`.
///
/// The v3 call-correlation ids (the `id` on `[TOOL_CALLS]` and `call_id` on
/// `[TOOL_RESULTS]`) are omitted; on-device single-turn tool use does not need
/// them.
///
/// No `<s>`/BOS is emitted — the native tokenizer adds it.
library;

import 'dart:convert';
import 'dart:typed_data';

import 'package:extensions/ai.dart';

import '../chat_format.dart';
import '../common/parsed_turn.dart';
import '../gemma/gemma_chat_template.dart' show GemmaChatTemplate;

/// Renders Mistral prompts from M.E.AI chat messages and tool declarations.
class MistralChatTemplate {
  /// Creates a [MistralChatTemplate].
  const MistralChatTemplate();

  /// Instruction wrappers.
  static const String instStart = '[INST]';
  static const String instEnd = '[/INST]';
  static const String eos = '</s>';

  /// Tool markers.
  static const String availableToolsStart = '[AVAILABLE_TOOLS]';
  static const String availableToolsEnd = '[/AVAILABLE_TOOLS]';
  static const String toolCalls = '[TOOL_CALLS]';
  static const String toolResultsStart = '[TOOL_RESULTS]';
  static const String toolResultsEnd = '[/TOOL_RESULTS]';

  /// The shared mtmd media marker; one is emitted per attached image.
  static const String mediaMarker = GemmaChatTemplate.mediaMarker;

  /// Stop sequence: `</s>` ends an assistant turn.
  static const List<String> stopSequences = <String>[eos];

  /// Renders [messages] (with optional [tools]) into a Mistral prompt.
  RenderedPrompt render(
    Iterable<ChatMessage> messages, {
    Iterable<AIFunctionDeclaration> tools = const <AIFunctionDeclaration>[],
    bool addGenerationPrompt = true,
  }) {
    final all = messages.toList();
    final images = <Uint8List>[];

    var loopStart = 0;
    String? system;
    if (all.isNotEmpty && all.first.role == ChatRole.system) {
      final text = all.first.text;
      if (text.isNotEmpty) system = text;
      loopStart = 1;
    }

    final body = all.sublist(loopStart);
    final toolsBlock = _toolsBlock(tools);
    final lastUser = body.lastIndexWhere((m) => m.role == ChatRole.user);
    final out = StringBuffer();
    var firstUserSeen = false;

    for (var i = 0; i < body.length; i++) {
      final msg = body[i];
      if (msg.role == ChatRole.user) {
        final prefix = StringBuffer();
        if (i == lastUser && toolsBlock.isNotEmpty) prefix.write(toolsBlock);
        prefix.write(instStart);
        prefix.write(' ');
        if (!firstUserSeen && system != null) {
          prefix
            ..write(system)
            ..write('\n\n');
          firstUserSeen = true;
        }
        out
          ..write(prefix)
          ..write(_userContent(msg, images))
          ..write(' ')
          ..write(instEnd);
      } else if (msg.role == ChatRole.tool) {
        out
          ..write(toolResultsStart)
          ..write(_toolResult(msg))
          ..write(toolResultsEnd);
      } else {
        // Assistant turn: prose and/or a tool-call block, then EOS.
        final calls = msg.contents.whereType<FunctionCallContent>().toList();
        if (calls.isNotEmpty) {
          out
            ..write(toolCalls)
            ..write('[')
            ..write(calls.map(_callJson).join(', '))
            ..write(']');
        } else {
          out
            ..write(' ')
            ..write(msg.text);
        }
        out.write(eos);
      }
    }

    return RenderedPrompt(
      text: out.toString(),
      stopSequences: stopSequences,
      media: images,
    );
  }

  /// Parses one raw generated turn into prose plus any tool calls.
  ///
  /// Throws [FormatException] when the `[TOOL_CALLS]` body is not a valid JSON
  /// array so the decoder can fall back to raw text.
  ParsedTurn parse(String generated) {
    final at = generated.indexOf(toolCalls);
    if (at < 0) {
      return ParsedTurn(text: generated.trim(), calls: const []);
    }
    final text = generated.substring(0, at).trim();
    final body = generated.substring(at + toolCalls.length).trim();
    final decoded = jsonDecode(body);
    if (decoded is! List) {
      throw FormatException('Tool calls are not a JSON array', body);
    }
    final calls = <FunctionCallContent>[];
    for (final entry in decoded) {
      if (entry is! Map) {
        throw FormatException('Tool call is not a JSON object', body);
      }
      final args = (entry['arguments'] ?? entry['parameters']) as Map?;
      calls.add(
        FunctionCallContent(
          callId: 'call_${calls.length}',
          name: entry['name'] as String? ?? '',
          arguments: args?.cast<String, Object?>() ?? const <String, Object?>{},
        ),
      );
    }
    return ParsedTurn(text: text, calls: calls);
  }

  String _userContent(ChatMessage msg, List<Uint8List> images) {
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
    return buf.toString();
  }

  String _toolResult(ChatMessage msg) {
    final results = msg.contents.whereType<FunctionResultContent>().toList();
    final value = results.length == 1
        ? results.first.result
        : results.map((r) => r.result).toList();
    return value is String ? value : jsonEncode(value);
  }

  String _callJson(FunctionCallContent call) => jsonEncode(<String, Object?>{
    'name': call.name,
    'arguments': call.arguments ?? const <String, Object?>{},
  });

  String _toolsBlock(Iterable<AIFunctionDeclaration> tools) {
    final list = tools.toList();
    if (list.isEmpty) return '';
    final json = list
        .map(
          (t) => jsonEncode(<String, Object?>{
            'type': 'function',
            'function': <String, Object?>{
              'name': t.name,
              'description': t.description ?? '',
              if (t.parametersSchema != null) 'parameters': t.parametersSchema,
            },
          }),
        )
        .join(', ');
    return '$availableToolsStart[$json]$availableToolsEnd';
  }
}
