/// Llama 3 / 3.1 prompt rendering for on-device llama.cpp inference.
///
/// Turns use Llama 3's header format
/// (`<|start_header_id|>role<|end_header_id|>\n\n…<|eot_id|>`). Tool calling
/// follows the Llama 3.1 `python_tag` JSON convention: tools are advertised in
/// the system turn and the model replies with
/// `<|python_tag|>{"name":…,"parameters":{…}}<|eom_id|>`. Tool results are fed
/// back in an `ipython` turn.
///
/// Decoding keys on `<|python_tag|>`; a checkpoint configured for the
/// bare-JSON tool mode (no tag) is not auto-detected and would surface the
/// JSON as prose.
///
/// No `<|begin_of_text|>`/BOS is emitted — the native tokenizer adds it.
library;

import 'dart:convert';
import 'dart:typed_data';

import 'package:extensions/ai.dart';

import '../chat_format.dart';
import '../common/parsed_turn.dart';
import '../gemma/gemma_chat_template.dart' show GemmaChatTemplate;

/// Renders Llama 3 prompts from M.E.AI chat messages and tool declarations.
class Llama3ChatTemplate {
  /// Creates a [Llama3ChatTemplate].
  const Llama3ChatTemplate();

  /// Header and turn markers.
  static const String headerStart = '<|start_header_id|>';
  static const String headerEnd = '<|end_header_id|>';
  static const String eot = '<|eot_id|>';
  static const String eom = '<|eom_id|>';

  /// Marker that prefixes a tool call.
  static const String pythonTag = '<|python_tag|>';

  /// The shared mtmd media marker; one is emitted per attached image.
  static const String mediaMarker = GemmaChatTemplate.mediaMarker;

  /// Stop sequences: a normal turn ends with `<|eot_id|>`; a tool-call turn
  /// ends with `<|eom_id|>`.
  static const List<String> stopSequences = <String>[eot, eom];

  /// Renders [messages] (with optional [tools]) into a Llama 3 prompt.
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
    final toolsSection = _toolsSection(tools);
    if (toolsSection.isNotEmpty) systemParts.add(toolsSection);
    if (systemParts.isNotEmpty) {
      _writeTurn(out, 'system', systemParts.join('\n\n'));
    }

    for (final msg in all.sublist(loopStart)) {
      final hasCall = msg.contents.any((c) => c is FunctionCallContent);
      _writeTurn(
        out,
        _roleOf(msg),
        _contentFor(msg, images),
        terminator: hasCall ? eom : eot,
      );
    }

    if (addGenerationPrompt) {
      out
        ..write(headerStart)
        ..write('assistant')
        ..write(headerEnd)
        ..write('\n\n');
    }

    return RenderedPrompt(
      text: out.toString(),
      stopSequences: stopSequences,
      media: images,
    );
  }

  /// Parses one raw generated turn into prose plus any tool calls.
  ///
  /// Throws [FormatException] when the `python_tag` body is not valid JSON so
  /// the decoder can fall back to raw text.
  ParsedTurn parse(String generated) {
    final at = generated.indexOf(pythonTag);
    if (at < 0) {
      return ParsedTurn(text: generated.trim(), calls: const []);
    }
    final text = generated.substring(0, at).trim();
    var body = generated.substring(at + pythonTag.length);
    final endAt = body.indexOf(eom);
    if (endAt >= 0) body = body.substring(0, endAt);
    final decoded = jsonDecode(body.trim());
    if (decoded is! Map) {
      throw FormatException('Tool call is not a JSON object', body);
    }
    final args = (decoded['parameters'] ?? decoded['arguments']) as Map?;
    return ParsedTurn(
      text: text,
      calls: <FunctionCallContent>[
        FunctionCallContent(
          callId: 'call_0',
          name: decoded['name'] as String? ?? '',
          arguments: args?.cast<String, Object?>() ?? const <String, Object?>{},
        ),
      ],
    );
  }

  void _writeTurn(
    StringBuffer out,
    String role,
    String content, {
    String terminator = eot,
  }) {
    out
      ..write(headerStart)
      ..write(role)
      ..write(headerEnd)
      ..write('\n\n')
      ..write(content)
      ..write(terminator)
      ..write('\n');
  }

  String _roleOf(ChatMessage msg) =>
      msg.role == ChatRole.tool ? 'ipython' : msg.role.value;

  String _contentFor(ChatMessage msg, List<Uint8List> images) {
    if (msg.role == ChatRole.tool) {
      return msg.contents
          .whereType<FunctionResultContent>()
          .map((r) => r.result is String ? r.result : jsonEncode(r.result))
          .join('\n');
    }

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
      buf
        ..write(pythonTag)
        ..write(
          jsonEncode(<String, Object?>{
            'name': call.name,
            'parameters': call.arguments ?? const <String, Object?>{},
          }),
        );
    }
    return buf.toString();
  }

  String _toolsSection(Iterable<AIFunctionDeclaration> tools) {
    final list = tools.toList();
    if (list.isEmpty) return '';
    final json = list
        .map(
          (t) => jsonEncode(<String, Object?>{
            'name': t.name,
            'description': t.description ?? '',
            if (t.parametersSchema != null) 'parameters': t.parametersSchema,
          }),
        )
        .join(', ');
    return 'You have access to the following functions. To call one, reply '
        'with a JSON object of the form {"name": <name>, "parameters": <args>} '
        'prefixed by $pythonTag and ended by $eom.\n\n[$json]';
  }
}
