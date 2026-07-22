/// The Hermes / Qwen tool-calling convention shared by the ChatML and Qwen
/// families.
///
/// Tools are advertised in the system turn inside `<tools></tools>` XML tags as
/// one JSON object per line, and the model replies with one or more
/// `<tool_call>{"name":…,"arguments":{…}}</tool_call>` blocks. This is the
/// convention used by Nous Hermes 2 Pro and Qwen2.5/Qwen3 instruct templates.
library;

import 'dart:convert';

import 'package:extensions/ai.dart';

import 'parsed_turn.dart';

/// Open/close tags that wrap a single tool call in generated output.
const String hermesToolCallOpen = '<tool_call>';
const String hermesToolCallClose = '</tool_call>';

/// Builds the `# Tools` section appended to the system turn for [tools].
///
/// Returns an empty string when [tools] is empty.
String hermesToolsSection(Iterable<AIFunctionDeclaration> tools) {
  final list = tools.toList();
  if (list.isEmpty) return '';
  final signatures = list.map(_toolSignature).join('\n');
  return '# Tools\n\n'
      'You may call one or more functions to assist with the user query.\n\n'
      'You are provided with function signatures within '
      '<tools></tools> XML tags:\n'
      '<tools>\n$signatures\n</tools>\n\n'
      'For each function call, return a json object with function name and '
      'arguments within <tool_call></tool_call> XML tags:\n'
      '$hermesToolCallOpen\n'
      '{"name": <function-name>, "arguments": <args-json-object>}\n'
      '$hermesToolCallClose';
}

/// Renders one assistant tool call as a `<tool_call>…</tool_call>` block.
String hermesToolCallBlock(FunctionCallContent call) {
  final payload = <String, Object?>{
    'name': call.name,
    'arguments': call.arguments ?? const <String, Object?>{},
  };
  return '$hermesToolCallOpen\n${jsonEncode(payload)}\n$hermesToolCallClose';
}

/// Parses [generated] into trailing prose plus any `<tool_call>` blocks.
///
/// Throws [FormatException] when a block's body is not valid JSON so the
/// decoder can fall back to raw text.
ParsedTurn parseHermesTurn(String generated) {
  final calls = <FunctionCallContent>[];
  final text = StringBuffer();
  var cursor = 0;
  while (cursor < generated.length) {
    final open = generated.indexOf(hermesToolCallOpen, cursor);
    if (open < 0) {
      text.write(generated.substring(cursor));
      break;
    }
    text.write(generated.substring(cursor, open));
    final bodyStart = open + hermesToolCallOpen.length;
    final close = generated.indexOf(hermesToolCallClose, bodyStart);
    final bodyEnd = close < 0 ? generated.length : close;
    final body = generated.substring(bodyStart, bodyEnd).trim();
    final decoded = jsonDecode(body);
    if (decoded is! Map) {
      throw FormatException('Tool call is not a JSON object', body);
    }
    calls.add(
      FunctionCallContent(
        callId: 'call_${calls.length}',
        name: decoded['name'] as String? ?? '',
        arguments:
            (decoded['arguments'] as Map?)?.cast<String, Object?>() ??
            const <String, Object?>{},
      ),
    );
    cursor = close < 0 ? generated.length : close + hermesToolCallClose.length;
  }
  return ParsedTurn(text: text.toString().trim(), calls: calls);
}

String _toolSignature(AIFunctionDeclaration tool) =>
    jsonEncode(<String, Object?>{
      'type': 'function',
      'function': <String, Object?>{
        'name': tool.name,
        'description': tool.description ?? '',
        if (tool.parametersSchema != null) 'parameters': tool.parametersSchema,
      },
    });
