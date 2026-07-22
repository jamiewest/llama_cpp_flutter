/// LFM / LFM-VL prompt rendering for on-device llama.cpp inference.
///
/// Follows Liquid's ChatML-like template:
/// `<|im_start|>role\n...<|im_end|>\n`, with image parts rendered as a media
/// marker. Tool calls are always wrapped in
/// `<|tool_call_start|>`/`<|tool_call_end|>`. LFM2 additionally wraps tool
/// definitions and responses in dedicated tool-list/tool-response tags; LFM2.5
/// uses plain JSON in the system/tool turns.
///
/// Two conventions match [GemmaChatTemplate]:
///   * No `<|startoftext|>` (BOS) is emitted — the native tokenizer adds it
///     (`add_bos_token: true`), so emitting it here too would double it.
///   * Image parts emit `<__media__>` (`GemmaChatTemplate.mediaMarker`), the
///     generic mtmd marker the runtime substitutes with the model's image
///     tokens; the raw jinja literal `<image>` is the post-substitution form.
library;

import 'dart:convert';
import 'dart:typed_data';

import 'package:extensions/ai.dart';

import '../gemma/gemma_chat_template.dart' show GemmaChatTemplate;

/// How Liquid-family prompts wrap tool declarations and tool responses.
enum LfmToolTagStyle {
  /// LFM2 style: wrap definitions and responses in Liquid tool tags.
  lfm2,

  /// LFM2.5 style: use plain JSON text for definitions and responses.
  lfm25,
}

/// How assistant tool-call bodies are rendered.
enum LfmToolCallSyntax {
  /// Liquid's default Pythonic form: `[name(arg="value")]`.
  pythonic,

  /// JSON form requested by adding an instruction to the system turn.
  json,
}

/// A rendered LFM prompt plus the stop sequences generation should halt on.
class Lfm2Prompt {
  const Lfm2Prompt({
    required this.text,
    required this.stopSequences,
    this.images = const <Uint8List>[],
  });

  /// The formatted prompt, ready to pass to `LlamaFlutter.generate`.
  final String text;

  /// Strings that terminate a generation turn. See
  /// [Lfm2ChatTemplate.stopSequences].
  final List<String> stopSequences;

  /// Image bytes referenced by the media markers embedded in [text], in the
  /// order the markers appear. Empty for a text-only prompt.
  final List<Uint8List> images;
}

/// The parsed result of one generated model turn.
class Lfm2Turn {
  const Lfm2Turn({required this.text, required this.calls});

  /// User-visible prose, with tool-call markup removed.
  final String text;

  /// Tool calls the model requested, in emission order. Empty for a plain
  /// answer.
  final List<FunctionCallContent> calls;
}

/// Renders Liquid LFM prompts from M.E.AI chat messages and tool declarations.
class Lfm2ChatTemplate {
  /// Creates an LFM prompt renderer.
  const Lfm2ChatTemplate({
    this.toolTagStyle = LfmToolTagStyle.lfm2,
    this.toolCallSyntax = LfmToolCallSyntax.pythonic,
  });

  /// Whether tool declarations/results use LFM2 tags or LFM2.5 plain JSON.
  final LfmToolTagStyle toolTagStyle;

  /// How assistant tool-call examples are rendered back into chat history.
  final LfmToolCallSyntax toolCallSyntax;

  // Control tokens.
  static const String imStart = '<|im_start|>';
  static const String imEnd = '<|im_end|>';
  static const String toolListStart = '<|tool_list_start|>';
  static const String toolListEnd = '<|tool_list_end|>';
  static const String toolCallStart = '<|tool_call_start|>';
  static const String toolCallEnd = '<|tool_call_end|>';
  static const String toolResponseStart = '<|tool_response_start|>';
  static const String toolResponseEnd = '<|tool_response_end|>';

  /// mtmd's default media marker, reused from the Gemma template. One is
  /// emitted per attached image; the runtime substitutes the model's image
  /// tokens.
  static const String mediaMarker = GemmaChatTemplate.mediaMarker;

  /// Stop sequences a caller passes to generation.
  ///
  /// `<|im_end|>` ends every turn (it is also the model's EOS). A tool-call
  /// turn emits `<|tool_call_start|>…<|tool_call_end|>` and then `<|im_end|>`,
  /// so stopping here captures the whole call block intact.
  static const List<String> stopSequences = <String>[imEnd];

  /// Renders [messages] (with optional [tools]) into an LFM2 prompt.
  ///
  /// [enableThinking] is ignored — LFM2's default template has no reasoning
  /// channel. When [addGenerationPrompt] is true a trailing
  /// `<|im_start|>assistant\n` is appended.
  Lfm2Prompt render(
    Iterable<ChatMessage> messages, {
    Iterable<AIFunctionDeclaration> tools = const <AIFunctionDeclaration>[],
    bool enableThinking = false,
    bool addGenerationPrompt = true,
  }) {
    final all = messages.toList();
    final toolList = tools.toList();
    final out = StringBuffer();
    final images = <Uint8List>[];

    // System turn: an explicit leading system message and/or the tool list.
    var loopStart = 0;
    final system = StringBuffer();
    if (all.isNotEmpty && _roleOf(all.first) == 'system') {
      system.write(all.first.text);
      loopStart = 1;
    }
    if (toolList.isNotEmpty) {
      if (system.isNotEmpty) system.write('\n');
      system
        ..write('List of tools: ')
        ..write(_formatDeclarations(toolList));
      if (toolCallSyntax == LfmToolCallSyntax.json) {
        system.write('\nOutput function calls as JSON.');
      }
    }
    if (system.isNotEmpty) {
      out
        ..write(imStart)
        ..write('system\n')
        ..write(system)
        ..write(imEnd)
        ..write('\n');
    }

    for (final msg in all.sublist(loopStart)) {
      final role = _roleOf(msg);
      out
        ..write(imStart)
        ..write(role)
        ..write('\n')
        ..write(_contentFor(msg, role, images))
        ..write(imEnd)
        ..write('\n');
    }

    if (addGenerationPrompt) {
      out
        ..write(imStart)
        ..write('assistant\n');
    }

    return Lfm2Prompt(
      text: out.toString(),
      stopSequences: stopSequences,
      images: images,
    );
  }

  /// Builds the body of one `<|im_start|>role\n…<|im_end|>` turn.
  String _contentFor(ChatMessage msg, String role, List<Uint8List> images) {
    if (role == 'tool') {
      final results = msg.contents.whereType<FunctionResultContent>().toList();
      final value = results.length == 1
          ? results.first.result
          : results.map((r) => r.result).toList();
      final encoded = value is String ? value : _jsonEncode(value);
      return switch (toolTagStyle) {
        LfmToolTagStyle.lfm2 => '$toolResponseStart$encoded$toolResponseEnd',
        LfmToolTagStyle.lfm25 => encoded,
      };
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

    final calls = msg.contents.whereType<FunctionCallContent>().toList();
    if (calls.isNotEmpty) {
      buf
        ..write(toolCallStart)
        ..write(_formatCalls(calls))
        ..write(toolCallEnd);
    }
    return buf.toString();
  }

  /// Parses one raw generated model turn into prose plus any tool calls.
  ///
  /// [generated] is the text emitted for a single model turn with the
  /// `<|im_end|>` stop sequence already stripped, so any
  /// `<|tool_call_start|>…<|tool_call_end|>` block is complete. Synthetic
  /// sequential [FunctionCallContent.callId]s are assigned so results can be
  /// correlated downstream.
  Lfm2Turn parse(String generated) {
    final calls = <FunctionCallContent>[];
    final text = StringBuffer();
    var cursor = 0;
    while (cursor < generated.length) {
      final open = generated.indexOf(toolCallStart, cursor);
      if (open < 0) {
        text.write(generated.substring(cursor));
        break;
      }
      text.write(generated.substring(cursor, open));
      final close = generated.indexOf(toolCallEnd, open);
      final bodyEnd = close < 0 ? generated.length : close;
      final block = generated.substring(open + toolCallStart.length, bodyEnd);
      _parseCalls(block, calls);
      cursor = close < 0 ? generated.length : close + toolCallEnd.length;
    }
    return Lfm2Turn(text: text.toString().trim(), calls: calls);
  }

  // --- rendering helpers ---

  String _roleOf(ChatMessage m) => m.role.value;

  /// Renders all tool declarations in the style this LFM variant expects.
  String _formatDeclarations(List<AIFunctionDeclaration> tools) {
    final json = '[${tools.map(_formatDeclaration).join(', ')}]';
    return switch (toolTagStyle) {
      LfmToolTagStyle.lfm2 => '$toolListStart$json$toolListEnd',
      LfmToolTagStyle.lfm25 => json,
    };
  }

  /// Renders one tool declaration as a JSON object:
  /// `{"name": ..., "description": ..., "parameters": {...}}`.
  String _formatDeclaration(AIFunctionDeclaration tool) =>
      _jsonEncode(<String, Object?>{
        'name': tool.name,
        'description': tool.description ?? '',
        if (tool.parametersSchema != null) 'parameters': tool.parametersSchema,
      });

  /// Renders assistant tool calls in the configured syntax.
  String _formatCalls(List<FunctionCallContent> calls) =>
      switch (toolCallSyntax) {
        LfmToolCallSyntax.pythonic =>
          '[${calls.map(_formatPythonicCall).join(', ')}]',
        LfmToolCallSyntax.json =>
          calls.length == 1
              ? _formatJsonCall(calls.single)
              : '[${calls.map(_formatJsonCall).join(', ')}]',
      };

  /// Renders one assistant tool call in Pythonic form: `name(arg="value")`.
  String _formatPythonicCall(FunctionCallContent call) {
    final args = call.arguments ?? const <String, Object?>{};
    final parts = args.entries.map((e) => '${e.key}=${_pyLiteral(e.value)}');
    return '${call.name}(${parts.join(', ')})';
  }

  /// Renders one assistant tool call as a JSON object.
  String _formatJsonCall(FunctionCallContent call) =>
      _jsonEncode(<String, Object?>{
        'name': call.name,
        'arguments': call.arguments ?? const <String, Object?>{},
      });

  /// Parses a tool-call block in either Liquid Pythonic or JSON form.
  void _parseCalls(String block, List<FunctionCallContent> calls) {
    final trimmed = block.trim();
    if (trimmed.startsWith('{') || trimmed.startsWith('[')) {
      final jsonCalls = _tryParseJsonCalls(trimmed, calls.length);
      if (jsonCalls != null) {
        calls.addAll(jsonCalls);
        return;
      }
    }
    _PyCallParser(block, calls.length).parseInto(calls);
  }

  List<FunctionCallContent>? _tryParseJsonCalls(String block, int startIndex) {
    Object? decoded;
    try {
      decoded = jsonDecode(block);
    } on FormatException {
      return null;
    }

    final entries = decoded is List ? decoded : <Object?>[decoded];
    final parsed = <FunctionCallContent>[];
    for (final entry in entries) {
      if (entry is! Map) return null;
      final name = entry['name'];
      if (name is! String) return null;
      final args = entry['arguments'] ?? entry['parameters'] ?? const {};
      if (args is! Map) return null;
      parsed.add(
        FunctionCallContent(
          callId: 'call_${startIndex + parsed.length}',
          name: name,
          arguments: args.cast<String, Object?>(),
        ),
      );
    }
    return parsed;
  }

  /// JSON with Python `json.dumps` default separators (`, ` and `: `) to match
  /// the upstream jinja `tojson` filter the model was trained on.
  static String _jsonEncode(Object? value) {
    if (value is Map) {
      final parts = value.entries.map(
        (e) => '${jsonEncode(e.key.toString())}: ${_jsonEncode(e.value)}',
      );
      return '{${parts.join(', ')}}';
    }
    if (value is Iterable) {
      return '[${value.map(_jsonEncode).join(', ')}]';
    }
    return jsonEncode(value);
  }

  /// Renders a value as a Python literal for a tool-call argument.
  static String _pyLiteral(Object? value) {
    if (value == null) return 'None';
    if (value is bool) return value ? 'True' : 'False';
    if (value is num) return value.toString();
    if (value is String) return jsonEncode(value);
    if (value is Map) {
      final parts = value.entries.map(
        (e) => '${jsonEncode(e.key.toString())}: ${_pyLiteral(e.value)}',
      );
      return '{${parts.join(', ')}}';
    }
    if (value is Iterable) {
      return '[${value.map(_pyLiteral).join(', ')}]';
    }
    return jsonEncode(value.toString());
  }
}

/// Recursive-descent reader for LFM2's Pythonic tool-call list.
///
/// Parses `[name(arg="s", n=1, flag=True, items=[1,2])]` — one or more calls
/// of the form `name(key=value, …)`. Values are Python literals: strings
/// (`"…"` or `'…'`), numbers, `True`/`False`/`None`, lists and dicts. Throws
/// [FormatException] on malformed input so the decoder can fall back to raw
/// text.
class _PyCallParser {
  _PyCallParser(this._source, this._startIndex);

  final String _source;
  final int _startIndex;
  int _pos = 0;

  /// Parses the call list and appends each call to [calls].
  ///
  /// The list is normally wrapped in `[ … ]`, but LFM2 also emits a single
  /// bare call (`name(arg=…)`) with no brackets; both shapes are accepted. A
  /// genuinely malformed block (unterminated string, trailing junk) throws
  /// [FormatException] so the decoder can fall back to raw text.
  void parseInto(List<FunctionCallContent> calls) {
    _skipWs();
    final bracketed = _peek() == '[';
    if (bracketed) {
      _pos++;
      _skipWs();
      if (_peek() == ']') {
        _pos++;
        return;
      }
    } else if (_peek().isEmpty) {
      return;
    }
    while (true) {
      calls.add(_parseCall(_startIndex + calls.length));
      _skipWs();
      final c = _peek();
      if (c == ',') {
        _pos++;
        _skipWs();
        continue;
      }
      if (bracketed) {
        _expect(']');
      } else if (c.isNotEmpty) {
        throw FormatException('Unexpected "$c" at $_pos', _source, _pos);
      }
      break;
    }
  }

  FunctionCallContent _parseCall(int index) {
    final name = _parseIdentifier();
    _skipWs();
    _expect('(');
    final args = <String, Object?>{};
    _skipWs();
    if (_peek() != ')') {
      while (true) {
        _skipWs();
        final key = _parseIdentifier();
        _skipWs();
        _expect('=');
        _skipWs();
        args[key] = _parseValue();
        _skipWs();
        if (_peek() == ',') {
          _pos++;
          continue;
        }
        break;
      }
    }
    _skipWs();
    _expect(')');
    return FunctionCallContent(
      callId: 'call_$index',
      name: name,
      arguments: args,
    );
  }

  String _parseIdentifier() {
    final start = _pos;
    while (_pos < _source.length && _isIdentChar(_source[_pos])) {
      _pos++;
    }
    if (_pos == start) {
      throw FormatException('Expected identifier at $_pos', _source, _pos);
    }
    return _source.substring(start, _pos);
  }

  Object? _parseValue() {
    final c = _peek();
    if (c == '"' || c == "'") return _parseString(c);
    if (c == '[') return _parseList();
    if (c == '{') return _parseDict();
    return _parseLiteral();
  }

  String _parseString(String quote) {
    _pos++;
    final buf = StringBuffer();
    while (_pos < _source.length) {
      final ch = _source[_pos];
      if (ch == '\\' && _pos + 1 < _source.length) {
        final next = _source[_pos + 1];
        buf.write(_unescape(next));
        _pos += 2;
        continue;
      }
      if (ch == quote) {
        _pos++;
        return buf.toString();
      }
      buf.write(ch);
      _pos++;
    }
    throw FormatException('Unterminated string at $_pos', _source, _pos);
  }

  String _unescape(String c) {
    switch (c) {
      case 'n':
        return '\n';
      case 't':
        return '\t';
      case 'r':
        return '\r';
      default:
        return c;
    }
  }

  List<Object?> _parseList() {
    _expect('[');
    final list = <Object?>[];
    _skipWs();
    if (_peek() == ']') {
      _pos++;
      return list;
    }
    while (true) {
      _skipWs();
      list.add(_parseValue());
      _skipWs();
      if (_peek() == ',') {
        _pos++;
        continue;
      }
      _expect(']');
      break;
    }
    return list;
  }

  Map<String, Object?> _parseDict() {
    _expect('{');
    final map = <String, Object?>{};
    _skipWs();
    if (_peek() == '}') {
      _pos++;
      return map;
    }
    while (true) {
      _skipWs();
      final keyChar = _peek();
      final key = (keyChar == '"' || keyChar == "'")
          ? _parseString(keyChar)
          : _parseIdentifier();
      _skipWs();
      _expect(':');
      _skipWs();
      map[key] = _parseValue();
      _skipWs();
      if (_peek() == ',') {
        _pos++;
        continue;
      }
      _expect('}');
      break;
    }
    return map;
  }

  Object? _parseLiteral() {
    final start = _pos;
    while (_pos < _source.length && !',)]}'.contains(_source[_pos])) {
      _pos++;
    }
    final raw = _source.substring(start, _pos).trim();
    // LFM2 mixes Python and JSON literals; accept both spellings so a boolean
    // argument is never silently coerced to the string "true"/"false"/"null".
    switch (raw) {
      case 'True':
      case 'true':
        return true;
      case 'False':
      case 'false':
        return false;
      case 'None':
      case 'null':
        return null;
    }
    return num.tryParse(raw) ?? raw;
  }

  bool _isIdentChar(String c) {
    final u = c.codeUnitAt(0);
    return (u >= 65 && u <= 90) || // A-Z
        (u >= 97 && u <= 122) || // a-z
        (u >= 48 && u <= 57) || // 0-9
        c == '_';
  }

  void _skipWs() {
    while (_pos < _source.length && _source[_pos].trim().isEmpty) {
      _pos++;
    }
  }

  String _peek() => _pos < _source.length ? _source[_pos] : '';

  void _expect(String char) {
    if (_peek() != char) {
      throw FormatException('Expected "$char" at $_pos', _source, _pos);
    }
    _pos++;
  }
}
