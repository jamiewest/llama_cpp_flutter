/// Gemma 4 prompt rendering for on-device llama.cpp inference.
///
/// This is a faithful Dart port of the upstream `gemma-4-E4B-it`
/// `chat_template.jinja`, validated byte-for-byte against fixtures generated
/// from that template (see `tool/gemma_fixtures/`). It converts
/// Microsoft.Extensions.AI [ChatMessage]s + tool declarations into the Gemma 4
/// wire format so the result can be fed straight to `LlamaFlutter.generate`.
///
/// Two deliberate conventions:
///   * No `<bos>` is emitted — the native tokenizer adds BOS, so emitting it
///     here too would double it.
///   * Map keys are emitted in sorted order, matching the template's
///     `dictsort`.
library;

import 'dart:typed_data';

import 'package:extensions/ai.dart';

/// A rendered Gemma prompt plus the stop sequences generation should halt on.
class GemmaPrompt {
  const GemmaPrompt({
    required this.text,
    required this.stopSequences,
    this.media = const <Uint8List>[],
  });

  /// The formatted prompt, ready to pass to `LlamaFlutter.generate`.
  final String text;

  /// Strings that terminate a generation turn. See
  /// [GemmaChatTemplate.stopSequences].
  final List<String> stopSequences;

  /// Encoded media bytes (image or audio) referenced by the [mediaMarker]s
  /// embedded in [text], in the order the markers appear. Passed straight to
  /// the runtime so mtmd can interleave them; mtmd auto-detects each blob's
  /// kind from its magic bytes. Empty for a text-only prompt.
  final List<Uint8List> media;
}

/// The parsed result of one generated model turn.
class GemmaTurn {
  const GemmaTurn({required this.text, required this.calls});

  /// User-visible prose, with tool-call markup and thinking removed.
  final String text;

  /// Tool calls the model requested, in emission order. Empty for a plain
  /// answer.
  final List<FunctionCallContent> calls;
}

/// Renders Gemma 4 prompts from M.E.AI chat messages and tool declarations.
class GemmaChatTemplate {
  const GemmaChatTemplate();

  // Control tokens.
  static const String turnOpen = '<|turn>';
  static const String turnClose = '<turn|>';
  static const String toolOpen = '<|tool>';
  static const String toolClose = '<tool|>';
  static const String toolCallOpen = '<|tool_call>';
  static const String toolCallClose = '<tool_call|>';
  static const String toolResponseOpen = '<|tool_response>';
  static const String toolResponseClose = '<tool_response|>';
  static const String stringDelimiter = '<|"|>';
  static const String thinkToken = '<|think|>';
  static const String channelOpen = '<|channel>';
  static const String channelClose = '<channel|>';

  /// mtmd's default media marker. One is emitted into the prompt text per
  /// attached image; mtmd ([LlamaSession] `evalMultimodalPrompt`) splits the
  /// prompt on these and substitutes the model-specific image tokens. Matches
  /// `mtmd_default_marker()` ("<__media__>") in the vendored `mtmd.h`.
  static const String mediaMarker = '<__media__>';

  /// Stop sequences a caller passes to generation.
  ///
  /// `<turn|>` ends a plain answer; `<|tool_response>` is the model's cue that
  /// it has finished emitting calls and wants results — stopping there (rather
  /// than at `<tool_call|>`) keeps any number of consecutive tool calls intact.
  /// The client decides "are there calls?" by scanning the captured text for
  /// [toolCallOpen].
  static const List<String> stopSequences = <String>[
    turnClose,
    toolResponseOpen,
  ];

  /// Renders [messages] (with optional [tools]) into a Gemma 4 prompt.
  ///
  /// When [addGenerationPrompt] is true a trailing `<|turn>model` is appended
  /// unless the conversation already ends mid-model-turn (after a tool call or
  /// response). [enableThinking] injects the `<|think|>` marker.
  GemmaPrompt render(
    Iterable<ChatMessage> messages, {
    Iterable<AIFunctionDeclaration> tools = const <AIFunctionDeclaration>[],
    bool enableThinking = false,
    bool addGenerationPrompt = true,
  }) {
    final all = messages.toList();
    final toolList = tools.toList();
    final out = StringBuffer();
    final media = <Uint8List>[];

    var loopStart = 0;
    final firstIsSystem = all.isNotEmpty && _roleOf(all.first) == 'system';
    if (enableThinking || toolList.isNotEmpty || firstIsSystem) {
      out.write(
        '$turnOpen'
        'system\n',
      );
      if (enableThinking) {
        out.write('$thinkToken\n');
      }
      if (firstIsSystem) {
        out.write(all.first.text.trim());
        loopStart = 1;
      }
      for (final tool in toolList) {
        out
          ..write(toolOpen)
          ..write(_formatDeclaration(tool).trim())
          ..write(toolClose);
      }
      out.write('$turnClose\n');
    }

    final loop = all.sublist(loopStart);
    String? prevType;
    for (var i = 0; i < loop.length; i++) {
      final msg = loop[i];
      if (_roleOf(msg) == 'tool') {
        continue;
      }
      prevType = null;

      final isAssistant = _roleOf(msg) == 'assistant';
      final roleStr = isAssistant ? 'model' : _roleOf(msg);
      final prev = _prevNonTool(loop, i);
      final continueSameTurn =
          roleStr == 'model' && prev != null && _roleOf(prev) == 'assistant';
      if (!continueSameTurn) {
        out.write('$turnOpen$roleStr\n');
      }

      final calls = msg.contents.whereType<FunctionCallContent>().toList();
      if (calls.isNotEmpty) {
        for (final call in calls) {
          out
            ..write(toolCallOpen)
            ..write('call:${call.name}')
            ..write(
              _formatArgument(
                call.arguments ?? const <String, Object?>{},
                escapeKeys: false,
              ),
            )
            ..write(toolCallClose);
        }
        prevType = 'tool_call';
      }

      var emittedResponse = false;
      if (calls.isNotEmpty) {
        for (
          var k = i + 1;
          k < loop.length && _roleOf(loop[k]) == 'tool';
          k++
        ) {
          for (final result
              in loop[k].contents.whereType<FunctionResultContent>()) {
            final name =
                result.name ??
                _nameForCallId(calls, result.callId) ??
                'unknown';
            out.write(_formatToolResponse(name, result.result));
            emittedResponse = true;
            prevType = 'tool_response';
          }
        }
      }

      // Gemma 4 accepts both vision and audio through mtmd. Collect either kind
      // of media blob in content order and emit one marker apiece; mtmd
      // substitutes the model-specific image/audio tokens and auto-detects the
      // kind from each blob's magic bytes.
      final mediaMarkers = StringBuffer();
      for (final data in msg.contents.whereType<DataContent>()) {
        final bytes = data.data;
        if (bytes != null &&
            (data.hasTopLevelMediaType('image') ||
                data.hasTopLevelMediaType('audio'))) {
          media.add(bytes);
          mediaMarkers.write(mediaMarker);
        }
      }

      final base = isAssistant ? _stripThinking(msg.text) : msg.text.trim();
      final content = '$mediaMarkers$base';
      out.write(content);
      final hasContent = content.trim().isNotEmpty;

      if (prevType == 'tool_call' && !emittedResponse) {
        out.write(toolResponseOpen);
      } else if (!(emittedResponse && !hasContent)) {
        // Deliberate divergence from the upstream jinja: closing the turn
        // must also clear the mid-turn state. Upstream keeps
        // prev_message_type == 'tool_response' here, so a completed tool
        // round that also carries prose suppresses the trailing
        // `<|turn>model` and the prompt ends headerless after `<turn|>` —
        // the model then invents its own turn/channel markup, which leaks
        // into user-visible text.
        out.write('$turnClose\n');
        prevType = null;
      }
    }

    if (addGenerationPrompt &&
        prevType != 'tool_response' &&
        prevType != 'tool_call') {
      out.write(
        '$turnOpen'
        'model\n',
      );
    }

    return GemmaPrompt(
      text: out.toString(),
      stopSequences: stopSequences,
      media: media,
    );
  }

  /// Parses one raw generated model turn into prose plus any tool calls.
  ///
  /// [generated] is the text emitted by `LlamaFlutter.generate` for a single
  /// model turn — with the stop sequence already stripped, so any
  /// `<|tool_call>…<tool_call|>` blocks are complete. Thinking-channel content
  /// is removed. Synthetic sequential [FunctionCallContent.callId]s are
  /// assigned so results can be correlated downstream.
  GemmaTurn parse(String generated) {
    final cleaned = _stripThinking(generated);
    final calls = <FunctionCallContent>[];
    final text = StringBuffer();
    var cursor = 0;
    while (cursor < cleaned.length) {
      final open = cleaned.indexOf(toolCallOpen, cursor);
      if (open < 0) {
        text.write(cleaned.substring(cursor));
        break;
      }
      text.write(cleaned.substring(cursor, open));
      final close = cleaned.indexOf(toolCallClose, open);
      final bodyEnd = close < 0 ? cleaned.length : close;
      final block = cleaned.substring(open + toolCallOpen.length, bodyEnd);
      final call = _parseCall(block, calls.length);
      if (call != null) {
        calls.add(call);
      }
      cursor = close < 0 ? cleaned.length : close + toolCallClose.length;
    }
    return GemmaTurn(text: text.toString().trim(), calls: calls);
  }

  FunctionCallContent? _parseCall(String block, int index) {
    const prefix = 'call:';
    if (!block.startsWith(prefix)) {
      return null;
    }
    final brace = block.indexOf('{');
    if (brace < 0) {
      return null;
    }
    final name = block.substring(prefix.length, brace).trim();
    final args = _ArgumentParser(block, brace).parseObject();
    return FunctionCallContent(
      callId: 'call_$index',
      name: name,
      arguments: args,
    );
  }

  // --- helpers ---

  String _roleOf(ChatMessage m) => m.role.value;

  ChatMessage? _prevNonTool(List<ChatMessage> loop, int index) {
    for (var j = index - 1; j >= 0; j--) {
      if (_roleOf(loop[j]) != 'tool') {
        return loop[j];
      }
    }
    return null;
  }

  String? _nameForCallId(List<FunctionCallContent> calls, String callId) {
    for (final call in calls) {
      if (call.callId == callId) {
        return call.name;
      }
    }
    return null;
  }

  String _wrapString(String value) => '$stringDelimiter$value$stringDelimiter';

  /// Formats a value using the model's argument grammar (not JSON).
  ///
  /// Strings are wrapped in [stringDelimiter]; maps emit sorted keys (bare when
  /// [escapeKeys] is false, otherwise wrapped); lists are bracketed.
  String _formatArgument(Object? value, {required bool escapeKeys}) {
    if (value is String) {
      return _wrapString(value);
    }
    if (value is bool) {
      return value ? 'true' : 'false';
    }
    if (value is Map) {
      final keys = value.keys.map((k) => k.toString()).toList()..sort();
      final parts = keys.map((k) {
        final renderedKey = escapeKeys ? _wrapString(k) : k;
        return '$renderedKey:${_formatArgument(value[k], escapeKeys: escapeKeys)}';
      });
      return '{${parts.join(',')}}';
    }
    if (value is Iterable) {
      return '[${value.map((e) => _formatArgument(e, escapeKeys: escapeKeys)).join(',')}]';
    }
    return '$value';
  }

  /// Renders a `<|tool_response>…<tool_response|>` block for one result.
  String _formatToolResponse(String name, Object? result) {
    final body = StringBuffer(toolResponseOpen)..write('response:$name');
    if (result is Map) {
      body.write(_formatArgument(result, escapeKeys: false));
    } else {
      body.write('{value:${_formatArgument(result, escapeKeys: false)}}');
    }
    body.write(toolResponseClose);
    return body.toString();
  }

  /// Renders a tool declaration: `declaration:NAME{description:…,parameters:…}`.
  String _formatDeclaration(AIFunctionDeclaration tool) {
    final out = StringBuffer()
      ..write('declaration:${tool.name}')
      ..write('{description:${_wrapString(tool.description ?? '')}');

    final params = tool.parametersSchema;
    if (params != null && params.isNotEmpty) {
      out.write(',parameters:{');
      final props = params['properties'];
      if (props is Map && props.isNotEmpty) {
        out.write(
          'properties:{${_formatParameters(props, _requiredOf(params))}},',
        );
      }
      final required = _requiredOf(params);
      if (required.isNotEmpty) {
        out.write('required:[${required.map(_wrapString).join(',')}],');
      }
      final type = params['type'];
      if (type != null) {
        out.write('type:${_wrapString(type.toString().toUpperCase())}}');
      }
    }

    final response = tool.returnSchema;
    if (response != null) {
      out.write(',response:{');
      if (response['description'] != null) {
        out.write(
          'description:${_wrapString(response['description'].toString())},',
        );
      }
      if (response['type']?.toString().toUpperCase() == 'OBJECT') {
        out.write('type:${_wrapString('OBJECT')}}');
      }
    }

    out.write('}');
    return out.toString();
  }

  /// Renders a JSON-schema `properties` map, comma-separated, sorted by key.
  String _formatParameters(
    Map<dynamic, dynamic> properties,
    List<String> required,
  ) {
    final keys = properties.keys.map((k) => k.toString()).toList()..sort();
    return keys
        .map((key) {
          final value = properties[key];
          final spec = value is Map ? value : const <dynamic, dynamic>{};
          final type = (spec['type'] ?? '').toString().toUpperCase();
          final out = StringBuffer('$key:{');
          var first = true;
          String sep() {
            if (first) {
              first = false;
              return '';
            }
            return ',';
          }

          if (spec['description'] != null) {
            out.write(
              '${sep()}description:${_wrapString(spec['description'].toString())}',
            );
          }
          if (type == 'STRING' && spec['enum'] != null) {
            out.write(
              '${sep()}enum:${_formatArgument(spec['enum'], escapeKeys: true)}',
            );
          } else if (type == 'ARRAY' && spec['items'] is Map) {
            out.write('${sep()}items:{${_formatItems(spec['items'] as Map)}}');
          }
          if (spec['nullable'] == true) {
            out.write('${sep()}nullable:true');
          }
          if (type == 'OBJECT' && spec['properties'] is Map) {
            out.write(
              '${sep()}properties:{${_formatParameters(spec['properties'] as Map, _requiredOf(spec))}}',
            );
            final nestedRequired = _requiredOf(spec);
            if (nestedRequired.isNotEmpty) {
              out.write(
                '${sep()}required:[${nestedRequired.map(_wrapString).join(',')}]',
              );
            }
          }
          // Only emit `type:` when the schema actually has one. The upstream
          // jinja writes `type:<|"|><|"|>` unconditionally, so a typeless prop
          // there renders a malformed empty type; we deliberately diverge and
          // skip it. The closing `}` always closes `$key:{`.
          if (type.isNotEmpty) {
            out.write('${sep()}type:${_wrapString(type)}');
          }
          out.write('}');
          return out.toString();
        })
        .join(',');
  }

  /// Renders an array's `items` schema body (comma-separated, sorted).
  String _formatItems(Map<dynamic, dynamic> items) {
    final keys = items.keys.map((k) => k.toString()).toList()..sort();
    final parts = <String>[];
    for (final key in keys) {
      final value = items[key];
      if (value == null) {
        continue;
      }
      if (key == 'properties' && value is Map) {
        parts.add(
          'properties:{${_formatParameters(value, _requiredOf(items))}}',
        );
      } else if (key == 'required' && value is List) {
        parts.add(
          'required:[${value.map((r) => _wrapString(r.toString())).join(',')}]',
        );
      } else if (key == 'type') {
        if (value is List) {
          parts.add(
            'type:${_formatArgument(value.map((e) => e.toString().toUpperCase()).toList(), escapeKeys: true)}',
          );
        } else {
          parts.add('type:${_wrapString(value.toString().toUpperCase())}');
        }
      } else {
        parts.add('$key:${_formatArgument(value, escapeKeys: true)}');
      }
    }
    return parts.join(',');
  }

  List<String> _requiredOf(Map<dynamic, dynamic> schema) {
    final required = schema['required'];
    if (required is List) {
      return required.map((e) => e.toString()).toList();
    }
    return const <String>[];
  }

  /// Removes `<|channel>thought…<channel|>` segments and trims.
  String _stripThinking(String text) {
    final out = StringBuffer();
    for (final part in text.split(channelClose)) {
      final marker = part.indexOf(channelOpen);
      out.write(marker >= 0 ? part.substring(0, marker) : part);
    }
    return out.toString().trim();
  }
}

/// Recursive-descent reader for Gemma's non-JSON argument grammar.
///
/// Values are strings (`<|"|>…<|"|>`), numbers, `true`/`false`, arrays
/// (`[…]`), or nested objects (`{…}`). Keys are bare. Because strings are
/// delimiter-bounded, structural characters inside a string are never mistaken
/// for grammar (a string containing `,` or `}` round-trips correctly).
class _ArgumentParser {
  _ArgumentParser(this._source, this._pos);

  final String _source;
  int _pos;

  static const String _delimiter = GemmaChatTemplate.stringDelimiter;

  Map<String, Object?> parseObject() {
    _expect('{');
    final map = <String, Object?>{};
    if (_peek() == '}') {
      _pos++;
      return map;
    }
    while (true) {
      final key = _parseKey();
      _expect(':');
      map[key] = _parseValue();
      if (_peek() == ',') {
        _pos++;
        continue;
      }
      _expect('}');
      break;
    }
    return map;
  }

  String _parseKey() {
    final start = _pos;
    while (_pos < _source.length && _source[_pos] != ':') {
      _pos++;
    }
    return _source.substring(start, _pos);
  }

  Object? _parseValue() {
    if (_source.startsWith(_delimiter, _pos)) {
      return _parseString();
    }
    switch (_peek()) {
      case '{':
        return parseObject();
      case '[':
        return _parseArray();
      default:
        return _parseLiteral();
    }
  }

  String _parseString() {
    _pos += _delimiter.length;
    final end = _source.indexOf(_delimiter, _pos);
    if (end < 0) {
      throw FormatException(
        'Unterminated $_delimiter string at $_pos',
        _source,
        _pos,
      );
    }
    final value = _source.substring(_pos, end);
    _pos = end + _delimiter.length;
    return value;
  }

  List<Object?> _parseArray() {
    _expect('[');
    final list = <Object?>[];
    if (_peek() == ']') {
      _pos++;
      return list;
    }
    while (true) {
      list.add(_parseValue());
      if (_peek() == ',') {
        _pos++;
        continue;
      }
      _expect(']');
      break;
    }
    return list;
  }

  Object? _parseLiteral() {
    final start = _pos;
    while (_pos < _source.length && !',}]'.contains(_source[_pos])) {
      _pos++;
    }
    final raw = _source.substring(start, _pos).trim();
    if (raw == 'true') {
      return true;
    }
    if (raw == 'false') {
      return false;
    }
    return num.tryParse(raw) ?? raw;
  }

  String _peek() => _pos < _source.length ? _source[_pos] : '';

  void _expect(String char) {
    if (_peek() != char) {
      throw FormatException('Expected "$char" at $_pos', _source, _pos);
    }
    _pos++;
  }
}
