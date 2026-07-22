/// Resolves a configured `llama.format` name to its [ChatFormat].
///
/// This is the single source of truth shared by the host app (which builds a
/// chat client) and its model-editor UI (which validates the name), so the two
/// can never drift. Add a new family here and both pick it up — or, for a
/// family this package doesn't know about, register it at startup with
/// [registerChatFormat] without modifying the package.
library;

import 'chat_format.dart';
import 'chatml/chatml_chat_format.dart';
import 'gemma/gemma_chat_format.dart';
import 'lfm2/lfm2_chat_format.dart';
import 'lfm2/lfm2_chat_template.dart';
import 'llama3/llama3_chat_format.dart';
import 'mistral/mistral_chat_format.dart';
import 'qwen/qwen_chat_format.dart';

/// The name used when `llama.format` is unset, for backwards compatibility.
const String defaultChatFormatName = 'gemma';

/// All known `llama.format` names mapped to their [ChatFormat].
///
/// `lfm2` and `lfm2-vl` use LFM2's tagged tool-list/tool-response style.
/// `lfm2.5`/`lfm25` aliases use LFM2.5's plain JSON tool style.
const Map<String, ChatFormat> _builtInFormats = <String, ChatFormat>{
  'gemma': GemmaChatFormat(),
  'lfm2': Lfm2ChatFormat(),
  'lfm2-vl': Lfm2ChatFormat(),
  'lfm2.5': Lfm2ChatFormat(toolTagStyle: LfmToolTagStyle.lfm25),
  'lfm2.5-vl': Lfm2ChatFormat(toolTagStyle: LfmToolTagStyle.lfm25),
  'lfm25': Lfm2ChatFormat(toolTagStyle: LfmToolTagStyle.lfm25),
  'lfm25-vl': Lfm2ChatFormat(toolTagStyle: LfmToolTagStyle.lfm25),
  'chatml': ChatmlChatFormat(),
  'llama3': Llama3ChatFormat(),
  'mistral': MistralChatFormat(),
  'qwen': QwenChatFormat(),
};

/// App-registered formats, consulted before the built-ins so an app can both
/// add new families and override a built-in rendering.
final Map<String, ChatFormat> _registeredFormats = <String, ChatFormat>{};

/// Registers [format] under [name] so [resolveChatFormat] can produce it.
///
/// Call once at startup for each model family this package doesn't ship —
/// e.g. a newly released model with its own wire format. Registering an
/// existing name (built-in or previously registered) replaces it.
void registerChatFormat(String name, ChatFormat format) {
  final key = name.trim();
  if (key.isEmpty) {
    throw ArgumentError.value(name, 'name', 'must not be empty');
  }
  _registeredFormats[key] = format;
}

/// The set of accepted `llama.format` names (an empty string is also accepted
/// and resolves to [defaultChatFormatName]).
Set<String> get supportedChatFormatNames => <String>{
  ..._builtInFormats.keys,
  ..._registeredFormats.keys,
};

/// Resolves [name] to a [ChatFormat], or `null` when [name] is unknown.
///
/// A null or empty [name] resolves to [defaultChatFormatName]. Formats
/// registered with [registerChatFormat] take precedence over built-ins.
ChatFormat? resolveChatFormat(String? name) {
  final key = (name == null || name.isEmpty) ? defaultChatFormatName : name;
  return _registeredFormats[key] ?? _builtInFormats[key];
}
