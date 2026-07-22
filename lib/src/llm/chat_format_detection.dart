/// Best-effort mapping from a model's own GGUF metadata to a registered
/// chat-format name.
///
/// Chat formats must track the model actually loaded, not the file name it
/// shipped under. A GGUF carries the two signals needed to pick one:
/// `general.architecture` and the embedded `tokenizer.chat_template`. Hosts
/// read the file's header prefix, call [detectChatFormatNameForGguf], and
/// feed the result to `resolveChatFormat` — typically through
/// `LlamaChatClient.formatResolver`, which is read per request after the
/// model file is on disk.
library;

import 'dart:typed_data';

import '../runtime/gguf_metadata.dart';

/// Detects the chat-format name for the GGUF whose first bytes are
/// [headerPrefix], or null when the header is unreadable or matches no
/// known family.
///
/// GGUF writers emit `general.*` keys early and the tokenizer keys after
/// them, so a prefix of a few MiB is normally enough; when the header is
/// larger the caller should retry with a bigger prefix (absence is
/// indistinguishable from truncation only past the key/value section).
String? detectChatFormatNameForGguf(Uint8List headerPrefix) {
  final result = readGgufMetadata(
    headerPrefix: headerPrefix,
    keys: const <String>{ggufArchitectureKey, ggufChatTemplateKey},
  );
  if (result is! GgufMetadata) return null;
  return detectChatFormatName(
    architecture: result.values[ggufArchitectureKey],
    chatTemplate: result.values[ggufChatTemplateKey],
  );
}

/// Maps GGUF metadata values to a registered chat-format name, or null when
/// neither signal identifies a known family.
///
/// The embedded [chatTemplate] is consulted first — it describes the wire
/// format directly, and distinguishes families that share an architecture
/// string (Mistral and Llama GGUFs both say `llama`). [architecture] breaks
/// the remaining ties (e.g. a Gemma file whose template was stripped).
String? detectChatFormatName({String? architecture, String? chatTemplate}) {
  final template = chatTemplate ?? '';
  if (template.isNotEmpty) {
    if (template.contains('<start_of_turn>') || template.contains('<|turn>')) {
      return 'gemma';
    }
    if (template.contains('<|start_header_id|>')) return 'llama3';
    if (template.contains('[INST]')) return 'mistral';
    if (template.contains('<|im_start|>')) {
      // Several families build on ChatML's turn markers; their tool-calling
      // conventions tell them apart. LFM2 tags tool sections with dedicated
      // markers; Qwen wraps tool results in <tool_response> and (Qwen3)
      // handles the <think> channel.
      if (template.contains('<|tool_list_start|>') ||
          template.contains('<|tool_call_start|>')) {
        return 'lfm2';
      }
      return template.contains('<tool_response>') ||
              template.contains('enable_thinking')
          ? 'qwen'
          : 'chatml';
    }
  }

  final arch = (architecture ?? '').toLowerCase();
  if (arch.startsWith('gemma')) return 'gemma';
  if (arch.startsWith('qwen')) return 'qwen';
  if (arch.startsWith('lfm2')) return 'lfm2';
  // 'llama' is ambiguous (Llama, Mistral, and many fine-tunes all use it);
  // without a template to disambiguate, no guess is safer than a wrong one.
  return null;
}
