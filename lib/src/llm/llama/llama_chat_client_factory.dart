/// Composition helpers for the llama.cpp inference engine.
library;

import 'package:extensions/ai.dart';
import 'package:extensions/logging.dart';

import '../../diagnostics/prompt_inspector.dart';
import '../../models/model_spec.dart';
import '../chat_format.dart';
import 'llama_chat_client.dart';

/// Builds the llama.cpp-backed [ChatClient] for [spec].
///
/// This is the one place llama-specific client construction happens: the
/// spec's [ModelSpec.format] drives prompt rendering/decoding and its
/// [ModelSpec.sampling] supplies generation defaults. The model itself is
/// loaded elsewhere (a hosted service) and resolved through
/// [sessionProvider].
///
/// [isThinkingEnabled], when supplied, is read per request to decide whether to
/// request the family's reasoning channel; it is ignored for formats without
/// one. [inspector], when supplied, records each rendered prompt so the UI can
/// show exactly what was sent to the model.
///
/// [formatResolver], when supplied, is read per request after the session
/// resolves and overrides the spec's format when it returns non-null; use it
/// to defer the format choice until the model file itself has been
/// inspected.
ChatClient createLlamaChatClient({
  required ModelSpec spec,
  required SessionProvider sessionProvider,
  int? contextSizeOverride,
  SamplingDefaults? samplingOverride,
  ChatFormat? Function()? formatResolver,
  bool Function()? isThinkingEnabled,
  PromptInspector? inspector,
  LoggerFactory? loggerFactory,
}) => LlamaChatClient(
  sessionProvider: sessionProvider,
  format: spec.format,
  formatResolver: formatResolver,
  contextSize: contextSizeOverride ?? spec.contextSize,
  sampling: samplingOverride ?? spec.sampling,
  imageTiling: spec.imageTiling,
  inspector: inspector,
  isThinkingEnabled: isThinkingEnabled,
  loggerFactory: loggerFactory,
);
