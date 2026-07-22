/// Captures the exact payload sent to the model so the UI can show it.
library;

import 'package:flutter/foundation.dart';

/// An immutable record of one rendered prompt and the sampling configuration
/// it was generated with.
///
/// [text] is the fully rendered, wire-format prompt — for the Gemma format it
/// already contains the system instructions, tool declarations, and message
/// history, so it answers "what is the model provided with" on its own. The
/// sampling fields are the values actually passed to `generate`, after any
/// per-request `ChatOptions` overrides were applied.
@immutable
class PromptSnapshot {
  const PromptSnapshot({
    required this.text,
    required this.stopSequences,
    required this.maxTokens,
    required this.temperature,
    required this.topK,
    required this.topP,
    required this.seed,
    required this.imageCount,
    required this.contextSize,
    required this.capturedAt,
  });

  /// The rendered wire-format prompt handed to the model.
  final String text;

  /// Stop sequences passed alongside the prompt.
  final List<String> stopSequences;

  /// Resolved sampling values actually sent to `generate`.
  final int maxTokens;
  final double temperature;
  final int? topK;
  final double? topP;
  final int? seed;

  /// Number of image attachments sent with this prompt.
  ///
  /// Note: image attachments add real context tokens that are **not** present
  /// in [text], so any token estimate derived from [text] reads low when this
  /// is non-zero.
  final int imageCount;

  /// The model's context window in tokens, used to gauge how full the context
  /// is from an estimated count of [text].
  final int contextSize;

  /// When the snapshot was captured.
  final DateTime capturedAt;
}

/// Holds the most recently rendered [PromptSnapshot] and notifies listeners
/// when it changes.
///
/// Written to at the render seam in `LlamaChatClient` and read by the in-app
/// prompt viewer. Latest-only by design: in a tool-calling turn the prompt is
/// re-rendered after each tool result, and this keeps the last one sent.
class PromptInspector extends ChangeNotifier {
  PromptSnapshot? _latest;

  /// The most recent prompt sent to the model, or null if none yet.
  PromptSnapshot? get latest => _latest;

  /// Records [snapshot] as the latest prompt and notifies listeners.
  void record(PromptSnapshot snapshot) {
    _latest = snapshot;
    notifyListeners();
  }
}
