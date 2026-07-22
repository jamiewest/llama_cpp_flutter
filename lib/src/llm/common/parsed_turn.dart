/// The parsed result of one generated model turn, shared across chat formats.
library;

import 'package:extensions/ai.dart';

/// Prose plus any tool calls extracted from a single generated model turn.
class ParsedTurn {
  /// Creates a [ParsedTurn].
  const ParsedTurn({required this.text, required this.calls});

  /// User-visible prose, with any tool-call markup removed.
  final String text;

  /// Tool calls the model requested, in emission order. Empty for a plain
  /// answer.
  final List<FunctionCallContent> calls;
}
