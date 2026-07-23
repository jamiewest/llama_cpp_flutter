import 'dart:math';

import 'package:extensions/ai.dart';

/// Small local tools that make tool-calling visible in the demo without any
/// network or platform dependencies. Try "what time is it?" or
/// "roll a 20-sided die" with a tool-capable model (Qwen3, Llama 3.2, LFM2).
List<AITool> buildDemoTools() => [
  AIFunctionFactory.create(
    name: 'current_time',
    description: 'Returns the current local date and time.',
    parametersSchema: {'type': 'object', 'properties': <String, Object?>{}},
    callback: (arguments, {cancellationToken}) async =>
        DateTime.now().toIso8601String(),
  ),
  AIFunctionFactory.create(
    name: 'roll_dice',
    description:
        'Rolls a die with the given number of sides and returns the result.',
    parametersSchema: {
      'type': 'object',
      'properties': {
        'sides': {
          'type': 'integer',
          'description': 'Number of sides on the die.',
          'minimum': 2,
        },
      },
      'required': ['sides'],
    },
    callback: (arguments, {cancellationToken}) async {
      final sides = (arguments['sides'] as num?)?.toInt() ?? 6;
      return Random().nextInt(sides) + 1;
    },
  ),
];
