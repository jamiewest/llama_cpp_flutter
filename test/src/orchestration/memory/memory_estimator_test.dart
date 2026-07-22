import 'package:flutter_test/flutter_test.dart';
import 'package:llama_flutter/llama_flutter.dart';

void main() {
  const architecture = 'gemma3';
  const metadata = <String, int>{
    'gemma3.block_count': 34,
    'gemma3.embedding_length': 2560,
    'gemma3.attention.head_count': 8,
    'gemma3.attention.head_count_kv': 4,
    'gemma3.attention.key_length': 256,
    'gemma3.attention.value_length': 256,
    'gemma3.context_length': 32768,
  };

  group('estimateModelMemory', () {
    test('computes KV bytes per token from the attention shape', () {
      final estimate = estimateModelMemory(
        architecture: architecture,
        metadata: metadata,
        modelFileSizeBytes: 3000000000,
      );

      expect(estimate, isNotNull);
      // 34 layers x 4 KV heads x (256 + 256) x 2 bytes (f16).
      expect(estimate!.kvBytesPerToken, 34 * 4 * 512 * 2);
      expect(estimate.weightsBytes, 3000000000);
      expect(estimate.trainedContextTokens, 32768);
    });

    test('adds the draft model to the weight cost', () {
      final estimate = estimateModelMemory(
        architecture: architecture,
        metadata: metadata,
        modelFileSizeBytes: 3000000000,
        draftFileSizeBytes: 250000000,
      );

      expect(estimate!.weightsBytes, 3250000000);
    });

    test('falls back to full MHA when KV head count is missing', () {
      final reduced = Map<String, int>.from(metadata)
        ..remove('gemma3.attention.head_count_kv');

      final estimate = estimateModelMemory(
        architecture: architecture,
        metadata: reduced,
        modelFileSizeBytes: 1,
      );

      expect(estimate!.kvBytesPerToken, 34 * 8 * 512 * 2);
    });

    test('derives head dimensions when key/value lengths are missing', () {
      final reduced = Map<String, int>.from(metadata)
        ..remove('gemma3.attention.key_length')
        ..remove('gemma3.attention.value_length');

      final estimate = estimateModelMemory(
        architecture: architecture,
        metadata: reduced,
        modelFileSizeBytes: 1,
      );

      // Head dimension = 2560 / 8 = 320 for both K and V.
      expect(estimate!.kvBytesPerToken, 34 * 4 * (320 + 320) * 2);
    });

    test('returns null when core hyperparameters are missing', () {
      expect(
        estimateModelMemory(
          architecture: architecture,
          metadata: const {'gemma3.embedding_length': 2560},
          modelFileSizeBytes: 1,
        ),
        isNull,
      );
    });
  });

  group('ModelMemoryEstimate', () {
    const estimate = ModelMemoryEstimate(
      weightsBytes: 1000,
      kvBytesPerToken: 10,
      fixedOverheadBytes: 100,
    );

    test('bytesForContext is linear in tokens', () {
      expect(estimate.bytesForContext(0), 1100);
      expect(estimate.bytesForContext(50), 1600);
    });

    test('maxContextForBudget inverts bytesForContext', () {
      expect(estimate.maxContextForBudget(1600), 50);
      expect(estimate.maxContextForBudget(1100), 0);
      expect(estimate.maxContextForBudget(0), 0);
    });
  });
}
