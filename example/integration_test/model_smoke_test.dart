import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:llama_cpp_flutter/bridge.dart';

/// End-to-end smoke test against the vendored llama.xcframework: loads a real
/// GGUF model and streams a short generation, exercising model load, the
/// Pigeon bridge, sampling, and the token EventChannel.
///
/// Run on a desktop device with a model path baked in:
///
///   flutter test integration_test/model_smoke_test.dart -d macos \
///     --dart-define=MODEL_PATH=/absolute/path/to/model.gguf
void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  const modelPath = String.fromEnvironment('MODEL_PATH');

  test('loads a model and generates tokens', skip: modelPath.isEmpty
      ? 'Pass --dart-define=MODEL_PATH=/path/to/model.gguf'
      : false, () async {
    final llama = LlamaCppFlutter();
    final session = await llama.loadModel(
      modelPath,
      contextSize: 512,
      gpuLayers: 0,
    );

    final output = StringBuffer();
    LlamaGenerationStats? stats;
    await for (final token in session.generate(
      'Once upon a time',
      maxTokens: 32,
      temperature: 0.8,
      seed: 42,
      onStats: (s) => stats = s,
    )) {
      output.write(token);
    }

    expect(output.toString(), isNotEmpty);
    expect(stats, isNotNull);
    expect(stats!.promptTokenCount, greaterThan(0));
    expect(stats!.generatedTokenCount, greaterThan(0));
    expect(stats!.prefillDuration, greaterThan(Duration.zero));
    expect(stats!.decodeDuration, greaterThan(Duration.zero));
    expect(
      stats!.finishReason,
      anyOf(LlamaFinishReason.maxTokens, LlamaFinishReason.eogToken),
    );

    await session.dispose();
    await llama.shutdown();
  });
}
