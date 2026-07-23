import 'dart:typed_data';

import 'package:llama_cpp_flutter/chat.dart';
import 'package:llama_cpp_flutter/llama_cpp_flutter.dart';
import 'package:llama_cpp_flutter/src/runtime/llama_runtime_native.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:llama_cpp_flutter/bridge.dart' as native;

final class _FakeNativeSession implements native.LlamaBridgeSession {
  _FakeNativeSession({this.stats});

  /// Stats reported through `onStats` when set.
  final native.LlamaGenerationStats? stats;

  /// Budget received by [setImageTokenBudget], recorded for assertions.
  int? imageTokenBudget;
  bool imageTokenBudgetSet = false;

  @override
  int get id => 1;

  @override
  int get maxSequences => 1;

  @override
  Stream<String> generate(
    String prompt, {
    int maxTokens = 256,
    double temperature = 0.8,
    int? topK,
    double? topP,
    int? seed,
    List<String> stopSequences = const <String>[],
    List<Uint8List>? images,
    int sequenceId = 0,
    void Function(native.LlamaGenerationStats)? onStats,
  }) {
    final reported = stats;
    if (reported != null) onStats?.call(reported);
    return const Stream<String>.empty();
  }

  @override
  Future<void> cancel() async {}

  @override
  Future<int> saveState(String path, {int sequenceId = 0}) async => 0;

  @override
  Future<int> loadState(String path, {int sequenceId = 0}) async => 0;

  @override
  Future<int> stateSizeBytes({int sequenceId = 0}) async => 0;

  @override
  Future<({int tokens, int bytes})> stashState(
    String key, {
    int sequenceId = 0,
  }) async => (tokens: 0, bytes: 0);

  @override
  Future<int> restoreStashedState(String key, {int sequenceId = 0}) async => 0;

  @override
  Future<int> dropStashedState(String key) async => 0;

  @override
  Future<void> clearSequence(int sequenceId) async {}

  @override
  Future<void> setImageTokenBudget(int? imageTokenBudget) async {
    this.imageTokenBudget = imageTokenBudget;
    imageTokenBudgetSet = true;
  }

  @override
  Future<void> dispose() async {}
}

final class _RecordingLlamaCppFlutter implements native.LlamaCppFlutter {
  _RecordingLlamaCppFlutter({this.stats});

  /// Stats the returned session reports through `onStats`, when set.
  final native.LlamaGenerationStats? stats;

  /// The session handed out by [loadModel], for post-load assertions.
  _FakeNativeSession? session;

  String? path;
  int? contextSize;
  int? gpuLayers;
  String? mmprojPath;
  int? imageTokenBudget;
  String? draftModelPath;
  int? draftGpuLayers;
  int? maxDraftTokens;

  int? maxSequences;

  @override
  Future<native.LlamaBridgeSession> loadModel(
    String path, {
    int contextSize = 4096,
    int gpuLayers = 999,
    String? mmprojPath,
    int? imageTokenBudget,
    String? draftModelPath,
    int draftGpuLayers = 999,
    int maxDraftTokens = 8,
    int maxSequences = 1,
  }) async {
    this.path = path;
    this.contextSize = contextSize;
    this.gpuLayers = gpuLayers;
    this.mmprojPath = mmprojPath;
    this.imageTokenBudget = imageTokenBudget;
    this.draftModelPath = draftModelPath;
    this.draftGpuLayers = draftGpuLayers;
    this.maxDraftTokens = maxDraftTokens;
    this.maxSequences = maxSequences;
    return session = _FakeNativeSession(stats: stats);
  }

  @override
  Future<void> shutdown() async {}
}

ModelSpec _spec({
  int draftGpuLayers = 999,
  int maxDraftTokens = 8,
  int? imageTokenBudget,
}) => ModelSpec(
  id: 'test-model',
  displayName: 'Test model',
  modelUrl: Uri.parse('https://example.com/model.gguf'),
  imageTokenBudget: imageTokenBudget,
  contextSize: 2048,
  gpuLayers: 12,
  draftGpuLayers: draftGpuLayers,
  maxDraftTokens: maxDraftTokens,
  format: const Lfm2ChatFormat(),
);

void main() {
  group('NativeLlamaRuntime.loadModel', () {
    test('forwards projector, draft path, and draft tuning', () async {
      final llama = _RecordingLlamaCppFlutter();
      final runtime = NativeLlamaRuntime(llama: llama);

      await runtime.loadModel(
        _spec(draftGpuLayers: 4, maxDraftTokens: 16, imageTokenBudget: 1120),
        localPath: '/models/main.gguf',
        localMmprojPath: '/models/mmproj.gguf',
        localDraftPath: '/models/draft.gguf',
      );

      expect(llama.path, '/models/main.gguf');
      expect(llama.contextSize, 2048);
      expect(llama.gpuLayers, 12);
      expect(llama.mmprojPath, '/models/mmproj.gguf');
      expect(llama.imageTokenBudget, 1120);
      expect(llama.draftModelPath, '/models/draft.gguf');
      expect(llama.draftGpuLayers, 4);
      expect(llama.maxDraftTokens, 16);
    });

    test('omitted artifacts forward null and keep text-only load', () async {
      final llama = _RecordingLlamaCppFlutter();
      final runtime = NativeLlamaRuntime(llama: llama);

      await runtime.loadModel(_spec(), localPath: '/models/main.gguf');

      expect(llama.mmprojPath, isNull);
      expect(llama.imageTokenBudget, isNull);
      expect(llama.draftModelPath, isNull);
      expect(llama.draftGpuLayers, 999);
      expect(llama.maxDraftTokens, 8);
    });

    test('forwards runtime image-budget changes to the plugin', () async {
      final llama = _RecordingLlamaCppFlutter();
      final runtime = NativeLlamaRuntime(llama: llama);
      final session = await runtime.loadModel(
        _spec(),
        localPath: '/models/main.gguf',
      );

      expect(session.capabilities.canSetImageTokenBudget, isTrue);

      await session.setImageTokenBudget(560);
      expect(llama.session!.imageTokenBudget, 560);

      await session.setImageTokenBudget(null);
      expect(llama.session!.imageTokenBudget, isNull);
      expect(llama.session!.imageTokenBudgetSet, isTrue);
    });

    test('requires a local model path', () async {
      final runtime = NativeLlamaRuntime(llama: _RecordingLlamaCppFlutter());

      expect(() => runtime.loadModel(_spec()), throwsArgumentError);
    });
  });

  group('NativeLlamaRuntime stats', () {
    test('maps native stats onto the engine-neutral shape', () async {
      final llama = _RecordingLlamaCppFlutter(
        stats: const native.LlamaGenerationStats(
          promptTokenCount: 10,
          cachedTokenCount: 4,
          generatedTokenCount: 6,
          finishReason: native.LlamaFinishReason.contextFull,
          prefillDuration: Duration(milliseconds: 80),
          decodeDuration: Duration(milliseconds: 300),
          draftedTokenCount: 9,
          acceptedTokenCount: 5,
        ),
      );
      final runtime = NativeLlamaRuntime(llama: llama);
      final session = await runtime.loadModel(
        _spec(),
        localPath: '/models/main.gguf',
      );

      LlamaGenerationStats? stats;
      await session.generate('Hi', onStats: (s) => stats = s).drain<void>();

      expect(stats, isNotNull);
      expect(stats!.promptTokenCount, 10);
      expect(stats!.cachedTokenCount, 4);
      expect(stats!.generatedTokenCount, 6);
      expect(stats!.finishReason, LlamaFinishReason.contextFull);
      expect(stats!.prefillDuration, const Duration(milliseconds: 80));
      expect(stats!.decodeDuration, const Duration(milliseconds: 300));
      expect(stats!.draftedTokenCount, 9);
      expect(stats!.acceptedTokenCount, 5);
    });
  });
}
