import 'package:agents/agents.dart';
import 'package:flutter/foundation.dart';
import 'package:llama_cpp_flutter/llama_cpp_flutter.dart';
import 'package:path_provider/path_provider.dart';

import 'demo_tools.dart';
import 'llama_agent_provider.dart';
import 'model_presets.dart';

/// What the model screen edits. Applied on "Load model".
class ModelConfig {
  ModelPreset? preset = modelPresets.first;

  /// Direct GGUF URL, used when [preset] is null.
  String customUrl = '';

  /// Chat format for custom URLs (presets know their own).
  String customFormatName = 'chatml';

  int contextSize = 4096;
  int gpuLayers = 999;
  bool enableThinking = true;
}

/// What the agent screen edits. Applied immediately; the transcript is kept.
class AgentConfig {
  String name = 'Assistant';
  String instructions =
      'You are a helpful assistant running fully on-device. '
      'Keep answers concise.';
  double temperature = 0.7;
  int maxTokens = 1024;
  int? topK;
  double? topP;
  bool enableTools = true;
}

enum ModelStatus { empty, loading, ready, error }

/// Owns the runtime, the loaded session, and the agent built on top of it.
class AppModel extends ChangeNotifier {
  final ModelConfig modelConfig = ModelConfig();
  final AgentConfig agentConfig = AgentConfig();

  late final LlamaAgentProvider provider = LlamaAgentProvider(
    agentResolver: () => _agent,
  );

  LlamaRuntime? _runtime;
  LlamaSession? _session;
  ModelSpec? _loadedSpec;
  AIAgent? _agent;

  ModelStatus status = ModelStatus.empty;
  double loadProgress = 0;
  String? errorMessage;

  /// Bumped per successful load so the chat view resets with the new model.
  int modelGeneration = 0;

  String? get loadedModelName => _loadedSpec?.displayName;

  /// False on web when the page is not cross-origin isolated, in which case
  /// wllama runs single-threaded (slow enough to look like a hang).
  bool get multiThreaded => _runtime?.supportsMultiThreading ?? true;

  Future<LlamaRuntime> _ensureRuntime() async {
    if (_runtime case final runtime?) return runtime;
    String? cacheDir;
    if (!kIsWeb) {
      final support = await getApplicationSupportDirectory();
      cacheDir = '${support.path}/models';
    }
    return _runtime = createLlamaRuntime(artifactCacheDirectory: cacheDir);
  }

  ModelSpec _buildSpec() {
    final config = modelConfig;
    final preset = config.preset;
    final url = preset?.url ?? Uri.parse(config.customUrl.trim());
    final formatName = preset?.formatName ?? config.customFormatName;
    final format = resolveChatFormat(formatName) ?? resolveChatFormat('chatml')!;
    return ModelSpec(
      id: preset?.id ?? 'custom',
      displayName: preset?.label ??
          (url.pathSegments.isNotEmpty ? url.pathSegments.last : 'Custom model'),
      modelUrl: url,
      contextSize: config.contextSize,
      gpuLayers: config.gpuLayers,
      format: format,
      enableThinking: config.enableThinking,
    );
  }

  /// Downloads (or reuses the cached copy of) the configured model and swaps
  /// the live session over to it.
  Future<void> loadModel() async {
    if (status == ModelStatus.loading) return;
    status = ModelStatus.loading;
    loadProgress = 0;
    errorMessage = null;
    notifyListeners();
    try {
      final runtime = await _ensureRuntime();
      final previous = _session;
      _session = null;
      _agent = null;
      _loadedSpec = null;
      await previous?.dispose();

      final spec = _buildSpec();
      final session = await runtime.loadModel(
        spec,
        onProgress: (progress) {
          loadProgress = progress;
          notifyListeners();
        },
      );
      _session = session;
      _loadedSpec = spec;
      _rebuildAgent();
      provider.clear();
      modelGeneration++;
      status = ModelStatus.ready;
    } catch (error) {
      errorMessage = '$error';
      status = ModelStatus.error;
    }
    notifyListeners();
  }

  /// Applies agent-screen changes to the live agent; the transcript is kept.
  void applyAgentConfig() {
    if (_session != null) _rebuildAgent();
    notifyListeners();
  }

  void _rebuildAgent() {
    final spec = _loadedSpec!;
    final config = agentConfig;
    final chatClient = createLlamaChatClient(
      spec: spec,
      sessionProvider: () async => _session!,
      samplingOverride: SamplingDefaults(
        maxTokens: config.maxTokens,
        temperature: config.temperature,
        topK: config.topK,
        topP: config.topP,
      ),
      isThinkingEnabled: () => modelConfig.enableThinking,
    );
    _agent = ChatClientAgent.withSettings(
      chatClient,
      name: config.name,
      instructions:
          config.instructions.trim().isEmpty ? null : config.instructions.trim(),
      tools: config.enableTools ? buildDemoTools() : null,
    );
  }

  @override
  void dispose() {
    _session?.dispose();
    provider.dispose();
    super.dispose();
  }
}
