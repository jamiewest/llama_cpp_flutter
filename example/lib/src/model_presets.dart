import 'package:llama_cpp_flutter/llama_cpp_flutter.dart';

/// A ready-to-load model the user can pick without typing a URL.
///
/// All presets are small enough for the web runtime's single-file limit
/// (~2 GiB) and download from Hugging Face `resolve` URLs, which send the
/// CORS headers browsers need.
class ModelPreset {
  const ModelPreset({
    required this.id,
    required this.label,
    required this.detail,
    required this.repo,
    required this.file,
    required this.formatName,
    this.supportsThinking = false,
    this.supportsTools = false,
  });

  final String id;
  final String label;
  final String detail;
  final String repo;
  final String file;

  /// Name of a built-in chat format (see `chatFormatNames`).
  final String formatName;

  /// Whether the model family has a reasoning channel.
  final bool supportsThinking;

  /// Whether the model is trained for tool calling.
  final bool supportsTools;

  Uri get url => huggingFaceModelUri(repo: repo, file: file);
}

const List<ModelPreset> modelPresets = [
  ModelPreset(
    id: 'qwen3-0.6b-q8',
    label: 'Qwen3 0.6B',
    detail: 'Q8_0 · ~610 MB · thinking + tools',
    repo: 'Qwen/Qwen3-0.6B-GGUF',
    file: 'Qwen3-0.6B-Q8_0.gguf',
    formatName: 'qwen',
    supportsThinking: true,
    supportsTools: true,
  ),
  ModelPreset(
    id: 'llama-3.2-1b-q4km',
    label: 'Llama 3.2 1B Instruct',
    detail: 'Q4_K_M · ~770 MB · tools',
    repo: 'bartowski/Llama-3.2-1B-Instruct-GGUF',
    file: 'Llama-3.2-1B-Instruct-Q4_K_M.gguf',
    formatName: 'llama3',
    supportsTools: true,
  ),
  ModelPreset(
    id: 'gemma-3-1b-q4km',
    label: 'Gemma 3 1B IT',
    detail: 'Q4_K_M · ~770 MB',
    repo: 'ggml-org/gemma-3-1b-it-GGUF',
    file: 'gemma-3-1b-it-Q4_K_M.gguf',
    formatName: 'gemma',
  ),
  ModelPreset(
    id: 'smollm2-360m-q8',
    label: 'SmolLM2 360M Instruct',
    detail: 'Q8_0 · ~370 MB',
    repo: 'bartowski/SmolLM2-360M-Instruct-GGUF',
    file: 'SmolLM2-360M-Instruct-Q8_0.gguf',
    formatName: 'chatml',
  ),
  ModelPreset(
    id: 'lfm2-350m-q8',
    label: 'LFM2 350M',
    detail: 'Q8_0 · ~360 MB · tools',
    repo: 'LiquidAI/LFM2-350M-GGUF',
    file: 'LFM2-350M-Q8_0.gguf',
    formatName: 'lfm2',
    supportsTools: true,
  ),
];
