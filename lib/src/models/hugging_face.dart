/// Hugging Face URL helpers for model artifacts.
library;

/// The direct-download URI for [file] in Hugging Face repository [repo]
/// (e.g. `'unsloth/gemma-3-4b-it-GGUF'`) at [revision].
///
/// This is the `resolve` endpoint, which serves the raw file bytes and is
/// what `ModelSpec.modelUrl` and the model downloader expect. It works on
/// both native (downloaded to disk) and web (streamed; Hugging Face sends
/// the CORS headers browsers need).
Uri huggingFaceModelUri({
  required String repo,
  required String file,
  String revision = 'main',
}) => Uri.https('huggingface.co', '/$repo/resolve/$revision/$file');
