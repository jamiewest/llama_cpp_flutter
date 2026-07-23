import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:llama_cpp_flutter/llama_cpp_flutter.dart';

import '../app_model.dart';
import '../model_presets.dart';

/// Picks which model to run and how to configure the engine for it.
class ModelScreen extends StatefulWidget {
  const ModelScreen({super.key, required this.model});

  final AppModel model;

  @override
  State<ModelScreen> createState() => _ModelScreenState();
}

class _ModelScreenState extends State<ModelScreen> {
  late ModelPreset? _preset;
  late final TextEditingController _customUrl;
  late String _customFormatName;
  late int _contextSize;
  late bool _gpuOffload;
  late bool _enableThinking;

  ModelConfig get _config => widget.model.modelConfig;

  @override
  void initState() {
    super.initState();
    _preset = _config.preset;
    _customUrl = TextEditingController(text: _config.customUrl);
    _customFormatName = _config.customFormatName;
    _contextSize = _config.contextSize;
    _gpuOffload = _config.gpuLayers > 0;
    _enableThinking = _config.enableThinking;
  }

  @override
  void dispose() {
    _customUrl.dispose();
    super.dispose();
  }

  bool get _canLoad =>
      _preset != null ||
      Uri.tryParse(_customUrl.text.trim())?.hasScheme == true;

  void _load() {
    _config
      ..preset = _preset
      ..customUrl = _customUrl.text.trim()
      ..customFormatName = _customFormatName
      ..contextSize = _contextSize
      ..gpuLayers = _gpuOffload ? 999 : 0
      ..enableThinking = _enableThinking;
    widget.model.loadModel();
    Navigator.of(context).pop();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Model')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          Text('Model source', style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: 8),
          RadioGroup<ModelPreset?>(
            groupValue: _preset,
            onChanged: (value) => setState(() => _preset = value),
            child: Column(
              children: [
                for (final preset in modelPresets)
                  RadioListTile<ModelPreset?>(
                    value: preset,
                    title: Text(preset.label),
                    subtitle: Text(preset.detail),
                  ),
                const RadioListTile<ModelPreset?>(
                  value: null,
                  title: Text('Custom GGUF URL'),
                  subtitle: Text(
                    'Direct link to a .gguf file '
                    '(e.g. a Hugging Face resolve URL)',
                  ),
                ),
              ],
            ),
          ),
          if (_preset == null) ...[
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: TextField(
                controller: _customUrl,
                decoration: const InputDecoration(
                  labelText: 'Model URL',
                  hintText: 'https://huggingface.co/…/resolve/main/model.gguf',
                ),
                keyboardType: TextInputType.url,
                onChanged: (_) => setState(() {}),
              ),
            ),
            const SizedBox(height: 8),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: DropdownButtonFormField<String>(
                initialValue: _customFormatName,
                decoration: const InputDecoration(labelText: 'Chat format'),
                items: [
                  for (final name in supportedChatFormatNames)
                    DropdownMenuItem(value: name, child: Text(name)),
                ],
                onChanged: (value) =>
                    setState(() => _customFormatName = value ?? 'chatml'),
              ),
            ),
          ],
          const Divider(height: 32),
          Text('Engine', style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: 8),
          DropdownButtonFormField<int>(
            initialValue: _contextSize,
            decoration: const InputDecoration(
              labelText: 'Context size (tokens)',
            ),
            items: const [
              DropdownMenuItem(value: 2048, child: Text('2048')),
              DropdownMenuItem(value: 4096, child: Text('4096')),
              DropdownMenuItem(value: 8192, child: Text('8192')),
              DropdownMenuItem(value: 16384, child: Text('16384')),
            ],
            onChanged: (value) => setState(() => _contextSize = value ?? 4096),
          ),
          if (!kIsWeb)
            SwitchListTile(
              contentPadding: EdgeInsets.zero,
              value: _gpuOffload,
              onChanged: (value) => setState(() => _gpuOffload = value),
              title: const Text('GPU offload (Metal)'),
              subtitle: const Text('Run all layers on the GPU'),
            ),
          SwitchListTile(
            contentPadding: EdgeInsets.zero,
            value: _enableThinking,
            onChanged: (value) => setState(() => _enableThinking = value),
            title: const Text('Thinking'),
            subtitle: const Text(
              'Request the reasoning channel on models that have one '
              '(e.g. Qwen3); ignored otherwise',
            ),
          ),
          if (kIsWeb) ...[
            const Divider(height: 32),
            Card(
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: Text(
                  'Web notes: models stream straight from the URL and are '
                  'cached by the browser. Multi-threading requires the page '
                  'to be served cross-origin isolated (COOP/COEP headers); '
                  'see the example README.',
                  style: Theme.of(context).textTheme.bodySmall,
                ),
              ),
            ),
          ],
          const SizedBox(height: 24),
          FilledButton.icon(
            icon: const Icon(Icons.download),
            label: const Text('Load model'),
            onPressed: _canLoad ? _load : null,
          ),
          const SizedBox(height: 8),
          Text(
            'Loading a model replaces the current session and starts a new '
            'chat. Downloads are cached, so a model only downloads once.',
            style: Theme.of(context).textTheme.bodySmall,
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }
}
