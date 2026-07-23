import 'package:flutter/material.dart';

import '../app_model.dart';

/// Configures the agent that sits on top of the loaded model: persona,
/// sampling, and tools. Changes apply immediately and keep the transcript.
class AgentScreen extends StatefulWidget {
  const AgentScreen({super.key, required this.model});

  final AppModel model;

  @override
  State<AgentScreen> createState() => _AgentScreenState();
}

class _AgentScreenState extends State<AgentScreen> {
  late final TextEditingController _name;
  late final TextEditingController _instructions;
  late final TextEditingController _topK;
  late final TextEditingController _topP;
  late double _temperature;
  late int _maxTokens;
  late bool _enableTools;

  AgentConfig get _config => widget.model.agentConfig;

  @override
  void initState() {
    super.initState();
    _name = TextEditingController(text: _config.name);
    _instructions = TextEditingController(text: _config.instructions);
    _topK = TextEditingController(text: _config.topK?.toString() ?? '');
    _topP = TextEditingController(text: _config.topP?.toString() ?? '');
    _temperature = _config.temperature;
    _maxTokens = _config.maxTokens;
    _enableTools = _config.enableTools;
  }

  @override
  void dispose() {
    _name.dispose();
    _instructions.dispose();
    _topK.dispose();
    _topP.dispose();
    super.dispose();
  }

  void _apply() {
    _config
      ..name = _name.text.trim().isEmpty ? 'Assistant' : _name.text.trim()
      ..instructions = _instructions.text
      ..temperature = _temperature
      ..maxTokens = _maxTokens
      ..topK = int.tryParse(_topK.text.trim())
      ..topP = double.tryParse(_topP.text.trim())
      ..enableTools = _enableTools;
    widget.model.applyAgentConfig();
    Navigator.of(context).pop();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Agent')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          Text('Persona', style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: 8),
          TextField(
            controller: _name,
            decoration: const InputDecoration(labelText: 'Name'),
          ),
          const SizedBox(height: 12),
          TextField(
            controller: _instructions,
            decoration: const InputDecoration(
              labelText: 'Instructions (system prompt)',
              alignLabelWithHint: true,
            ),
            minLines: 3,
            maxLines: 8,
          ),
          const Divider(height: 32),
          Text('Sampling', style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: 8),
          Row(
            children: [
              const SizedBox(width: 110, child: Text('Temperature')),
              Expanded(
                child: Slider(
                  value: _temperature,
                  min: 0,
                  max: 2,
                  divisions: 40,
                  label: _temperature.toStringAsFixed(2),
                  onChanged: (value) => setState(() => _temperature = value),
                ),
              ),
              SizedBox(width: 40, child: Text(_temperature.toStringAsFixed(2))),
            ],
          ),
          DropdownButtonFormField<int>(
            initialValue: _maxTokens,
            decoration: const InputDecoration(
              labelText: 'Max tokens per reply',
            ),
            items: const [
              DropdownMenuItem(value: 256, child: Text('256')),
              DropdownMenuItem(value: 512, child: Text('512')),
              DropdownMenuItem(value: 1024, child: Text('1024')),
              DropdownMenuItem(value: 2048, child: Text('2048')),
            ],
            onChanged: (value) => setState(() => _maxTokens = value ?? 1024),
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              Expanded(
                child: TextField(
                  controller: _topK,
                  decoration: const InputDecoration(
                    labelText: 'Top-k',
                    hintText: 'engine default',
                  ),
                  keyboardType: TextInputType.number,
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: TextField(
                  controller: _topP,
                  decoration: const InputDecoration(
                    labelText: 'Top-p',
                    hintText: 'engine default',
                  ),
                  keyboardType: const TextInputType.numberWithOptions(
                    decimal: true,
                  ),
                ),
              ),
            ],
          ),
          const Divider(height: 32),
          SwitchListTile(
            contentPadding: EdgeInsets.zero,
            value: _enableTools,
            onChanged: (value) => setState(() => _enableTools = value),
            title: const Text('Tools'),
            subtitle: const Text(
              'Expose demo tools (current_time, roll_dice) to the agent. '
              'Needs a tool-trained model — Qwen3, Llama 3.2, or LFM2.',
            ),
          ),
          const SizedBox(height: 24),
          FilledButton(onPressed: _apply, child: const Text('Apply')),
          const SizedBox(height: 8),
          Text(
            'Changes take effect on the next message; the conversation is '
            'kept.',
            style: Theme.of(context).textTheme.bodySmall,
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }
}
