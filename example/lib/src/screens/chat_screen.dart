import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_ai_toolkit/flutter_ai_toolkit.dart';

import '../app_model.dart';
import 'agent_screen.dart';
import 'model_screen.dart';

class ChatScreen extends StatelessWidget {
  const ChatScreen({super.key, required this.model});

  final AppModel model;

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: model,
      builder: (context, _) => Scaffold(
        appBar: AppBar(
          title: Text(model.loadedModelName ?? 'llama_cpp_flutter'),
          actions: [
            IconButton(
              tooltip: 'New chat',
              icon: const Icon(Icons.refresh),
              onPressed: model.status == ModelStatus.ready
                  ? model.provider.clear
                  : null,
            ),
            IconButton(
              tooltip: 'Agent settings',
              icon: const Icon(Icons.psychology_outlined),
              onPressed: () => _push(context, AgentScreen(model: model)),
            ),
            IconButton(
              tooltip: 'Model settings',
              icon: const Icon(Icons.memory_outlined),
              onPressed: () => _push(context, ModelScreen(model: model)),
            ),
          ],
        ),
        body: switch (model.status) {
          ModelStatus.empty => _Welcome(model: model),
          ModelStatus.loading => _LoadingView(model: model),
          ModelStatus.error => _ErrorView(model: model),
          ModelStatus.ready => Column(
            children: [
              if (kIsWeb && !model.multiThreaded)
                MaterialBanner(
                  content: const Text(
                    'This page is not cross-origin isolated, so inference '
                    'runs on a single thread and will be slow. Serve with '
                    'COOP/COEP headers to enable multi-threading (see the '
                    'example README).',
                  ),
                  leading: const Icon(Icons.speed),
                  actions: const [SizedBox.shrink()],
                ),
              Expanded(
                child: LlmChatView(
                  key: ValueKey(model.modelGeneration),
                  provider: model.provider,
                  welcomeMessage:
                      'The model runs entirely on this device. '
                      'With tools enabled, try "what time is it?" or '
                      '"roll a 20-sided die".',
                  enableAttachments: false,
                  enableVoiceNotes: false,
                ),
              ),
            ],
          ),
        },
      ),
    );
  }

  void _push(BuildContext context, Widget screen) {
    Navigator.of(context).push(MaterialPageRoute<void>(builder: (_) => screen));
  }
}

class _Welcome extends StatelessWidget {
  const _Welcome({required this.model});

  final AppModel model;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 420),
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(Icons.smart_toy_outlined, size: 56),
              const SizedBox(height: 16),
              Text(
                'On-device chat with llama.cpp',
                style: Theme.of(context).textTheme.titleLarge,
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 8),
              const Text(
                'Pick a model to download and run locally — no server, '
                'no API key. The same code path runs natively on '
                'macOS/iOS and in the browser via Wasm.',
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 24),
              FilledButton.icon(
                icon: const Icon(Icons.download),
                label: const Text('Choose a model'),
                onPressed: () => Navigator.of(context).push(
                  MaterialPageRoute<void>(
                    builder: (_) => ModelScreen(model: model),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _LoadingView extends StatelessWidget {
  const _LoadingView({required this.model});

  final AppModel model;

  @override
  Widget build(BuildContext context) {
    final percent = (model.loadProgress * 100).clamp(0, 100).toStringAsFixed(0);
    return Center(
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 320),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            LinearProgressIndicator(
              value: model.loadProgress > 0 ? model.loadProgress : null,
            ),
            const SizedBox(height: 12),
            Text('Downloading & loading model… $percent%'),
          ],
        ),
      ),
    );
  }
}

class _ErrorView extends StatelessWidget {
  const _ErrorView({required this.model});

  final AppModel model;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 480),
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(Icons.error_outline, size: 48),
              const SizedBox(height: 12),
              Text(
                model.errorMessage ?? 'Model failed to load.',
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 16),
              Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  OutlinedButton(
                    onPressed: () => Navigator.of(context).push(
                      MaterialPageRoute<void>(
                        builder: (_) => ModelScreen(model: model),
                      ),
                    ),
                    child: const Text('Model settings'),
                  ),
                  const SizedBox(width: 12),
                  FilledButton(
                    onPressed: model.loadModel,
                    child: const Text('Retry'),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}
