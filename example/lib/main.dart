import 'package:flutter/material.dart';

import 'src/app_model.dart';
import 'src/screens/chat_screen.dart';

/// Demo app for llama_cpp_flutter: a flutter_ai_toolkit chat UI over a local
/// GGUF model, with screens for configuring the model (which weights, engine
/// settings) and the agent (persona, sampling, tools) at runtime. Runs
/// natively on macOS/iOS and on web via Wasm.
///
/// CI's macOS job builds this app as the link-level smoke test: it forces the
/// final app link to resolve every llama.cpp symbol the plugin references.
void main() {
  runApp(const ExampleApp());
}

class ExampleApp extends StatefulWidget {
  const ExampleApp({super.key});

  @override
  State<ExampleApp> createState() => _ExampleAppState();
}

class _ExampleAppState extends State<ExampleApp> {
  final AppModel model = AppModel();

  @override
  void dispose() {
    model.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'llama_cpp_flutter example',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
      ),
      darkTheme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.deepPurple,
          brightness: Brightness.dark,
        ),
      ),
      home: ChatScreen(model: model),
    );
  }
}
