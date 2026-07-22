import 'package:flutter/material.dart';
import 'package:llama_flutter/bridge.dart';

/// Minimal smoke-test app: instantiating [LlamaFlutter] registers the plugin
/// and forces the final app link to resolve every llama.cpp symbol the plugin
/// references, which is what CI's macOS job exists to verify. No model is
/// loaded.
void main() {
  runApp(const ExampleApp());
}

class ExampleApp extends StatelessWidget {
  const ExampleApp({super.key});

  @override
  Widget build(BuildContext context) {
    final llama = LlamaFlutter();
    return MaterialApp(
      home: Scaffold(
        body: Center(
          child: Text('llama_flutter linked: ${llama.runtimeType}'),
        ),
      ),
    );
  }
}
