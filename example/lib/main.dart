import 'package:flutter/material.dart';
import 'package:llama_cpp_flutter/bridge.dart';

/// Minimal smoke-test app: instantiating [LlamaCppFlutter] registers the plugin
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
    final llama = LlamaCppFlutter();
    return MaterialApp(
      home: Scaffold(
        body: Center(
          child: Text('llama_cpp_flutter linked: ${llama.runtimeType}'),
        ),
      ),
    );
  }
}
