/// Compile-time check that the README's code examples stay valid: these
/// functions mirror the snippets and are never executed — the analyzer
/// failing here means the README needs the same fix.
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:extensions/ai.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:llama_cpp_flutter/llama_cpp_flutter.dart';

Future<void> readmeQuickStart() async {
  final runtime = createLlamaRuntime();

  final spec = ModelSpec(
    id: 'gemma-3-4b-it-q4km',
    displayName: 'Gemma 3 4B IT',
    modelUrl: huggingFaceModelUri(
      repo: 'unsloth/gemma-3-4b-it-GGUF',
      file: 'gemma-3-4b-it-Q4_K_M.gguf',
    ),
    contextSize: 4096,
    format: resolveChatFormat('gemma')!,
  );

  final session = await runtime.loadModel(
    spec,
    localPath: '/path/to/gemma-3-4b-it-Q4_K_M.gguf',
  );

  await for (final text in session.generate('Why is the sky blue?')) {
    stdout.write(text);
  }

  await session.dispose();
}

Future<void> readmeDownloadingRuntime(
  Directory cacheDir,
  ModelSpec spec,
) async {
  final runtime = createLlamaRuntime(
    artifactCacheDirectory: cacheDir.path,
  );
  final session = await runtime.loadModel(
    spec,
    onProgress: (progress) => stdout.write('load: $progress'),
  );
  await session.dispose();
}

Future<void> readmeManualDownload(
  Directory cacheDir,
  ModelSpec spec,
  LlamaRuntime runtime,
) async {
  final artifacts = await ModelDownloader().downloadSpecArtifacts(
    spec,
    directory: cacheDir.path,
    onProgress: (progress) => stdout.write('download: $progress'),
  );
  final session = await runtime.loadModel(
    spec,
    localPath: artifacts.modelPath,
    localMmprojPath: artifacts.mmprojPath,
    localDraftPath: artifacts.draftPath,
  );
  await session.dispose();
}

Future<void> readmeChatClient(ModelSpec spec, LlamaSession session) async {
  final client = createLlamaChatClient(
    spec: spec,
    sessionProvider: () async => session,
  );

  final response = await client.getResponse(
    messages: [ChatMessage.fromText(ChatRole.user, 'Hello!')],
  );
  stdout.write(response.text);
}

void readmeFormatDetection(List<int> headerBytes) {
  final name = detectChatFormatNameForGguf(Uint8List.fromList(headerBytes));
  final format = resolveChatFormat(name) ?? resolveChatFormat('chatml')!;
  stdout.write(format.runtimeType);
}

void main() {
  test('README example code compiles', () {
    // Nothing to run: the value of this test is that the snippet mirrors
    // above type-check. Referencing them keeps the analyzer from flagging
    // unused declarations.
    expect(readmeQuickStart, isNotNull);
    expect(readmeDownloadingRuntime, isNotNull);
    expect(readmeManualDownload, isNotNull);
    expect(readmeChatClient, isNotNull);
    expect(readmeFormatDetection, isNotNull);
  });
}
