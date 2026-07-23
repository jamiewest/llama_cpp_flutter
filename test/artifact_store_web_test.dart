@TestOn('browser')
library;

import 'dart:js_interop';

import 'package:flutter_test/flutter_test.dart';
import 'package:llama_cpp_flutter/llama_cpp_flutter.dart';
import 'package:llama_cpp_flutter/src/runtime/opfs_artifacts.dart';

void main() {
  late ArtifactStore store;
  final imported = <String>[];

  setUp(() {
    store = createArtifactStore();
  });

  // OPFS outlives the page, so every artifact this suite writes is removed
  // again rather than left for the next run.
  tearDown(() async {
    for (final key in imported) {
      await store.delete(key);
    }
    imported.clear();
  });

  Future<ManagedArtifact> import(String name, List<int> bytes) async {
    final artifact = await store.importStream(
      Stream<List<int>>.value(bytes),
      fileName: name,
      totalBytes: bytes.length,
    );
    imported.add(artifact.key);
    return artifact;
  }

  group('OpfsArtifactStore', () {
    test('imports a stream and reports it as stored', () async {
      final progress = <double>[];
      final artifact = await store.importStream(
        Stream<List<int>>.value(const [1, 2, 3, 4]),
        fileName: 'weights.gguf',
        totalBytes: 4,
        onProgress: progress.add,
      );
      imported.add(artifact.key);

      expect(artifact.fileName, 'weights.gguf');
      expect(artifact.sizeBytes, 4);
      expect(progress.last, 1.0);
      expect(await store.lookup(artifact.key), artifact);
      expect(await store.list(), contains(artifact));
      expect(await store.totalSizeBytes(), greaterThanOrEqualTo(4));
    });

    test('resolves stored artifacts into loadModel handles', () async {
      final model = await import('model.gguf', const [1, 2]);
      final mmproj = await import('mmproj.gguf', const [3]);

      final paths = await store.resolve(
        modelKey: model.key,
        mmprojKey: mmproj.key,
      );

      // Opaque to callers, but stable and distinct per artifact — the web
      // runtime dereferences them straight to their OPFS files.
      expect(paths.modelPath, isNotEmpty);
      expect(paths.mmprojPath, isNot(paths.modelPath));
      expect(paths.draftPath, isNull);
      expect(
        (await store.resolve(modelKey: model.key)).modelPath,
        paths.modelPath,
      );
    });

    test('resolved handles dereference to the stored file', () async {
      final artifact = await import('handle.gguf', const [7, 8, 9]);

      final paths = await store.resolve(modelKey: artifact.key);
      final key = artifactKeyOf(paths.modelPath);
      final file = await artifactFile(await requireArtifactDirectory(), key!);

      // This blob is what the web runtime stages — whole when it fits
      // wasm32's per-file limit, sliced into GGUF splits when it doesn't —
      // so the bytes reach wllama without a trip back through the network
      // stack.
      expect(key, artifact.key);
      expect(file, isNotNull);
      expect(file!.size, 3);
      final bytes = (await file.arrayBuffer().toDart).toDart.asUint8List();
      expect(bytes, <int>[7, 8, 9]);
    });

    test('refuses to resolve an artifact that is not stored', () async {
      await expectLater(
        store.resolve(modelKey: 'never-stored.gguf'),
        throwsA(isA<ArtifactStorageException>()),
      );
    });

    test('rejects a stream that ends before its declared length', () async {
      final before = await store.totalSizeBytes();

      await expectLater(
        store.importStream(
          Stream<List<int>>.value(const [1, 2, 3]),
          fileName: 'short.gguf',
          totalBytes: 10,
        ),
        throwsA(isA<ArtifactStorageException>()),
      );

      // A spent stream cannot be resumed, so the failure leaves nothing
      // occupying storage.
      expect(await store.totalSizeBytes(), before);
      expect(
        (await store.list()).map((a) => a.fileName),
        isNot(contains('short.gguf')),
      );
    });

    test('has no filesystem path to import from', () {
      expect(() => store.importFile('/tmp/model.gguf'), throwsUnsupportedError);
    });

    test('deletes permanently, and survives deleting twice', () async {
      final artifact = await import('doomed.gguf', const [1, 2, 3]);

      await store.delete(artifact.key);
      await store.delete(artifact.key);

      expect(await store.lookup(artifact.key), isNull);
      expect(await store.list(), isNot(contains(artifact)));
    });
  });
}
