import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:llama_cpp_flutter/gguf.dart';
import 'package:llama_cpp_flutter/llama_cpp_flutter.dart';

void main() {
  late Directory storeDir;
  late Directory sourceDir;
  late ArtifactStore store;

  setUp(() async {
    storeDir = await Directory.systemTemp.createTemp('artifact_store_test');
    sourceDir = await Directory.systemTemp.createTemp('artifact_source_test');
    store = createArtifactStore(directory: storeDir.path);
  });

  tearDown(() async {
    await storeDir.delete(recursive: true);
    await sourceDir.delete(recursive: true);
  });

  group('createArtifactStore', () {
    test('requires a directory to store artifacts in', () {
      expect(createArtifactStore, throwsArgumentError);
    });
  });

  group('fetch', () {
    test(
      'downloads under the URL\'s stable key and reports progress',
      () async {
        final bytes = List<int>.generate(1000, (i) => i % 251);
        final server = await _serve((request) {
          request.response
            ..contentLength = bytes.length
            ..add(bytes);
        });
        addTearDown(() => server.close());
        final url = Uri.parse('http://127.0.0.1:${server.port}/model.gguf');

        final progress = <double>[];
        final artifact = await store.fetch(url, onProgress: progress.add);

        expect(artifact.key, stableArtifactFileName(url));
        expect(artifact.fileName, 'model.gguf');
        expect(artifact.sizeBytes, bytes.length);
        expect(progress.last, 1.0);
        expect(
          File('${storeDir.path}/${artifact.key}').readAsBytesSync(),
          bytes,
        );
      },
    );

    test('adopts an artifact an earlier download already cached', () async {
      var hits = 0;
      final server = await _serve((request) {
        hits++;
        request.response.add(const [1, 2, 3]);
      });
      addTearDown(() => server.close());
      final url = Uri.parse('http://127.0.0.1:${server.port}/model.gguf');

      // What ModelDownloader (or a previous release of the app) left behind.
      File(
        '${storeDir.path}/${stableArtifactFileName(url)}',
      ).writeAsBytesSync(const [1, 2, 3]);

      final artifact = await store.fetch(url);

      expect(hits, 0);
      expect(artifact.sizeBytes, 3);
    });

    test('leaves no listable artifact when the download fails', () async {
      final server = await _serve((request) {
        request.response
          ..headers.set(HttpHeaders.contentLengthHeader, 100)
          ..add(const [1, 2, 3]);
      });
      addTearDown(() => server.close());
      final url = Uri.parse('http://127.0.0.1:${server.port}/model.gguf');

      await expectLater(store.fetch(url), throwsA(anything));

      expect(await store.list(), isEmpty);
      expect(await store.lookup(stableArtifactFileName(url)), isNull);
      // Whatever arrived is kept for a later resume, out of sight of the
      // library but still charged against storage.
      expect(
        File(
          '${storeDir.path}/${stableArtifactFileName(url)}.part',
        ).existsSync(),
        isTrue,
      );
    });
  });

  group('import', () {
    test('copies the source file by default', () async {
      final source = File('${sourceDir.path}/weights.gguf')
        ..writeAsBytesSync(List<int>.generate(500, (i) => i % 251));

      final progress = <double>[];
      final artifact = await store.importFile(
        source.path,
        onProgress: progress.add,
      );

      expect(source.existsSync(), isTrue, reason: 'the user still owns it');
      expect(artifact.fileName, 'weights.gguf');
      expect(artifact.sizeBytes, 500);
      expect(progress.last, 1.0);
    });

    test('moves the source when the caller owns it', () async {
      final source = File('${sourceDir.path}/picked.gguf')
        ..writeAsBytesSync(const [1, 2, 3, 4]);

      final artifact = await store.importFile(source.path, moveSource: true);

      expect(source.existsSync(), isFalse);
      expect(artifact.sizeBytes, 4);
      expect(File('${storeDir.path}/${artifact.key}').readAsBytesSync(), const [
        1,
        2,
        3,
        4,
      ]);
    });

    test('keeps repeat imports of one file name apart', () async {
      final source = File('${sourceDir.path}/weights.gguf')
        ..writeAsBytesSync(const [1]);

      final first = await store.importFile(source.path);
      final second = await store.importFile(source.path);

      expect(first.key, isNot(second.key));
      expect(first.fileName, second.fileName);
      expect((await store.list()).length, 2);
    });

    test('rejects a stream that ends before its declared length', () async {
      await expectLater(
        store.importStream(
          Stream<List<int>>.value(const [1, 2, 3]),
          fileName: 'short.gguf',
          totalBytes: 10,
        ),
        throwsA(isA<ArtifactStorageException>()),
      );

      expect(await store.list(), isEmpty);
      expect(await store.totalSizeBytes(), 0);
    });

    test('fails on a missing source file', () async {
      await expectLater(
        store.importFile('${sourceDir.path}/absent.gguf'),
        throwsA(isA<ArtifactStorageException>()),
      );
    });
  });

  group('resolve', () {
    test('returns loadModel paths for every stored artifact', () async {
      final model = await _import(store, sourceDir, 'model.gguf');
      final mmproj = await _import(store, sourceDir, 'mmproj.gguf');
      final draft = await _import(store, sourceDir, 'draft.gguf');

      final paths = await store.resolve(
        modelKey: model.key,
        mmprojKey: mmproj.key,
        draftKey: draft.key,
      );

      expect(paths.modelPath, '${storeDir.path}/${model.key}');
      expect(paths.mmprojPath, '${storeDir.path}/${mmproj.key}');
      expect(paths.draftPath, '${storeDir.path}/${draft.key}');
      expect(File(paths.modelPath).existsSync(), isTrue);
    });

    test('omits companions that were not requested', () async {
      final model = await _import(store, sourceDir, 'model.gguf');

      final paths = await store.resolve(modelKey: model.key);

      expect(paths.mmprojPath, isNull);
      expect(paths.draftPath, isNull);
    });

    test('refuses to resolve an artifact that is not stored', () async {
      await expectLater(
        store.resolve(modelKey: 'nothing-here.gguf'),
        throwsA(
          isA<ArtifactStorageException>().having(
            (e) => e.message,
            'message',
            contains('not in managed storage'),
          ),
        ),
      );
    });
  });

  group('delete', () {
    test('removes the artifact and any partial transfer of it', () async {
      final artifact = await _import(store, sourceDir, 'model.gguf');
      File('${storeDir.path}/${artifact.key}.part').writeAsBytesSync(const [1]);

      await store.delete(artifact.key);

      expect(await store.list(), isEmpty);
      expect(await store.totalSizeBytes(), 0);
      expect(await store.lookup(artifact.key), isNull);
    });

    test('is a no-op for an unknown key', () async {
      await store.delete('never-stored.gguf');
      expect(await store.list(), isEmpty);
    });
  });

  group('list', () {
    test('is empty before anything is stored', () async {
      expect(await store.list(), isEmpty);
      expect(await store.totalSizeBytes(), 0);
    });

    test('reports every stored artifact with its display name', () async {
      await _import(store, sourceDir, 'weights.gguf', bytes: 4);
      await _import(store, sourceDir, 'mmproj.gguf', bytes: 6);

      final artifacts = await store.list();

      expect(
        artifacts.map((a) => a.fileName),
        containsAll(<String>['weights.gguf', 'mmproj.gguf']),
      );
      expect(await store.totalSizeBytes(), 10);
    });
  });
}

Future<ManagedArtifact> _import(
  ArtifactStore store,
  Directory sourceDir,
  String name, {
  int bytes = 1,
}) async {
  final source = File('${sourceDir.path}/$name')
    ..writeAsBytesSync(List<int>.filled(bytes, 7));
  final artifact = await store.importFile(source.path);
  await source.delete();
  return artifact;
}

/// Starts a loopback server whose requests are handled by [handle].
Future<HttpServer> _serve(void Function(HttpRequest request) handle) async {
  final server = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
  server.listen((request) async {
    handle(request);
    try {
      await request.response.close();
    } on HttpException {
      // Deliberately truncated response (see the failed-download test).
    }
  });
  return server;
}
