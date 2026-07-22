import 'dart:io';

import 'package:llama_cpp_flutter/llama_cpp_flutter.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  late Directory cacheDir;

  setUp(() async {
    cacheDir = await Directory.systemTemp.createTemp('model_downloader_test');
  });

  tearDown(() async {
    await cacheDir.delete(recursive: true);
  });

  group('ModelDownloader', () {
    test('downloads to a stable cache name and reports progress', () async {
      final bytes = List<int>.generate(1000, (i) => i % 251);
      final server = await _serve((request, hits) {
        request.response
          ..contentLength = bytes.length
          ..add(bytes);
      });
      addTearDown(() => server.close());
      final url = Uri.parse('http://127.0.0.1:${server.port}/model.gguf');

      final progress = <double>[];
      final path = await ModelDownloader().download(
        url,
        directory: cacheDir.path,
        onProgress: progress.add,
      );

      expect(File(path).readAsBytesSync(), bytes);
      expect(path, endsWith('-model.gguf'));
      expect(progress, isNotEmpty);
      expect(progress.last, 1.0);
    });

    test('returns a cached file without contacting the server', () async {
      var hits = 0;
      final server = await _serve((request, _) {
        hits++;
        request.response.add(const [1, 2, 3]);
      });
      addTearDown(() => server.close());
      final url = Uri.parse('http://127.0.0.1:${server.port}/model.gguf');

      final downloader = ModelDownloader();
      final first = await downloader.download(url, directory: cacheDir.path);
      final second = await downloader.download(url, directory: cacheDir.path);

      expect(second, first);
      expect(hits, 1);
    });

    test('resumes an interrupted download with a Range request', () async {
      final bytes = List<int>.generate(1000, (i) => i % 251);
      String? seenRange;
      final server = await _serve((request, _) {
        seenRange = request.headers.value(HttpHeaders.rangeHeader);
        final from = int.parse(seenRange!.replaceAll(RegExp(r'bytes=|-$'), ''));
        request.response
          ..statusCode = HttpStatus.partialContent
          ..headers.set(
            HttpHeaders.contentRangeHeader,
            'bytes $from-${bytes.length - 1}/${bytes.length}',
          )
          ..add(bytes.sublist(from));
      });
      addTearDown(() => server.close());
      final url = Uri.parse('http://127.0.0.1:${server.port}/model.gguf');

      // A previous run left the first 400 bytes behind.
      final partial = File(
        '${cacheDir.path}/${stableArtifactFileName(url)}.part',
      );
      partial.writeAsBytesSync(bytes.sublist(0, 400));

      final progress = <double>[];
      final path = await ModelDownloader().download(
        url,
        directory: cacheDir.path,
        onProgress: progress.add,
      );

      expect(seenRange, 'bytes=400-');
      expect(File(path).readAsBytesSync(), bytes);
      // Progress resumes partway in, never restarting at zero.
      expect(progress.first, greaterThan(0.4));
      expect(progress.last, 1.0);
    });

    test('recovers when the part file already covers the artifact', () async {
      // A crash between the final write and the rename leaves a complete
      // .part; the ranged retry then gets 416 and must restart cleanly.
      final bytes = List<int>.generate(100, (i) => i);
      final server = await _serve((request, _) {
        if (request.headers.value(HttpHeaders.rangeHeader) != null) {
          request.response.statusCode = HttpStatus.requestedRangeNotSatisfiable;
        } else {
          request.response
            ..contentLength = bytes.length
            ..add(bytes);
        }
      });
      addTearDown(() => server.close());
      final url = Uri.parse('http://127.0.0.1:${server.port}/model.gguf');
      File(
        '${cacheDir.path}/${stableArtifactFileName(url)}.part',
      ).writeAsBytesSync(bytes);

      final path = await ModelDownloader().download(
        url,
        directory: cacheDir.path,
      );

      expect(File(path).readAsBytesSync(), bytes);
    });

    test('fails when the response ends before the declared length', () async {
      final server = await _serve((request, _) {
        request.response
          ..headers.set(HttpHeaders.contentLengthHeader, 100)
          ..add(const [1, 2, 3]);
      });
      addTearDown(() => server.close());
      final url = Uri.parse('http://127.0.0.1:${server.port}/model.gguf');

      await expectLater(
        ModelDownloader().download(url, directory: cacheDir.path),
        throwsA(anything),
      );
      // The partial download stays as a .part for a later resume; no final
      // file appears.
      expect(
        File('${cacheDir.path}/${stableArtifactFileName(url)}').existsSync(),
        isFalse,
      );
    });

    test('sends the auth token as a bearer header', () async {
      String? seenAuth;
      final server = await _serve((request, _) {
        seenAuth = request.headers.value(HttpHeaders.authorizationHeader);
        request.response.add(const [1]);
      });
      addTearDown(() => server.close());
      final url = Uri.parse('http://127.0.0.1:${server.port}/gated.gguf');

      await ModelDownloader(
        authToken: 'hf_test',
      ).download(url, directory: cacheDir.path);

      expect(seenAuth, 'Bearer hf_test');
    });

    test('downloadSpecArtifacts fetches every declared artifact', () async {
      final server = await _serve((request, _) {
        request.response.add(request.uri.path.codeUnits);
      });
      addTearDown(() => server.close());
      Uri url(String name) =>
          Uri.parse('http://127.0.0.1:${server.port}/$name');

      final spec = ModelSpec(
        id: 'test',
        displayName: 'Test',
        modelUrl: url('weights.gguf'),
        mmprojUrl: url('mmproj.gguf'),
        contextSize: 4096,
        format: const ChatmlChatFormat(),
      );

      final artifacts = await ModelDownloader().downloadSpecArtifacts(
        spec,
        directory: cacheDir.path,
      );

      expect(File(artifacts.modelPath).existsSync(), isTrue);
      expect(artifacts.mmprojPath, isNotNull);
      expect(File(artifacts.mmprojPath!).existsSync(), isTrue);
      expect(artifacts.draftPath, isNull);
    });
  });

  group('huggingFaceModelUri', () {
    test('builds the resolve endpoint', () {
      expect(
        huggingFaceModelUri(
          repo: 'unsloth/gemma-3-4b-it-GGUF',
          file: 'gemma-3-4b-it-Q4_K_M.gguf',
        ).toString(),
        'https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/resolve/main/'
        'gemma-3-4b-it-Q4_K_M.gguf',
      );
    });
  });
}

/// Starts a loopback server whose requests are handled by [handle].
Future<HttpServer> _serve(
  void Function(HttpRequest request, int hit) handle,
) async {
  final server = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
  var hits = 0;
  server.listen((request) async {
    handle(request, hits++);
    try {
      await request.response.close();
    } on HttpException {
      // Deliberately truncated response (see the ends-early test).
    }
  });
  return server;
}
