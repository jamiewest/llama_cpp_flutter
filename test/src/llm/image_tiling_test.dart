import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:llama_flutter/llama_flutter.dart';
import 'package:extensions/ai.dart';
import 'package:flutter_test/flutter_test.dart';

/// Renders a solid-color [width] x [height] image and encodes it as PNG.
Future<Uint8List> _pngBytes(int width, int height) async {
  final recorder = ui.PictureRecorder();
  ui.Canvas(recorder).drawRect(
    ui.Rect.fromLTWH(0, 0, width.toDouble(), height.toDouble()),
    ui.Paint()..color = const ui.Color(0xFF336699),
  );
  final picture = recorder.endRecording();
  final image = await picture.toImage(width, height);
  final data = await image.toByteData(format: ui.ImageByteFormat.png);
  return data!.buffer.asUint8List();
}

Future<(int width, int height)> _decodedSize(Uint8List bytes) async {
  final codec = await ui.instantiateImageCodec(bytes);
  final frame = await codec.getNextFrame();
  final size = (frame.image.width, frame.image.height);
  frame.image.dispose();
  codec.dispose();
  return size;
}

void main() {
  group('tileImageBytes', () {
    test('splits into a row-major grid of overlapping PNG crops', () async {
      final bytes = await _pngBytes(100, 80);
      const tiling = ImageTiling(minimumEdge: 10);

      final tiles = await tileImageBytes(bytes, tiling);

      expect(tiles, hasLength(4));
      // 50x40 base cells plus 10% overlap on each interior edge: corner
      // tiles span 55x44.
      for (final tile in tiles) {
        expect(await _decodedSize(tile), (55, 44));
      }
    });

    test('passes small images through unchanged', () async {
      final bytes = await _pngBytes(100, 80);

      final tiles = await tileImageBytes(bytes, const ImageTiling());

      expect(tiles.single, same(bytes));
    });

    test('keeps the full image ahead of its tiles when asked', () async {
      final bytes = await _pngBytes(60, 60);
      const tiling = ImageTiling(
        columns: 3,
        rows: 1,
        overlap: 0,
        minimumEdge: 10,
        includeFullImage: true,
      );

      final tiles = await tileImageBytes(bytes, tiling);

      expect(tiles, hasLength(4));
      expect(tiles.first, same(bytes));
      expect(await _decodedSize(tiles.last), (20, 60));
    });
  });

  group('tiledMessages', () {
    test('expands image content in place and leaves the rest alone', () async {
      final imageBytes = await _pngBytes(100, 80);
      final audioBytes = Uint8List.fromList([82, 73, 70, 70]);
      final textOnly = ChatMessage.fromText(ChatRole.assistant, 'Hi');
      final withImage = ChatMessage(
        role: ChatRole.user,
        contents: [
          TextContent('Read this page'),
          DataContent(imageBytes, mediaType: 'image/png'),
          DataContent(audioBytes, mediaType: 'audio/wav'),
        ],
      );

      final tiled = await tiledMessages(
        [textOnly, withImage],
        const ImageTiling(minimumEdge: 10),
      );

      expect(tiled.first, same(textOnly));
      final contents = tiled.last.contents;
      expect(contents, hasLength(6));
      expect(contents.first, isA<TextContent>());
      expect(
        contents.sublist(1, 5),
        everyElement(
          isA<DataContent>().having(
            (data) => data.hasTopLevelMediaType('image'),
            'is image',
            isTrue,
          ),
        ),
      );
      expect((contents.last as DataContent).data, same(audioBytes));
    });
  });
}
