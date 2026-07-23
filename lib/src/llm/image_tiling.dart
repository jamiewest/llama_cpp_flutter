/// Grid tiling of image inputs for vision fidelity (OCR, dense documents).
///
/// Vision encoders spend a fixed token budget per image, so one full-page
/// photo of small text gets few tokens per glyph. Splitting the page into
/// overlapping crops gives the encoder that budget once per crop —
/// often a bigger fidelity win than raising the budget itself.
library;

import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:extensions/ai.dart';

/// How to split each attached image into tiles before prompt rendering.
///
/// Set on `ModelSpec.imageTiling` (or directly on `LlamaChatClient`) to
/// have every image-typed [DataContent] replaced, in place, by
/// [columns] x [rows] overlapping PNG crops. Each crop consumes its own
/// vision-encoder token budget and its own media marker, so prefill cost
/// scales with the tile count.
class ImageTiling {
  /// Creates a tiling configuration; the default splits into 2x2 tiles
  /// with 10% overlap.
  const ImageTiling({
    this.columns = 2,
    this.rows = 2,
    this.overlap = 0.1,
    this.minimumEdge = 512,
    this.includeFullImage = false,
  }) : assert(columns >= 1, 'columns must be at least 1'),
       assert(rows >= 1, 'rows must be at least 1'),
       assert(
         overlap >= 0 && overlap < 0.5,
         'overlap must be in [0, 0.5)',
       );

  /// Number of tile columns.
  final int columns;

  /// Number of tile rows.
  final int rows;

  /// Fraction of a tile's width/height duplicated into each neighbouring
  /// tile, so text straddling a cut line appears whole in at least one
  /// tile. `0.1` duplicates 10% on each interior edge.
  final double overlap;

  /// Images whose width AND height are both below this skip tiling and
  /// pass through unchanged — small images gain nothing from being split.
  final int minimumEdge;

  /// Whether to keep the untiled image ahead of its tiles, giving the
  /// model a whole-page view for layout plus the crops for detail. Costs
  /// one extra image budget per attachment.
  final bool includeFullImage;

  /// Whether [image] is large enough to be worth splitting.
  bool applies(ui.Image image) =>
      (columns > 1 || rows > 1) &&
      (image.width >= minimumEdge || image.height >= minimumEdge);

  @override
  bool operator ==(Object other) =>
      other is ImageTiling &&
      other.columns == columns &&
      other.rows == rows &&
      other.overlap == overlap &&
      other.minimumEdge == minimumEdge &&
      other.includeFullImage == includeFullImage;

  @override
  int get hashCode =>
      Object.hash(columns, rows, overlap, minimumEdge, includeFullImage);

  @override
  String toString() =>
      'ImageTiling(columns: $columns, rows: $rows, overlap: $overlap, '
      'minimumEdge: $minimumEdge, includeFullImage: $includeFullImage)';
}

/// Splits [bytes] (an encoded image) into [tiling]'s grid of PNG crops.
///
/// Returns the original bytes unchanged when the image is below
/// [ImageTiling.minimumEdge] or the grid is 1x1. Tiles are ordered
/// row-major, preceded by the full image when
/// [ImageTiling.includeFullImage] is set. Throws if [bytes] is not a
/// decodable image.
Future<List<Uint8List>> tileImageBytes(
  Uint8List bytes,
  ImageTiling tiling,
) async {
  final codec = await ui.instantiateImageCodec(bytes);
  try {
    final frame = await codec.getNextFrame();
    final image = frame.image;
    try {
      if (!tiling.applies(image)) return <Uint8List>[bytes];

      final width = image.width.toDouble();
      final height = image.height.toDouble();
      final tileWidth = width / tiling.columns;
      final tileHeight = height / tiling.rows;
      final overlapX = tileWidth * tiling.overlap;
      final overlapY = tileHeight * tiling.overlap;

      final tiles = <Uint8List>[
        if (tiling.includeFullImage) bytes,
      ];
      for (var row = 0; row < tiling.rows; row++) {
        for (var column = 0; column < tiling.columns; column++) {
          final source = ui.Rect.fromLTRB(
            (column * tileWidth - overlapX).clamp(0, width),
            (row * tileHeight - overlapY).clamp(0, height),
            ((column + 1) * tileWidth + overlapX).clamp(0, width),
            ((row + 1) * tileHeight + overlapY).clamp(0, height),
          );
          tiles.add(await _encodeCrop(image, source));
        }
      }
      return tiles;
    } finally {
      image.dispose();
    }
  } finally {
    codec.dispose();
  }
}

/// Returns [messages] with each image-typed [DataContent] replaced by its
/// [tiling] crops (as `image/png` [DataContent]s), preserving content
/// order. Non-image content — text, audio, tool calls — is untouched.
Future<List<ChatMessage>> tiledMessages(
  List<ChatMessage> messages,
  ImageTiling tiling,
) async {
  final result = <ChatMessage>[];
  for (final message in messages) {
    final hasImage = message.contents.any(
      (content) =>
          content is DataContent &&
          content.data != null &&
          content.hasTopLevelMediaType('image'),
    );
    if (!hasImage) {
      result.add(message);
      continue;
    }
    final contents = <AIContent>[];
    for (final content in message.contents) {
      if (content is DataContent &&
          content.data != null &&
          content.hasTopLevelMediaType('image')) {
        for (final tile in await tileImageBytes(content.data!, tiling)) {
          contents.add(
            identical(tile, content.data)
                ? content
                : DataContent(tile, mediaType: 'image/png'),
          );
        }
      } else {
        contents.add(content);
      }
    }
    result.add(ChatMessage(role: message.role, contents: contents));
  }
  return result;
}

Future<Uint8List> _encodeCrop(ui.Image image, ui.Rect source) async {
  final recorder = ui.PictureRecorder();
  final canvas = ui.Canvas(recorder);
  final destination = ui.Rect.fromLTWH(0, 0, source.width, source.height);
  canvas.drawImageRect(image, source, destination, ui.Paint());
  final picture = recorder.endRecording();
  try {
    final tile = await picture.toImage(
      source.width.round(),
      source.height.round(),
    );
    try {
      final data = await tile.toByteData(format: ui.ImageByteFormat.png);
      if (data == null) {
        throw StateError('image tile failed to encode as PNG');
      }
      return data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
    } finally {
      tile.dispose();
    }
  } finally {
    picture.dispose();
  }
}
