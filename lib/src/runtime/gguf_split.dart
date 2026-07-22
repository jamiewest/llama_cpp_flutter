// Copyright 2024 The Flutter Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

/// Client-side GGUF splitting.
///
/// The web llama runtime (wllama) cannot load a single file of 2 GiB or
/// more: llama.cpp compiled to wasm32 addresses each staged file with
/// 32-bit offsets. Upstream's answer is `llama-gguf-split`, which asks
/// users to pre-shard models and host multiple files. This planner makes
/// that seamless instead: it reads the header of a monolithic GGUF and
/// emits replacement headers for a set of split files that follow the
/// same on-disk format `llama-gguf-split` produces (`split.no`,
/// `split.count`, `split.tensors.count` metadata, per-file tensor infos
/// with rebased offsets). Tensor data is never rewritten — each split's
/// data section is a contiguous byte range of the source file, so the
/// caller can compose split Blobs from zero-copy slices.
///
/// Pure Dart and byte-oriented so it is unit-testable off the web.
library;

import 'dart:convert';
import 'dart:typed_data';

import 'gguf_reader.dart';

/// Upstream-recommended split size (512 MiB), small enough to avoid
/// wasm out-of-memory spikes while staging a file.
const int ggufSplitDefaultMaxBytes = 512 * 1024 * 1024;

/// Hard per-file ceiling: wasm32 file offsets are signed 32-bit.
const int ggufSplitHardMaxBytes = 0x7fffffff;

/// Outcome of [planGgufSplit].
sealed class GgufSplitPlanResult {
  const GgufSplitPlanResult();
}

/// The prefix ended before the header (metadata + tensor infos) did.
///
/// Retry with a larger [planGgufSplit] `headerPrefix`.
final class GgufSplitNeedsLargerPrefix extends GgufSplitPlanResult {
  const GgufSplitNeedsLargerPrefix();
}

/// The file cannot be split client-side; the [reason] is user-facing.
final class GgufSplitUnsupported extends GgufSplitPlanResult {
  const GgufSplitUnsupported(this.reason);

  final String reason;
}

/// A viable split layout for the source file.
final class GgufSplitPlan extends GgufSplitPlanResult {
  const GgufSplitPlan(this.parts);

  final List<GgufSplitPart> parts;
}

/// One split file: freshly built header bytes followed by an untouched
/// byte range of the source file.
final class GgufSplitPart {
  const GgufSplitPart({
    required this.headerBytes,
    required this.dataStart,
    required this.dataEnd,
  });

  /// Complete header (magic through alignment padding). The split file
  /// is exactly `headerBytes + source[dataStart..dataEnd)`.
  final Uint8List headerBytes;

  /// Absolute source-file offset where this split's tensor data begins.
  final int dataStart;

  /// Absolute source-file offset where this split's tensor data ends
  /// (exclusive).
  final int dataEnd;

  int get sizeBytes => headerBytes.length + (dataEnd - dataStart);
}

/// Plans a split of the GGUF whose first bytes are [headerPrefix] and
/// whose full length is [totalBytes].
///
/// [headerPrefix] must cover the whole header (all metadata key/values
/// and tensor infos); otherwise [GgufSplitNeedsLargerPrefix] is
/// returned and the caller should retry with a longer prefix. Each
/// planned part's tensor data stays at or under [maxSplitBytes] unless
/// a single tensor is larger, in which case the part grows to hold that
/// tensor (still capped by [ggufSplitHardMaxBytes]).
GgufSplitPlanResult planGgufSplit({
  required Uint8List headerPrefix,
  required int totalBytes,
  int maxSplitBytes = ggufSplitDefaultMaxBytes,
}) {
  final _ParsedGguf parsed;
  try {
    parsed = _ParsedGguf.parse(headerPrefix);
  } on GgufTruncated {
    return headerPrefix.length >= totalBytes
        ? const GgufSplitUnsupported('The GGUF header is corrupt.')
        : const GgufSplitNeedsLargerPrefix();
  } on FormatException catch (error) {
    return GgufSplitUnsupported(error.message);
  }
  if (parsed.tensors.isEmpty) {
    return const GgufSplitUnsupported(
      'The GGUF file declares no tensors, so it cannot be split.',
    );
  }

  final dataStart = _align(parsed.headerEnd, parsed.alignment);
  final dataSize = totalBytes - dataStart;
  if (dataSize < 0) {
    return const GgufSplitUnsupported('The GGUF header is corrupt.');
  }

  // Tensor data extents in offset order. A tensor's extent runs to the
  // next tensor's offset (that gap includes alignment padding), and the
  // last one runs to the end of the data section.
  final tensors = [...parsed.tensors]
    ..sort((a, b) => a.offset.compareTo(b.offset));
  final extents = <int>[
    for (var i = 0; i + 1 < tensors.length; i++)
      tensors[i + 1].offset - tensors[i].offset,
    dataSize - tensors.last.offset,
  ];
  if (extents.any((extent) => extent < 0) || tensors.first.offset != 0) {
    return const GgufSplitUnsupported('The GGUF tensor layout is corrupt.');
  }

  // Greedy packing: start a new part when adding the next tensor would
  // push the current one past the budget.
  final partTensorCounts = <int>[];
  var count = 0;
  var size = 0;
  for (final extent in extents) {
    if (count > 0 && size + extent > maxSplitBytes) {
      partTensorCounts.add(count);
      count = 0;
      size = 0;
    }
    count += 1;
    size += extent;
  }
  partTensorCounts.add(count);

  final parts = <GgufSplitPart>[];
  var firstTensor = 0;
  for (var partNo = 0; partNo < partTensorCounts.length; partNo++) {
    final partTensors = tensors.sublist(
      firstTensor,
      firstTensor + partTensorCounts[partNo],
    );
    firstTensor += partTensorCounts[partNo];
    final base = partTensors.first.offset;
    final end = firstTensor < tensors.length
        ? tensors[firstTensor].offset
        : dataSize;
    final header = _buildSplitHeader(
      parsed: parsed,
      partTensors: partTensors,
      partNo: partNo,
      partCount: partTensorCounts.length,
      baseOffset: base,
    );
    final part = GgufSplitPart(
      headerBytes: header,
      dataStart: dataStart + base,
      dataEnd: dataStart + end,
    );
    if (part.sizeBytes > ggufSplitHardMaxBytes) {
      return const GgufSplitUnsupported(
        'A single tensor in this GGUF is larger than the 2 GiB per-file '
        'limit of the web runtime.',
      );
    }
    parts.add(part);
  }
  return GgufSplitPlan(parts);
}

int _align(int offset, int alignment) =>
    (offset + alignment - 1) ~/ alignment * alignment;

/// Builds the header for one split file.
///
/// Mirrors `llama-gguf-split`: the first split carries every source
/// metadata key/value plus the three `split.*` keys; later splits carry
/// only the `split.*` keys (and `general.alignment` when the source
/// overrides the default, since offsets keep the source alignment).
Uint8List _buildSplitHeader({
  required _ParsedGguf parsed,
  required List<_TensorInfo> partTensors,
  required int partNo,
  required int partCount,
  required int baseOffset,
}) {
  final builder = BytesBuilder(copy: false);
  final kvRanges = partNo == 0
      ? parsed.kvRanges
      : <_ByteRange>[?parsed.alignmentKvRange];

  final fixed = ByteData(24)
    ..setUint32(0, ggufMagic, Endian.little)
    ..setUint32(4, parsed.version, Endian.little)
    ..setUint64(8, partTensors.length, Endian.little)
    ..setUint64(16, kvRanges.length + 3, Endian.little);
  builder
    ..add(fixed.buffer.asUint8List())
    ..add(_kvBytes(parsed.bytes, kvRanges))
    ..add(_uint16Kv('split.no', partNo))
    ..add(_uint16Kv('split.count', partCount))
    ..add(_int32Kv('split.tensors.count', parsed.tensors.length));

  for (final tensor in partTensors) {
    // A tensor info is name, dims, and type followed by an 8-byte data
    // offset; copy it verbatim and rebase just the offset.
    builder.add(
      Uint8List.sublistView(parsed.bytes, tensor.start, tensor.end - 8),
    );
    final offset = ByteData(8)
      ..setUint64(0, tensor.offset - baseOffset, Endian.little);
    builder.add(offset.buffer.asUint8List());
  }

  final padding = _align(builder.length, parsed.alignment) - builder.length;
  builder.add(Uint8List(padding));
  return builder.toBytes();
}

Uint8List _kvBytes(Uint8List source, List<_ByteRange> ranges) {
  final builder = BytesBuilder(copy: false);
  for (final range in ranges) {
    builder.add(Uint8List.sublistView(source, range.start, range.end));
  }
  return builder.toBytes();
}

Uint8List _uint16Kv(String key, int value) {
  final data = ByteData(2)..setUint16(0, value, Endian.little);
  return _kv(key, ggufTypeUint16, data);
}

Uint8List _int32Kv(String key, int value) {
  final data = ByteData(4)..setInt32(0, value, Endian.little);
  return _kv(key, ggufTypeInt32, data);
}

Uint8List _kv(String key, int type, ByteData value) {
  final keyBytes = utf8.encode(key);
  final data = ByteData(8 + keyBytes.length + 4 + value.lengthInBytes)
    ..setUint64(0, keyBytes.length, Endian.little)
    ..setUint32(8 + keyBytes.length, type, Endian.little);
  final bytes = data.buffer.asUint8List()
    ..setRange(8, 8 + keyBytes.length, keyBytes);
  bytes.setRange(
    8 + keyBytes.length + 4,
    bytes.length,
    value.buffer.asUint8List(),
  );
  return bytes;
}

const int _defaultAlignment = 32;

final class _ByteRange {
  const _ByteRange(this.start, this.end);

  final int start;
  final int end;
}

final class _TensorInfo {
  const _TensorInfo({
    required this.start,
    required this.end,
    required this.offset,
  });

  /// Byte range of the whole tensor-info record in the source header.
  final int start;
  final int end;

  /// Data offset relative to the source file's data section.
  final int offset;
}

/// The pieces of a GGUF header the splitter needs: raw byte ranges of
/// each metadata key/value (minus any pre-existing `split.*` keys) and
/// of each tensor info, plus the declared alignment.
final class _ParsedGguf {
  _ParsedGguf._({
    required this.bytes,
    required this.version,
    required this.kvRanges,
    required this.alignmentKvRange,
    required this.tensors,
    required this.alignment,
    required this.headerEnd,
  });

  final Uint8List bytes;
  final int version;
  final List<_ByteRange> kvRanges;
  final _ByteRange? alignmentKvRange;
  final List<_TensorInfo> tensors;
  final int alignment;
  final int headerEnd;

  /// Throws [GgufTruncated] when [bytes] ends mid-header and
  /// [FormatException] when the content is not a supported GGUF.
  static _ParsedGguf parse(Uint8List bytes) {
    final reader = GgufReader(bytes);
    if (reader.uint32() != ggufMagic) {
      throw const FormatException('The file is not a GGUF model.');
    }
    final version = reader.uint32();
    if (version < 2 || version > 3) {
      throw FormatException(
        'GGUF version $version is not supported for client-side '
        'splitting (only little-endian v2/v3).',
      );
    }
    final tensorCount = reader.uint64();
    final kvCount = reader.uint64();

    final kvRanges = <_ByteRange>[];
    _ByteRange? alignmentKvRange;
    var alignment = _defaultAlignment;
    for (var i = 0; i < kvCount; i++) {
      final start = reader.offset;
      final key = reader.string();
      final type = reader.uint32();
      final value = reader.numericValueOrSkip(type);
      final range = _ByteRange(start, reader.offset);
      // Drop split bookkeeping from an already-merged file; the new
      // headers write their own.
      if (key == 'split.no' ||
          key == 'split.count' ||
          key == 'split.tensors.count') {
        continue;
      }
      if (key == 'general.alignment') {
        if (value == null || value <= 0 || (value & (value - 1)) != 0) {
          throw const FormatException(
            'The GGUF declares an invalid tensor alignment.',
          );
        }
        alignment = value;
        alignmentKvRange = range;
      }
      kvRanges.add(range);
    }

    final tensors = <_TensorInfo>[];
    for (var i = 0; i < tensorCount; i++) {
      final start = reader.offset;
      reader.string(); // name
      final dims = reader.uint32();
      for (var d = 0; d < dims; d++) {
        reader.uint64();
      }
      reader.uint32(); // ggml type
      final offset = reader.uint64();
      tensors.add(
        _TensorInfo(start: start, end: reader.offset, offset: offset),
      );
    }

    return _ParsedGguf._(
      bytes: bytes,
      version: version,
      kvRanges: kvRanges,
      alignmentKvRange: alignmentKvRange,
      tensors: tensors,
      alignment: alignment,
      headerEnd: reader.offset,
    );
  }
}
