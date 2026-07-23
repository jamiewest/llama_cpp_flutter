import 'package:flutter_test/flutter_test.dart';
import 'package:llama_cpp_flutter_example/src/library/byte_format.dart';
import 'package:llama_cpp_flutter_example/src/library/model_catalog.dart';
import 'package:llama_cpp_flutter_example/src/library/model_entry.dart';

void main() {
  final weightsUrl = Uri.parse('https://example.test/weights.gguf');
  final mmprojUrl = Uri.parse('https://example.test/mmproj.gguf');
  final draftUrl = Uri.parse('https://example.test/draft.gguf');

  ModelEntry fullEntry() => ModelEntry(
    id: 'entry-1',
    name: 'Vision model',
    main: ArtifactRef(
      fileName: 'weights.gguf',
      url: weightsUrl,
      key: 'abc-weights.gguf',
      sizeBytes: 2000,
    ),
    projector: ArtifactRef(
      fileName: 'mmproj.gguf',
      url: mmprojUrl,
      key: 'def-mmproj.gguf',
      sizeBytes: 500,
    ),
    draft: ArtifactRef(fileName: 'draft.gguf', url: draftUrl),
    formatName: 'qwen',
    contextSize: 8192,
    gpuOffload: false,
    enableThinking: false,
  );

  group('ModelEntry', () {
    test('round-trips through JSON', () {
      final restored = ModelEntry.fromJson(fullEntry().toJson());

      expect(restored.id, 'entry-1');
      expect(restored.name, 'Vision model');
      expect(restored.main.url, weightsUrl);
      expect(restored.main.key, 'abc-weights.gguf');
      expect(restored.main.sizeBytes, 2000);
      expect(restored.projector?.url, mmprojUrl);
      expect(restored.draft?.url, draftUrl);
      expect(restored.draft?.isStored, isFalse);
      expect(restored.formatName, 'qwen');
      expect(restored.contextSize, 8192);
      expect(restored.gpuOffload, isFalse);
      expect(restored.enableThinking, isFalse);
    });

    test('sizes only what is actually stored', () {
      final entry = fullEntry();

      expect(entry.storedSizeBytes, 2500);
      expect(entry.isReady(), isFalse, reason: 'the draft is not downloaded');
      expect(entry.hasLostArtifact(), isFalse);
    });

    test('ignores the draft model where it will never be loaded', () {
      // On the web the draft is never fetched, so waiting for it would
      // leave the entry stuck at "not downloaded" forever.
      final entry = fullEntry();

      expect(entry.isReady(includeDraft: false), isTrue);
      expect(entry.isReady(), isFalse);
    });

    test('ignores a lost draft where it will never be loaded', () {
      final entry = fullEntry().copyWith(
        draft: const ArtifactRef(fileName: 'local-draft.gguf'),
      );

      expect(entry.hasLostArtifact(), isTrue);
      expect(entry.hasLostArtifact(includeDraft: false), isFalse);
    });

    test('flags an imported artifact that is no longer stored', () {
      const entry = ModelEntry(
        id: 'entry-1',
        name: 'Imported',
        main: ArtifactRef(fileName: 'local.gguf'),
      );

      expect(entry.hasLostArtifact(), isTrue);
      expect(entry.isReady(), isFalse);
    });

    test('copyWith removes companions that a null cannot express', () {
      final stripped = fullEntry().copyWith(
        clearProjector: true,
        clearDraft: true,
      );

      expect(stripped.projector, isNull);
      expect(stripped.draft, isNull);
      expect(stripped.main.key, 'abc-weights.gguf');
      expect(stripped.id, 'entry-1');
    });

    test('clearing an artifact key marks it for transfer again', () {
      final ref = fullEntry().main.copyWith(clearKey: true);

      expect(ref.isStored, isFalse);
      expect(ref.sizeBytes, 0);
      expect(ref.url, weightsUrl, reason: 'it can be downloaded again');
      expect(ref.isLost, isFalse);
    });
  });

  group('toSpec', () {
    test('carries the entry\'s configuration and artifact URLs', () {
      final spec = fullEntry().toSpec();

      expect(spec.id, 'entry-1');
      expect(spec.displayName, 'Vision model');
      expect(spec.modelUrl, weightsUrl);
      expect(spec.mmprojUrl, mmprojUrl);
      expect(spec.draftUrl, draftUrl);
      expect(spec.contextSize, 8192);
      expect(spec.gpuLayers, 0);
      expect(spec.enableThinking, isFalse);
    });

    test('drops the draft model where it is unsupported', () {
      final spec = fullEntry().toSpec(includeDraft: false);

      // The web runtime rejects a spec that declares a draft at all, so an
      // entry with one still has to produce a loadable spec there.
      expect(spec.draftUrl, isNull);
      expect(spec.mmprojUrl, mmprojUrl);
    });

    test('stands in a placeholder URL for an imported model', () {
      const entry = ModelEntry(
        id: 'entry-1',
        name: 'Imported',
        main: ArtifactRef(fileName: 'local.gguf', key: 'local0-local.gguf'),
      );

      // Nothing fetches it — the artifact reaches loadModel as a path — but
      // ModelSpec requires a URL.
      expect(entry.toSpec().modelUrl.toString(), contains('local0-local'));
    });

    test('falls back to a known format for an unknown name', () {
      final spec = fullEntry().copyWith(formatName: 'not-a-format').toSpec();

      expect(spec.format, isNotNull);
    });
  });

  group('ModelCatalog', () {
    test('round-trips the library and the active entry', () async {
      final storage = MemoryCatalogStorage();
      final catalog = ModelCatalog(storage);

      await catalog.save(<ModelEntry>[fullEntry()], 'entry-1');
      final snapshot = await catalog.load();

      expect(snapshot.entries.single.name, 'Vision model');
      expect(snapshot.activeId, 'entry-1');
    });

    test('starts empty when nothing is saved', () async {
      final snapshot = await ModelCatalog(MemoryCatalogStorage()).load();

      expect(snapshot.entries, isEmpty);
      expect(snapshot.activeId, isNull);
    });

    test('starts empty rather than failing on a corrupt document', () async {
      final catalog = ModelCatalog(MemoryCatalogStorage('{not json'));

      expect((await catalog.load()).entries, isEmpty);
    });
  });

  group('formatBytes', () {
    test('scales to the unit that reads best', () {
      expect(formatBytes(0), '0 MB');
      expect(formatBytes(900), '900 B');
      expect(formatBytes(1024 * 1024 * 700), '700 MB');
      expect(formatBytes((1024 * 1024 * 1024 * 1.2).round()), '1.2 GB');
    });
  });
}
