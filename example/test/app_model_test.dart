import 'package:flutter_test/flutter_test.dart';
import 'package:llama_cpp_flutter/gguf.dart';
import 'package:llama_cpp_flutter_example/src/app_model.dart';
import 'package:llama_cpp_flutter_example/src/library/model_catalog.dart';
import 'package:llama_cpp_flutter_example/src/library/model_entry.dart';

import 'fakes.dart';

void main() {
  late FakeArtifactStore store;
  late FakeLlamaRuntime runtime;
  late MemoryCatalogStorage catalogStorage;
  late AppModel model;

  final weightsUrl = Uri.parse('https://example.test/weights.gguf');
  final mmprojUrl = Uri.parse('https://example.test/mmproj.gguf');
  final draftUrl = Uri.parse('https://example.test/draft.gguf');

  ModelEntry entry({
    String id = 'entry-1',
    String name = 'Test model',
    Uri? main,
    ArtifactRef? mainRef,
    ArtifactRef? projector,
    ArtifactRef? draft,
  }) => ModelEntry(
    id: id,
    name: name,
    main:
        mainRef ??
        ArtifactRef(fileName: 'weights.gguf', url: main ?? weightsUrl),
    projector: projector,
    draft: draft,
  );

  setUp(() async {
    store = FakeArtifactStore();
    runtime = FakeLlamaRuntime();
    catalogStorage = MemoryCatalogStorage();
    model = AppModel(
      runtime: runtime,
      store: store,
      catalogStorage: catalogStorage,
    );
    await model.initialize();
  });

  tearDown(() => model.dispose());

  group('library', () {
    test('starts empty and reports storage use', () async {
      expect(model.isInitialized, isTrue);
      expect(model.library, isEmpty);
      expect(model.storageUsedBytes, 0);
    });

    test('adds an entry without downloading anything', () async {
      await model.addEntry(entry());

      expect(model.library.single.name, 'Test model');
      expect(store.fetched, isEmpty);
      expect(model.stateOf(model.library.single), EntryState.notDownloaded);
    });

    test('restores the saved library on the next launch', () async {
      await model.addEntry(
        entry(
          projector: ArtifactRef(fileName: 'mmproj.gguf', url: mmprojUrl),
        ),
      );

      final reopened = AppModel(
        runtime: FakeLlamaRuntime(),
        store: store,
        catalogStorage: catalogStorage,
      );
      addTearDown(reopened.dispose);
      await reopened.initialize();

      expect(reopened.library.single.id, 'entry-1');
      expect(reopened.library.single.projector?.url, mmprojUrl);
    });

    test('unstores artifacts that vanished from managed storage', () async {
      await model.addEntry(entry());
      await model.loadEntry('entry-1');
      expect(model.library.single.main.isStored, isTrue);

      // The browser evicted origin-private storage between launches.
      store.artifacts.clear();
      final reopened = AppModel(
        runtime: FakeLlamaRuntime(),
        store: store,
        catalogStorage: catalogStorage,
      );
      addTearDown(reopened.dispose);
      await reopened.initialize();

      final restored = reopened.library.single;
      expect(restored.main.isStored, isFalse);
      expect(reopened.stateOf(restored), EntryState.notDownloaded);
    });
  });

  group('loadEntry', () {
    test('downloads every artifact and loads the resolved paths', () async {
      await model.addEntry(
        entry(
          projector: ArtifactRef(fileName: 'mmproj.gguf', url: mmprojUrl),
          draft: ArtifactRef(fileName: 'draft.gguf', url: draftUrl),
        ),
      );

      await model.loadEntry('entry-1');

      expect(store.fetched, [weightsUrl, mmprojUrl, draftUrl]);
      final load = runtime.loads.single;
      expect(load.model, '/managed/${stableArtifactFileName(weightsUrl)}');
      expect(load.mmproj, '/managed/${stableArtifactFileName(mmprojUrl)}');
      expect(load.draft, '/managed/${stableArtifactFileName(draftUrl)}');
      expect(load.spec.displayName, 'Test model');
      expect(load.spec.contextSize, 4096);
      expect(model.status, ModelStatus.ready);
      expect(model.loadedModelName, 'Test model');
      expect(model.stateOf(model.library.single), EntryState.loaded);
    });

    test('remembers stored artifacts instead of downloading again', () async {
      await model.addEntry(entry());
      await model.loadEntry('entry-1');
      await model.loadEntry('entry-1');

      expect(store.fetched, [weightsUrl]);
      expect(runtime.loads.length, 2);
    });

    test('frees the previous session before loading the next', () async {
      await model.addEntry(entry());
      await model.addEntry(entry(id: 'entry-2', name: 'Second'));

      await model.loadEntry('entry-1');
      await model.loadEntry('entry-2');

      expect(runtime.sessions.first.disposed, isTrue);
      expect(runtime.sessions.last.disposed, isFalse);
      expect(model.activeEntry?.id, 'entry-2');
    });

    test('surfaces a failure on the entry and stays unloaded', () async {
      await model.addEntry(entry());
      runtime.failWith = StateError('no GPU');

      await model.loadEntry('entry-1');

      expect(model.status, ModelStatus.error);
      expect(model.stateOf(model.library.single), EntryState.failed);
      expect(model.errorFor(model.library.single), contains('no GPU'));
      expect(model.loadedModelName, isNull);
    });

    test('reports vision support only with a projector', () async {
      await model.addEntry(entry());
      await model.loadEntry('entry-1');
      expect(model.supportsImages, isFalse);

      await model.addEntry(
        entry(
          id: 'entry-2',
          projector: ArtifactRef(fileName: 'mmproj.gguf', url: mmprojUrl),
        ),
      );
      await model.loadEntry('entry-2');
      expect(model.supportsImages, isTrue);
    });

    test('fails clearly when an imported file is gone', () async {
      await model.addEntry(
        entry(mainRef: const ArtifactRef(fileName: 'local.gguf')),
      );

      await model.loadEntry('entry-1');

      expect(model.status, ModelStatus.error);
      expect(model.errorFor(model.library.single), contains('Import it again'));
    });
  });

  group('deleteEntry', () {
    test('unloads the entry and deletes its artifacts', () async {
      await model.addEntry(entry());
      await model.loadEntry('entry-1');
      final key = stableArtifactFileName(weightsUrl);

      await model.deleteEntry('entry-1');

      expect(model.library, isEmpty);
      expect(store.deleted, [key]);
      expect(store.artifacts, isEmpty);
      expect(model.activeEntry, isNull);
      expect(model.status, ModelStatus.empty);
      expect(runtime.sessions.single.disposed, isTrue);
    });

    test('keeps an artifact another entry still references', () async {
      // Two entries over the same weights URL share one stored file.
      await model.addEntry(entry());
      await model.addEntry(entry(id: 'entry-2', name: 'Same weights'));
      await model.loadEntry('entry-1');
      await model.loadEntry('entry-2');

      await model.deleteEntry('entry-2');

      expect(store.deleted, isEmpty);
      expect(store.artifacts.keys, [stableArtifactFileName(weightsUrl)]);
      expect(model.library.single.id, 'entry-1');
    });

    test('forgets the entry even when nothing was downloaded', () async {
      await model.addEntry(entry());

      await model.deleteEntry('entry-1');

      expect(model.library, isEmpty);
      expect(store.deleted, isEmpty);
    });
  });

  group('discardUnreferencedArtifacts', () {
    test('reclaims an import nothing ended up referencing', () async {
      // What the editor does when the user picks a file and then backs out.
      final orphan = await store.importFile('/tmp/picked.gguf');
      final kept = await store.importFile('/tmp/used.gguf');
      await model.addEntry(
        entry(
          mainRef: ArtifactRef(
            fileName: kept.fileName,
            key: kept.key,
            sizeBytes: kept.sizeBytes,
          ),
        ),
      );

      await model.discardUnreferencedArtifacts(<String>[orphan.key, kept.key]);

      expect(store.deleted, [orphan.key]);
      expect(store.artifacts.keys, [kept.key]);
    });

    test('does nothing when there is nothing to reclaim', () async {
      await model.discardUnreferencedArtifacts(const <String>[]);

      expect(store.deleted, isEmpty);
    });
  });

  group('saveEntry', () {
    test('persists edits without touching the session', () async {
      await model.addEntry(entry());
      await model.loadEntry('entry-1');

      await model.saveEntry(
        model.library.single.copyWith(name: 'Renamed', contextSize: 8192),
      );

      expect(model.library.single.name, 'Renamed');
      expect(runtime.loads.length, 1);
    });

    test('deletes the artifact an edit replaced', () async {
      await model.addEntry(entry());
      await model.loadEntry('entry-1');
      final replaced = stableArtifactFileName(weightsUrl);
      final newUrl = Uri.parse('https://example.test/newer.gguf');

      await model.saveEntry(
        model.library.single.copyWith(
          main: ArtifactRef(fileName: 'newer.gguf', url: newUrl),
        ),
      );

      expect(store.deleted, [replaced]);
      expect(store.artifacts, isEmpty);
      expect(model.storageUsedBytes, 0);
    });

    test('keeps a replaced artifact another entry still uses', () async {
      await model.addEntry(entry());
      await model.addEntry(entry(id: 'entry-2', name: 'Same weights'));
      await model.loadEntry('entry-1');
      await model.loadEntry('entry-2');

      await model.saveEntry(
        model.library.first.copyWith(
          main: ArtifactRef(
            fileName: 'newer.gguf',
            url: Uri.parse('https://example.test/newer.gguf'),
          ),
        ),
      );

      expect(store.deleted, isEmpty);
      expect(store.artifacts.keys, [stableArtifactFileName(weightsUrl)]);
    });

    test('deletes a companion an edit removed', () async {
      await model.addEntry(
        entry(
          projector: ArtifactRef(fileName: 'mmproj.gguf', url: mmprojUrl),
        ),
      );
      await model.loadEntry('entry-1');

      await model.saveEntry(
        model.library.single.copyWith(clearProjector: true),
      );

      expect(store.deleted, [stableArtifactFileName(mmprojUrl)]);
      expect(store.artifacts.keys, [stableArtifactFileName(weightsUrl)]);
    });

    test('reloads on request, applying the new configuration', () async {
      await model.addEntry(entry());
      await model.loadEntry('entry-1');

      await model.saveEntry(
        model.library.single.copyWith(contextSize: 8192),
        reload: true,
      );

      expect(runtime.loads.length, 2);
      expect(runtime.loads.last.spec.contextSize, 8192);
    });
  });
}
