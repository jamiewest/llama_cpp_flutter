import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:llama_cpp_flutter_example/src/app_model.dart';
import 'package:llama_cpp_flutter_example/src/library/model_catalog.dart';
import 'package:llama_cpp_flutter_example/src/library/model_entry.dart';
import 'package:llama_cpp_flutter_example/src/screens/library_screen.dart';
import 'package:llama_cpp_flutter_example/src/screens/model_editor_screen.dart';

import 'fakes.dart';

void main() {
  late FakeArtifactStore store;
  late FakeLlamaRuntime runtime;
  late AppModel model;

  final weightsUrl = Uri.parse('https://example.test/weights.gguf');

  ModelEntry entry({String id = 'entry-1', String name = 'Test model'}) =>
      ModelEntry(
        id: id,
        name: name,
        main: ArtifactRef(fileName: 'weights.gguf', url: weightsUrl),
        formatName: 'qwen',
      );

  setUp(() async {
    store = FakeArtifactStore();
    runtime = FakeLlamaRuntime();
    model = AppModel(
      runtime: runtime,
      store: store,
      catalogStorage: MemoryCatalogStorage(),
    );
    await model.initialize();
  });

  tearDown(() => model.dispose());

  Future<void> showLibrary(WidgetTester tester) async {
    await tester.pumpWidget(MaterialApp(home: LibraryScreen(model: model)));
    await tester.pumpAndSettle();
  }

  testWidgets('invites the first model when the library is empty', (
    tester,
  ) async {
    await showLibrary(tester);

    expect(find.text('No models yet'), findsOneWidget);
    expect(find.text('Add model'), findsOneWidget);
  });

  testWidgets('shows each model with its state and storage cost', (
    tester,
  ) async {
    await model.addEntry(entry());
    await showLibrary(tester);

    expect(find.text('Test model'), findsOneWidget);
    expect(find.textContaining('qwen'), findsOneWidget);
    expect(find.textContaining('4096 ctx'), findsOneWidget);
    expect(find.textContaining('Not downloaded'), findsOneWidget);
    expect(find.textContaining('1 in library'), findsOneWidget);

    await model.loadEntry('entry-1');
    await tester.pumpAndSettle();

    expect(find.text('Loaded'), findsOneWidget);
    // The fake store reports 100 bytes per artifact.
    expect(find.textContaining('100 B'), findsWidgets);
  });

  testWidgets('tapping a model downloads and loads it', (tester) async {
    await model.addEntry(entry());
    await showLibrary(tester);

    await tester.tap(find.text('Test model'));
    await tester.pumpAndSettle();

    expect(store.fetched, [weightsUrl]);
    expect(runtime.loads, hasLength(1));
    expect(model.status, ModelStatus.ready);
  });

  testWidgets('reports a failed load on the model itself', (tester) async {
    await model.addEntry(entry());
    runtime.failWith = StateError('out of memory');
    await showLibrary(tester);

    await tester.tap(find.text('Test model'));
    await tester.pumpAndSettle();

    expect(find.textContaining('out of memory'), findsOneWidget);
  });

  testWidgets('deleting asks first, then removes the model', (tester) async {
    await model.addEntry(entry());
    await model.loadEntry('entry-1');
    await showLibrary(tester);

    await tester.tap(find.byType(PopupMenuButton<String>));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Delete'));
    await tester.pumpAndSettle();

    expect(find.text('Delete "Test model"?'), findsOneWidget);
    expect(model.library, hasLength(1), reason: 'not deleted until confirmed');

    await tester.tap(find.widgetWithText(FilledButton, 'Delete'));
    await tester.pumpAndSettle();

    expect(model.library, isEmpty);
    expect(store.artifacts, isEmpty);
    expect(find.text('No models yet'), findsOneWidget);
  });

  testWidgets('cancelling the delete dialog keeps the model', (tester) async {
    await model.addEntry(entry());
    await showLibrary(tester);

    await tester.tap(find.byType(PopupMenuButton<String>));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Delete'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Cancel'));
    await tester.pumpAndSettle();

    expect(model.library, hasLength(1));
  });

  group('editor', () {
    Future<void> showEditor(WidgetTester tester, {ModelEntry? existing}) async {
      await tester.pumpWidget(
        MaterialApp(
          home: ModelEditorScreen(model: model, entry: existing),
        ),
      );
      await tester.pumpAndSettle();
    }

    testWidgets('refuses to save a model with no weights', (tester) async {
      await showEditor(tester);

      await tester.tap(find.widgetWithText(OutlinedButton, 'Add'));
      await tester.pumpAndSettle();

      expect(find.textContaining('A model needs its weights'), findsOneWidget);
      expect(model.library, isEmpty);
    });

    testWidgets('adds a model from a URL', (tester) async {
      await showEditor(tester);

      await tester.enterText(find.byType(TextField).at(1), '$weightsUrl');
      await tester.tap(find.widgetWithText(OutlinedButton, 'Add'));
      await tester.pumpAndSettle();

      expect(model.library.single.main.url, weightsUrl);
      // The name defaults to the weights file name.
      expect(model.library.single.name, 'weights.gguf');
      expect(store.fetched, isEmpty, reason: 'adding does not download');
    });

    testWidgets('edits an existing model without reloading it', (tester) async {
      await model.addEntry(entry());
      await model.loadEntry('entry-1');
      await showEditor(tester, existing: model.library.single);

      await tester.enterText(find.byType(TextField).first, 'Renamed');
      await tester.tap(find.widgetWithText(OutlinedButton, 'Save'));
      await tester.pumpAndSettle();

      expect(model.library.single.name, 'Renamed');
      expect(runtime.loads, hasLength(1));
    });

    testWidgets('save and reload restarts the session', (tester) async {
      await model.addEntry(entry());
      await model.loadEntry('entry-1');
      await showEditor(tester, existing: model.library.single);

      await tester.tap(find.widgetWithText(FilledButton, 'Save and reload'));
      await tester.pumpAndSettle();

      expect(runtime.loads, hasLength(2));
    });
  });
}
