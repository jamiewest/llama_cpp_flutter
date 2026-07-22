import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:llama_flutter/llama_flutter.dart';

void main() {
  late Directory root;
  late SnapshotStore store;

  setUp(() async {
    root = await Directory.systemTemp.createTemp('snapshot_store_test');
    store = SnapshotStore(root.path);
  });

  tearDown(() async {
    await root.delete(recursive: true);
  });

  Future<void> writeState(String modelKey, String agentId) =>
      File(store.statePathFor(modelKey, agentId)).writeAsString('state');

  test('find returns null when nothing was recorded', () async {
    expect(await store.find('model', 'db'), isNull);
  });

  test('record then find round-trips the snapshot metadata', () async {
    await writeState('model', 'db');
    await store.record('model', 'db', tokens: 120, contextTokens: 4096);

    final snapshot = await store.find('model', 'db');
    expect(snapshot, isNotNull);
    expect(snapshot!.tokens, 120);
    expect(snapshot.contextTokens, 4096);
    expect(snapshot.statePath, store.statePathFor('model', 'db'));
  });

  test('find hides snapshots whose state file is missing', () async {
    await store.record('model', 'db', tokens: 120, contextTokens: 4096);
    expect(await store.find('model', 'db'), isNull);
  });

  test('find filters snapshots larger than maxTokens', () async {
    await writeState('model', 'db');
    await store.record('model', 'db', tokens: 7000, contextTokens: 8192);

    expect(await store.find('model', 'db', maxTokens: 6144), isNull);
    expect(await store.find('model', 'db', maxTokens: 8192), isNotNull);
  });

  test('invalidate removes both files', () async {
    await writeState('model', 'db');
    await store.record('model', 'db', tokens: 120, contextTokens: 4096);

    await store.invalidate('model', 'db');
    expect(await store.find('model', 'db'), isNull);
    expect(File(store.statePathFor('model', 'db')).existsSync(), isFalse);
  });

  test('invalidateWhere drops only matching snapshots', () async {
    await writeState('model', 'big');
    await store.record('model', 'big', tokens: 7000, contextTokens: 8192);
    await writeState('model', 'small');
    await store.record('model', 'small', tokens: 500, contextTokens: 8192);

    await store.invalidateWhere('model', (s) => s.tokens > 6144);

    expect(await store.find('model', 'big'), isNull);
    expect(await store.find('model', 'small'), isNotNull);
  });

  test('sanitizes model keys and agent ids in paths', () async {
    final path = store.statePathFor('hf.co/model:v1', 'db agent');
    expect(path, isNot(contains('/model:')));
    expect(path, endsWith('db_agent.llstate'));
  });
}
