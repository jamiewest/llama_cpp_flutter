import 'dart:convert';

import 'package:shared_preferences/shared_preferences.dart';

import 'model_entry.dart';

/// The library as it was last saved.
typedef CatalogSnapshot = ({List<ModelEntry> entries, String? activeId});

/// Where the model library is persisted between launches.
///
/// An interface so tests can run the library against memory instead of the
/// platform's preference store.
abstract interface class CatalogStorage {
  /// The stored catalog document, or null when nothing has been saved.
  Future<String?> read();

  /// Replaces the stored catalog document with [document].
  Future<void> write(String document);
}

/// Catalog storage backed by the platform's shared preferences.
final class PreferencesCatalogStorage implements CatalogStorage {
  /// Creates preference-backed storage.
  const PreferencesCatalogStorage();

  static const String _key = 'model_library';

  @override
  Future<String?> read() async =>
      (await SharedPreferences.getInstance()).getString(_key);

  @override
  Future<void> write(String document) async =>
      (await SharedPreferences.getInstance()).setString(_key, document);
}

/// Catalog storage that keeps the document in memory, for tests.
final class MemoryCatalogStorage implements CatalogStorage {
  /// Creates memory-backed storage, optionally pre-seeded with [document].
  MemoryCatalogStorage([this.document]);

  /// The stored document.
  String? document;

  @override
  Future<String?> read() async => document;

  @override
  Future<void> write(String value) async => document = value;
}

/// Reads and writes the model library.
///
/// A corrupt or unreadable document is treated as an empty library rather
/// than a fatal error: the demo should still start, and the user can add
/// models again.
final class ModelCatalog {
  /// Creates a catalog over [storage].
  const ModelCatalog(this._storage);

  final CatalogStorage _storage;

  /// The saved library, or an empty one when nothing is stored.
  Future<CatalogSnapshot> load() async {
    final document = await _storage.read();
    if (document == null || document.isEmpty) {
      return (entries: <ModelEntry>[], activeId: null);
    }
    try {
      final json = jsonDecode(document) as Map<String, Object?>;
      final entries = <ModelEntry>[
        for (final entry in json['entries']! as List<Object?>)
          ModelEntry.fromJson(entry! as Map<String, Object?>),
      ];
      return (entries: entries, activeId: json['activeId'] as String?);
    } catch (_) {
      return (entries: <ModelEntry>[], activeId: null);
    }
  }

  /// Persists [entries] and which of them was last loaded.
  Future<void> save(List<ModelEntry> entries, String? activeId) =>
      _storage.write(
        jsonEncode(<String, Object?>{
          'entries': <Object?>[for (final entry in entries) entry.toJson()],
          'activeId': ?activeId,
        }),
      );
}
