import 'package:flutter/material.dart';

import '../app_model.dart';
import '../library/byte_format.dart';
import '../library/model_entry.dart';
import '../model_presets.dart';
import 'model_editor_screen.dart';

/// The user's model library: what is on this device, what it costs in
/// storage, and what runs next.
class LibraryScreen extends StatelessWidget {
  const LibraryScreen({super.key, required this.model});

  final AppModel model;

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: model,
      builder: (context, _) => Scaffold(
        appBar: AppBar(
          title: const Text('Models'),
          bottom: PreferredSize(
            preferredSize: const Size.fromHeight(24),
            child: Padding(
              padding: const EdgeInsets.only(left: 16, bottom: 8),
              child: Align(
                alignment: Alignment.centerLeft,
                child: Text(
                  '${model.library.length} in library · '
                  '${formatBytes(model.storageUsedBytes)} on this device',
                  style: Theme.of(context).textTheme.bodySmall,
                ),
              ),
            ),
          ),
        ),
        body: model.library.isEmpty
            ? const _EmptyLibrary()
            : ListView.builder(
                padding: const EdgeInsets.only(bottom: 88),
                itemCount: model.library.length,
                itemBuilder: (context, index) =>
                    _EntryTile(model: model, entry: model.library[index]),
              ),
        floatingActionButton: FloatingActionButton.extended(
          onPressed: () => _addModel(context),
          icon: const Icon(Icons.add),
          label: const Text('Add model'),
        ),
      ),
    );
  }

  Future<void> _addModel(BuildContext context) async {
    final preset = await showModalBottomSheet<_AddChoice>(
      context: context,
      showDragHandle: true,
      isScrollControlled: true,
      builder: (context) => const _AddSheet(),
    );
    if (preset == null || !context.mounted) return;
    switch (preset) {
      case _AddCustom():
        await Navigator.of(context).push(
          MaterialPageRoute<void>(
            builder: (_) => ModelEditorScreen(model: model),
          ),
        );
      case _AddPreset(:final preset):
        await model.addEntry(
          ModelEntry(
            id: newEntryId(),
            name: preset.label,
            main: ArtifactRef(fileName: preset.file, url: preset.url),
            formatName: preset.formatName,
            enableThinking: preset.supportsThinking,
          ),
        );
    }
  }
}

/// What the add sheet returns.
sealed class _AddChoice {
  const _AddChoice();
}

final class _AddPreset extends _AddChoice {
  const _AddPreset(this.preset);
  final ModelPreset preset;
}

final class _AddCustom extends _AddChoice {
  const _AddCustom();
}

class _AddSheet extends StatelessWidget {
  const _AddSheet();

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: ListView(
        shrinkWrap: true,
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 0, 16, 8),
            child: Text(
              'Add a model',
              style: Theme.of(context).textTheme.titleMedium,
            ),
          ),
          for (final preset in modelPresets)
            ListTile(
              leading: const Icon(Icons.cloud_download_outlined),
              title: Text(preset.label),
              subtitle: Text(preset.detail),
              onTap: () => Navigator.of(context).pop(_AddPreset(preset)),
            ),
          const Divider(),
          ListTile(
            leading: const Icon(Icons.tune),
            title: const Text('Custom model…'),
            subtitle: const Text(
              'From a download URL, or GGUF files already on this device — '
              'with an optional vision projector and draft model.',
            ),
            onTap: () => Navigator.of(context).pop(const _AddCustom()),
          ),
        ],
      ),
    );
  }
}

class _EntryTile extends StatelessWidget {
  const _EntryTile({required this.model, required this.entry});

  final AppModel model;
  final ModelEntry entry;

  @override
  Widget build(BuildContext context) {
    final state = model.stateOf(entry);
    final theme = Theme.of(context);
    final busy = state == EntryState.transferring;
    return Card(
      margin: const EdgeInsets.fromLTRB(12, 6, 12, 6),
      child: ListTile(
        leading: _StateIcon(state: state),
        title: Text(entry.name),
        isThreeLine: true,
        subtitle: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(_details(entry)),
            Text(
              _stateLabel(model, entry, state),
              style: theme.textTheme.bodySmall?.copyWith(
                color: state == EntryState.failed
                    ? theme.colorScheme.error
                    : theme.textTheme.bodySmall?.color,
              ),
            ),
            if (busy)
              Padding(
                padding: const EdgeInsets.only(top: 6),
                child: LinearProgressIndicator(
                  value: model.loadProgress > 0 ? model.loadProgress : null,
                ),
              ),
          ],
        ),
        trailing: PopupMenuButton<String>(
          onSelected: (value) => _onMenu(context, value),
          itemBuilder: (context) => const [
            PopupMenuItem<String>(value: 'load', child: Text('Load')),
            PopupMenuItem<String>(value: 'edit', child: Text('Edit')),
            PopupMenuItem<String>(value: 'delete', child: Text('Delete')),
          ],
        ),
        onTap: busy ? null : () => model.loadEntry(entry.id),
      ),
    );
  }

  static String _details(ModelEntry entry) {
    final parts = <String>[
      entry.formatName,
      '${entry.contextSize} ctx',
      if (entry.projector != null) 'vision',
      if (entry.draft != null) 'MTP',
      if (entry.storedSizeBytes > 0) formatBytes(entry.storedSizeBytes),
    ];
    return parts.join(' · ');
  }

  static String _stateLabel(
    AppModel model,
    ModelEntry entry,
    EntryState state,
  ) => switch (state) {
    EntryState.loaded => 'Loaded',
    EntryState.stored => 'Downloaded — tap to load',
    EntryState.notDownloaded => 'Not downloaded — tap to download and load',
    EntryState.transferring => 'Downloading…',
    EntryState.incomplete =>
      'An imported file is missing. Edit this model to import it again.',
    EntryState.failed => model.errorFor(entry) ?? 'Failed',
  };

  Future<void> _onMenu(BuildContext context, String value) async {
    switch (value) {
      case 'load':
        await model.loadEntry(entry.id);
      case 'edit':
        await Navigator.of(context).push(
          MaterialPageRoute<void>(
            builder: (_) => ModelEditorScreen(model: model, entry: entry),
          ),
        );
      case 'delete':
        final confirmed = await showDialog<bool>(
          context: context,
          builder: (context) => AlertDialog(
            title: Text('Delete "${entry.name}"?'),
            content: Text(
              'This permanently deletes ${formatBytes(entry.storedSizeBytes)} '
              'of model files from this device. Files shared with another '
              'model in the library are kept.',
            ),
            actions: [
              TextButton(
                onPressed: () => Navigator.of(context).pop(false),
                child: const Text('Cancel'),
              ),
              FilledButton(
                onPressed: () => Navigator.of(context).pop(true),
                child: const Text('Delete'),
              ),
            ],
          ),
        );
        if (confirmed ?? false) await model.deleteEntry(entry.id);
    }
  }
}

class _StateIcon extends StatelessWidget {
  const _StateIcon({required this.state});

  final EntryState state;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return switch (state) {
      EntryState.loaded => Icon(Icons.check_circle, color: scheme.primary),
      EntryState.stored => const Icon(Icons.download_done),
      EntryState.notDownloaded => const Icon(Icons.cloud_outlined),
      EntryState.transferring => const SizedBox.square(
        dimension: 24,
        child: CircularProgressIndicator(strokeWidth: 2),
      ),
      EntryState.incomplete => Icon(Icons.help_outline, color: scheme.tertiary),
      EntryState.failed => Icon(Icons.error_outline, color: scheme.error),
    };
  }
}

class _EmptyLibrary extends StatelessWidget {
  const _EmptyLibrary();

  @override
  Widget build(BuildContext context) {
    return Center(
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 380),
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(Icons.inventory_2_outlined, size: 48),
              const SizedBox(height: 12),
              Text(
                'No models yet',
                style: Theme.of(context).textTheme.titleMedium,
              ),
              const SizedBox(height: 8),
              const Text(
                'Add one from the catalog, from a download URL, or from '
                'GGUF files already on this device. Everything you add is '
                'stored here and can be deleted from here.',
                textAlign: TextAlign.center,
              ),
            ],
          ),
        ),
      ),
    );
  }
}
