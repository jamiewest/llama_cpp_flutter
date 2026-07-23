import 'dart:async';

import 'package:file_selector/file_selector.dart';
import 'package:flutter/material.dart';
import 'package:llama_cpp_flutter/llama_cpp_flutter.dart';

import '../app_model.dart';
import '../library/byte_format.dart';
import '../library/model_entry.dart';

/// Adds a model to the library, or edits one that is already in it.
///
/// Every artifact slot takes either a download URL or a file already on
/// this device; picking a file imports it into managed storage right away,
/// so the library owns a copy it can account for and delete.
class ModelEditorScreen extends StatefulWidget {
  const ModelEditorScreen({super.key, required this.model, this.entry});

  final AppModel model;

  /// The entry being edited, or null when adding a new one.
  final ModelEntry? entry;

  @override
  State<ModelEditorScreen> createState() => _ModelEditorScreenState();
}

class _ModelEditorScreenState extends State<ModelEditorScreen> {
  late final TextEditingController _name;
  late final _Slot _main;
  late final _Slot _projector;
  late final _Slot _draft;
  late String _formatName;
  late int _contextSize;
  late bool _gpuOffload;
  late bool _enableThinking;
  String? _validationError;

  bool get _isNew => widget.entry == null;

  bool get _isLoaded =>
      !_isNew && widget.model.activeEntry?.id == widget.entry!.id;

  @override
  void initState() {
    super.initState();
    final entry = widget.entry;
    _name = TextEditingController(text: entry?.name ?? '');
    _main = _Slot(entry?.main);
    _projector = _Slot(entry?.projector);
    _draft = _Slot(entry?.draft);
    _formatName = entry?.formatName ?? 'chatml';
    _contextSize = entry?.contextSize ?? 4096;
    _gpuOffload = entry?.gpuOffload ?? true;
    _enableThinking = entry?.enableThinking ?? true;
  }

  @override
  void dispose() {
    // Files picked here were imported into managed storage on the spot.
    // Whatever the saved entry does not reference — because the user backed
    // out, or replaced one pick with another — is reclaimed now.
    unawaited(
      widget.model.discardUnreferencedArtifacts(<String>[
        ..._main.importedKeys,
        ..._projector.importedKeys,
        ..._draft.importedKeys,
      ]),
    );
    _name.dispose();
    _main.dispose();
    _projector.dispose();
    _draft.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      appBar: AppBar(title: Text(_isNew ? 'Add model' : 'Edit model')),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(16, 16, 16, 96),
        children: [
          TextField(
            controller: _name,
            decoration: const InputDecoration(
              labelText: 'Name',
              helperText: 'Defaults to the weights file name.',
              border: OutlineInputBorder(),
            ),
          ),
          const SizedBox(height: 24),
          Text('Artifacts', style: theme.textTheme.titleMedium),
          const SizedBox(height: 8),
          _ArtifactSlot(
            slot: _main,
            model: widget.model,
            label: 'Weights (GGUF)',
            hint: 'https://…/model.gguf',
          ),
          _ArtifactSlot(
            slot: _projector,
            model: widget.model,
            label: 'Vision projector (mmproj, optional)',
            hint: 'https://…/mmproj.gguf',
            description: 'Lets this model accept image attachments.',
            optional: true,
          ),
          _ArtifactSlot(
            slot: _draft,
            model: widget.model,
            label: 'Draft model (MTP, optional)',
            hint: 'https://…/draft.gguf',
            description: AppModel.supportsDraftModels
                ? 'Speculative decoding: a small model drafts tokens the '
                      'main model verifies.'
                : 'Speculative decoding is native-only — the web runtime '
                      'cannot stage a second GGUF alongside the weights.',
            optional: true,
            enabled: AppModel.supportsDraftModels,
          ),
          const SizedBox(height: 16),
          Text('Configuration', style: theme.textTheme.titleMedium),
          const SizedBox(height: 8),
          DropdownButtonFormField<String>(
            initialValue: _formatName,
            decoration: const InputDecoration(
              labelText: 'Chat format',
              border: OutlineInputBorder(),
            ),
            items: [
              for (final name in supportedChatFormatNames.toList()..sort())
                DropdownMenuItem<String>(value: name, child: Text(name)),
            ],
            onChanged: (value) =>
                setState(() => _formatName = value ?? 'chatml'),
          ),
          const SizedBox(height: 16),
          DropdownButtonFormField<int>(
            initialValue: _contextSize,
            decoration: const InputDecoration(
              labelText: 'Context size',
              border: OutlineInputBorder(),
            ),
            items: const [
              DropdownMenuItem<int>(value: 2048, child: Text('2048 tokens')),
              DropdownMenuItem<int>(value: 4096, child: Text('4096 tokens')),
              DropdownMenuItem<int>(value: 8192, child: Text('8192 tokens')),
              DropdownMenuItem<int>(value: 16384, child: Text('16384 tokens')),
            ],
            onChanged: (value) => setState(() => _contextSize = value ?? 4096),
          ),
          SwitchListTile(
            contentPadding: EdgeInsets.zero,
            title: const Text('GPU offload'),
            subtitle: const Text('Runs the layers on Metal where available.'),
            value: _gpuOffload,
            onChanged: (value) => setState(() => _gpuOffload = value),
          ),
          SwitchListTile(
            contentPadding: EdgeInsets.zero,
            title: const Text('Thinking'),
            subtitle: const Text(
              'Lets the model reason before answering, for families that '
              'have a reasoning channel.',
            ),
            value: _enableThinking,
            onChanged: (value) => setState(() => _enableThinking = value),
          ),
        ],
      ),
      // The buttons — and any complaint about the form — stay in view; the
      // form itself is longer than a phone screen.
      bottomNavigationBar: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              if (_validationError case final error?)
                Padding(
                  padding: const EdgeInsets.only(bottom: 12),
                  child: Text(
                    error,
                    style: TextStyle(color: theme.colorScheme.error),
                  ),
                ),
              Row(
                children: [
                  Expanded(
                    child: OutlinedButton(
                      onPressed: () => _save(reload: false),
                      child: Text(_isNew ? 'Add' : 'Save'),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: FilledButton(
                      onPressed: () => _save(reload: true),
                      child: Text(
                        _isLoaded ? 'Save and reload' : 'Save and load',
                      ),
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }

  Future<void> _save({required bool reload}) async {
    final main = _main.toRef();
    if (main == null) {
      setState(
        () => _validationError =
            'A model needs its weights: paste a GGUF URL or choose a file.',
      );
      return;
    }
    final projector = _projector.toRef();
    final draft = AppModel.supportsDraftModels ? _draft.toRef() : null;
    final name = _name.text.trim();
    final entry = ModelEntry(
      id: widget.entry?.id ?? newEntryId(),
      name: name.isEmpty ? main.fileName : name,
      main: main,
      projector: projector,
      draft: draft,
      formatName: _formatName,
      contextSize: _contextSize,
      gpuOffload: _gpuOffload,
      enableThinking: _enableThinking,
    );

    final navigator = Navigator.of(context);
    if (_isNew) {
      await widget.model.addEntry(entry);
      if (reload) unawaited(widget.model.loadEntry(entry.id));
    } else {
      await widget.model.saveEntry(entry, reload: reload);
    }
    if (mounted) navigator.pop();
  }
}

/// The editable state of one artifact slot: a URL, or a file imported into
/// managed storage.
class _Slot {
  _Slot(this.original)
    : url = TextEditingController(text: original?.url?.toString() ?? ''),
      imported = original?.url == null ? original : null;

  /// What the entry referenced when the editor opened.
  final ArtifactRef? original;

  final TextEditingController url;

  /// Set when this slot holds a file rather than a URL.
  ArtifactRef? imported;

  /// Keys of every file imported through this slot while the editor was
  /// open, so ones the saved entry does not use can be reclaimed.
  final List<String> importedKeys = <String>[];

  /// Progress of an import in flight, or null when none is.
  double? importProgress;

  bool get isEmpty => imported == null && url.text.trim().isEmpty;

  /// Whether this slot still points at the downloaded artifact it opened
  /// with.
  bool get isDownloaded =>
      original != null &&
      original!.isStored &&
      original!.url != null &&
      original!.url.toString() == url.text.trim();

  /// This slot as a reference, or null when it is empty.
  ///
  /// An unchanged URL keeps the artifact it already downloaded; a changed
  /// one drops the key so the new file is fetched on the next load.
  ArtifactRef? toRef() {
    if (imported case final imported?) return imported;
    final text = url.text.trim();
    if (text.isEmpty) return null;
    final uri = Uri.tryParse(text);
    if (uri == null || !uri.hasScheme) return null;
    if (original?.url == uri) return original;
    final segments = uri.pathSegments;
    return ArtifactRef(
      fileName: segments.isEmpty ? 'model.gguf' : segments.last,
      url: uri,
    );
  }

  void clear() {
    imported = null;
    url.clear();
  }

  void dispose() => url.dispose();
}

class _ArtifactSlot extends StatefulWidget {
  const _ArtifactSlot({
    required this.slot,
    required this.model,
    required this.label,
    required this.hint,
    this.description,
    this.optional = false,
    this.enabled = true,
  });

  final _Slot slot;
  final AppModel model;
  final String label;
  final String hint;
  final String? description;
  final bool optional;
  final bool enabled;

  @override
  State<_ArtifactSlot> createState() => _ArtifactSlotState();
}

class _ArtifactSlotState extends State<_ArtifactSlot> {
  String? _importError;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final slot = widget.slot;
    return Padding(
      padding: const EdgeInsets.only(bottom: 20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(widget.label, style: theme.textTheme.labelLarge),
          if (widget.description case final description?)
            Padding(
              padding: const EdgeInsets.only(top: 2, bottom: 6),
              child: Text(description, style: theme.textTheme.bodySmall),
            ),
          if (slot.imported case final imported?)
            ListTile(
              contentPadding: EdgeInsets.zero,
              leading: const Icon(Icons.insert_drive_file_outlined),
              title: Text(imported.fileName),
              subtitle: Text(
                imported.isStored
                    ? 'In storage · ${formatBytes(imported.sizeBytes)}'
                    : 'Missing from storage — choose the file again',
              ),
              trailing: IconButton(
                tooltip: 'Remove',
                icon: const Icon(Icons.close),
                onPressed: widget.enabled ? () => setState(slot.clear) : null,
              ),
            )
          else ...[
            TextField(
              controller: slot.url,
              enabled: widget.enabled,
              decoration: InputDecoration(
                hintText: widget.hint,
                border: const OutlineInputBorder(),
                suffixIcon: widget.optional && slot.url.text.isNotEmpty
                    ? IconButton(
                        tooltip: 'Clear',
                        icon: const Icon(Icons.close),
                        onPressed: () => setState(slot.clear),
                      )
                    : null,
              ),
              onChanged: (_) => setState(() {}),
            ),
            if (slot.isDownloaded)
              Padding(
                padding: const EdgeInsets.only(top: 6),
                child: Text(
                  'Downloaded · ${formatBytes(slot.original!.sizeBytes)}. '
                  'Changing this URL downloads the new file on the next '
                  'load.',
                  style: theme.textTheme.bodySmall,
                ),
              ),
          ],
          if (slot.importProgress case final progress?)
            Padding(
              padding: const EdgeInsets.only(top: 8),
              child: LinearProgressIndicator(value: progress),
            ),
          if (_importError case final error?)
            Padding(
              padding: const EdgeInsets.only(top: 8),
              child: Text(
                error,
                style: TextStyle(color: theme.colorScheme.error),
              ),
            ),
          Align(
            alignment: Alignment.centerLeft,
            child: TextButton.icon(
              onPressed: widget.enabled && slot.importProgress == null
                  ? _pickFile
                  : null,
              icon: const Icon(Icons.folder_open),
              label: Text(
                slot.imported == null
                    ? 'Choose a file on this device…'
                    : 'Choose a different file…',
              ),
            ),
          ),
        ],
      ),
    );
  }

  Future<void> _pickFile() async {
    const gguf = XTypeGroup(label: 'GGUF', extensions: <String>['gguf']);
    final file = await openFile(acceptedTypeGroups: const [gguf]);
    if (file == null) return;
    setState(() {
      _importError = null;
      widget.slot.importProgress = 0;
    });
    try {
      final ref = await widget.model.importArtifact(
        file,
        onProgress: (progress) {
          if (mounted) setState(() => widget.slot.importProgress = progress);
        },
      );
      if (!mounted) return;
      setState(() {
        widget.slot
          ..imported = ref
          ..url.clear()
          ..importProgress = null;
        if (ref.key case final key?) widget.slot.importedKeys.add(key);
      });
    } catch (error) {
      if (!mounted) return;
      setState(() {
        _importError = '$error';
        widget.slot.importProgress = null;
      });
    }
  }
}
