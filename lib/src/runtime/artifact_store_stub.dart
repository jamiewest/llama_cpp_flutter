import 'package:extensions/logging.dart';

import 'artifact_store_api.dart';

/// Creates managed artifact storage for the current platform.
///
/// On iOS and macOS this is a directory-backed store; [directory] is
/// required there and is where every managed artifact lands. On the web it
/// is an OPFS-backed store and [directory] is ignored. [authToken] is sent
/// as a `Bearer` token on artifact downloads, for gated Hugging Face
/// repositories. This stub is what the analyzer and platforms without local
/// inference resolve.
ArtifactStore createArtifactStore({
  String? directory,
  String? authToken,
  LoggerFactory? loggerFactory,
}) => throw UnsupportedError(
  'Managed artifact storage is not supported on this platform.',
);
