enum PrototypeMediaRetentionState { available, expired, deleted }

/// App-private A2 media index. It intentionally contains no permanent URL.
final class PrototypeShotMediaManifestV0 {
  const PrototypeShotMediaManifestV0({
    required this.shotId,
    required this.sessionId,
    required this.mediaAssetId,
    required this.recordingRevision,
    required this.clipStartMs,
    required this.clipEndMs,
    required this.sessionTimebaseId,
    required this.contentSha256,
    required this.retentionState,
  });

  final String shotId;
  final String sessionId;
  final String mediaAssetId;
  final BigInt recordingRevision;
  final BigInt clipStartMs;
  final BigInt clipEndMs;
  final String sessionTimebaseId;
  final String contentSha256;
  final PrototypeMediaRetentionState retentionState;
}

abstract interface class PlaybackRepository {
  Future<PrototypeShotMediaManifestV0> loadForShot({required String shotId});
}
