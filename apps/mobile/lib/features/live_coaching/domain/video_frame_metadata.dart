enum VideoPresentationConnectionState {
  connecting,
  buffering,
  ready,
  reconnecting,
  failed,
}

enum VideoRotation { degrees0, degrees90, degrees180, degrees270 }

final class VideoContentRect {
  const VideoContentRect({
    required this.leftPx,
    required this.topPx,
    required this.widthPx,
    required this.heightPx,
  });

  final int leftPx;
  final int topPx;
  final int widthPx;
  final int heightPx;
}

final class VideoFrameMetadata {
  const VideoFrameMetadata({
    required this.streamId,
    required this.streamGeneration,
    required this.sessionTimebaseId,
    required this.videoPts,
    required this.frameSequence,
    required this.decodedWidthPx,
    required this.decodedHeightPx,
    required this.rotation,
    required this.contentRect,
    required this.connectionState,
  });

  final String streamId;
  final String streamGeneration;
  final String sessionTimebaseId;
  final BigInt videoPts;
  final BigInt frameSequence;
  final int decodedWidthPx;
  final int decodedHeightPx;
  final VideoRotation rotation;
  final VideoContentRect contentRect;
  final VideoPresentationConnectionState connectionState;
}
