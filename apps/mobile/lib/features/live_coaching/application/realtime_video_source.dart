import '../domain/video_frame_metadata.dart';

abstract interface class RealtimeVideoSource {
  Stream<VideoFrameMetadata> get frames;

  Future<void> connect();

  Future<void> disconnect();
}
