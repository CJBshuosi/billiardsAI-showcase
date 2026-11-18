import '../../../core/contracts/realtime_coaching_event_v2.dart';
import 'video_frame_metadata.dart';

/// Immutable App-domain output shell for the later PresentationCoordinator.
///
/// This shell deliberately makes no calibration, timing, visibility, or
/// rendering decisions.
final class PresentationFrame {
  const PresentationFrame({
    required this.videoFrame,
    required this.coachingEvent,
  });

  final VideoFrameMetadata videoFrame;
  final RealtimeCoachingEventV2? coachingEvent;
}
