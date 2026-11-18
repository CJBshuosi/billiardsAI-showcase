import 'dart:io';

import 'package:billiards_ai_mobile/features/live_coaching/application/coaching_event_source.dart';
import 'package:billiards_ai_mobile/features/live_coaching/application/playback_repository.dart';
import 'package:billiards_ai_mobile/features/live_coaching/application/realtime_video_source.dart';
import 'package:billiards_ai_mobile/features/live_coaching/domain/presentation_frame.dart';
import 'package:billiards_ai_mobile/features/live_coaching/domain/video_frame_metadata.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('App port types expose only App-owned domain values', () {
    expect(RealtimeVideoSource, isNotNull);
    expect(CoachingEventSource, isNotNull);
    expect(PlaybackRepository, isNotNull);
    expect(PresentationFrame, isNotNull);
    expect(VideoFrameMetadata, isNotNull);
  });

  test('App port files do not import transport or platform libraries', () {
    const files = [
      'lib/features/live_coaching/application/realtime_video_source.dart',
      'lib/features/live_coaching/application/coaching_event_source.dart',
      'lib/features/live_coaching/application/playback_repository.dart',
      'lib/features/live_coaching/domain/video_frame_metadata.dart',
      'lib/features/live_coaching/domain/presentation_frame.dart',
    ];
    final forbidden = RegExp(
      r'''(?:package:(?:flutter_webrtc|video_player)|dart:(?:io|ffi)|MethodChannel|EventChannel)''',
    );

    for (final path in files) {
      expect(
        File(path).readAsStringSync(),
        isNot(matches(forbidden)),
        reason: '$path must remain transport and platform neutral',
      );
    }
  });
}
