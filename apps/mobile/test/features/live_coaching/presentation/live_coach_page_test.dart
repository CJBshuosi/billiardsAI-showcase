import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:billiards_ai_mobile/core/contracts/contract_failure.dart';
import 'package:billiards_ai_mobile/core/contracts/realtime_coaching_event_v2.dart';
import 'package:billiards_ai_mobile/features/live_coaching/application/coaching_event_source.dart';
import 'package:billiards_ai_mobile/features/live_coaching/application/live_coach_controller.dart';
import 'package:billiards_ai_mobile/features/live_coaching/application/replay_coaching_event_source.dart';
import 'package:billiards_ai_mobile/features/live_coaching/presentation/live_coach_page.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  late RealtimeCoachingEventV2 baseEvent;
  late RealtimeCoachingEventV2 abstainedEvent;
  late RealtimeCoachingEventV2 unsupportedEvent;
  late RealtimeCoachingEventV2 degradedEvent;

  setUpAll(() {
    baseEvent = _decodedFixture('post_shot_success.json');
    abstainedEvent = _decodedFixture('abstained.json');
    unsupportedEvent = _decodedFixture('unsupported.json');
    degradedEvent = _decodedFixture('degraded.json');
  });

  testWidgets(
    'renders connecting, verified, abstained, unsupported and degraded states',
    (tester) async {
      final source = _WidgetEventSource();
      final controller = LiveCoachController(source: source);
      addTearDown(controller.dispose);
      addTearDown(source.close);
      await tester.pumpWidget(
        MaterialApp(home: LiveCoachPage(controller: controller)),
      );

      expect(find.byKey(const Key('live-coach-status-connecting')), findsOne);
      expect(find.text('正在等待已验证的教练事件'), findsOne);

      source.emit(VerifiedCoachingEvent(baseEvent));
      await tester.pump();
      expect(find.byKey(const Key('live-coach-status-verified')), findsOne);
      expect(find.text(baseEvent.textSummary), findsOne);

      source.emit(VerifiedCoachingEvent(abstainedEvent));
      await tester.pump();
      expect(find.byKey(const Key('live-coach-status-abstained')), findsOne);
      expect(find.textContaining('不计为失败'), findsOne);

      source.emit(VerifiedCoachingEvent(unsupportedEvent));
      await tester.pump();
      expect(find.byKey(const Key('live-coach-status-unsupported')), findsOne);
      expect(find.textContaining('不代表用户失败'), findsOne);

      source.emit(VerifiedCoachingEvent(degradedEvent));
      await tester.pump();
      expect(find.byKey(const Key('live-coach-status-degraded')), findsOne);
      expect(find.textContaining('路线已隐藏'), findsOne);
    },
  );

  testWidgets(
    'raw stream error clears verified widget content and terminates safely',
    (tester) async {
      final source = _WidgetEventSource();
      final controller = LiveCoachController(source: source);
      addTearDown(controller.dispose);
      addTearDown(source.close);
      await tester.pumpWidget(
        MaterialApp(home: LiveCoachPage(controller: controller)),
      );

      source.emit(VerifiedCoachingEvent(baseEvent));
      await tester.pump();
      expect(find.text(baseEvent.textSummary), findsOne);

      source.emitRawError(StateError('RAW_WIDGET_SECRET'));
      await tester.pump();

      expect(find.byKey(const Key('live-coach-status-stopped')), findsOne);
      expect(find.text(baseEvent.textSummary), findsNothing);
      expect(find.textContaining('RAW_WIDGET_SECRET'), findsNothing);
      expect(source.hasListener, isFalse);
    },
  );

  testWidgets(
    'canonical replay bytes drive verified UI then rejection clears it',
    (tester) async {
      final payloads = StreamController<ReplayCoachingPayload>();
      final source = ReplayCoachingEventSource.fromStream(payloads.stream);
      final controller = LiveCoachController(source: source);
      addTearDown(controller.dispose);
      addTearDown(payloads.close);
      await tester.pumpWidget(
        MaterialApp(home: LiveCoachPage(controller: controller)),
      );

      payloads.add(
        ReplayCoachingPayload.capture(
          File(
            'test/fixtures/realtime_v2/post_shot_success.json',
          ).readAsBytesSync(),
        ),
      );
      await tester.pump();
      expect(find.byKey(const Key('live-coach-status-verified')), findsOne);
      expect(find.text(baseEvent.textSummary), findsOne);

      final rejected = Uint8List.fromList(
        File('test/fixtures/replay_v2/sequence_10.json').readAsBytesSync(),
      )..[0] = 0x5b;
      payloads.add(ReplayCoachingPayload.capture(rejected));
      await tester.pumpAndSettle();

      expect(
        find.byKey(const Key('live-coach-status-contract-rejected')),
        findsOne,
      );
      expect(find.text(baseEvent.textSummary), findsNothing);
    },
  );

  testWidgets(
    'contract rejection removes prior text and never renders rejected details',
    (tester) async {
      final source = _WidgetEventSource();
      final controller = LiveCoachController(source: source);
      addTearDown(controller.dispose);
      addTearDown(source.close);
      await tester.pumpWidget(
        MaterialApp(home: LiveCoachPage(controller: controller)),
      );

      source.emit(VerifiedCoachingEvent(baseEvent));
      await tester.pump();
      expect(find.text(baseEvent.textSummary), findsOne);

      source.emit(
        const CoachingEventSourceFailed(
          CoachingEventSourceFailure(
            code: CoachingEventSourceFailureCode.contractRejected,
            message: 'REJECTED_PAYLOAD_TEXT',
            contractFailure: ContractFailure(
              ContractFailureCode.overlayGeometry,
              'REJECTED_OVERLAY_LABEL',
            ),
          ),
        ),
      );
      await tester.pump();

      expect(
        find.byKey(const Key('live-coach-status-contract-rejected')),
        findsOne,
      );
      expect(find.text(baseEvent.textSummary), findsNothing);
      expect(find.textContaining('REJECTED_PAYLOAD_TEXT'), findsNothing);
      expect(find.textContaining('REJECTED_OVERLAY_LABEL'), findsNothing);

      source.emit(VerifiedCoachingEvent(baseEvent));
      await tester.pump();
      expect(find.text(baseEvent.textSummary), findsNothing);
    },
  );

  testWidgets('special safe states never render contradictory event summary', (
    tester,
  ) async {
    final source = _WidgetEventSource();
    final controller = LiveCoachController(source: source);
    addTearDown(controller.dispose);
    addTearDown(source.close);
    await tester.pumpWidget(
      MaterialApp(home: LiveCoachPage(controller: controller)),
    );

    final cases = [
      (event: abstainedEvent, message: '本杆已弃权，不计为失败。'),
      (event: unsupportedEvent, message: '当前指导暂不支持，这不代表用户失败。'),
      (event: degradedEvent, message: '实时指导已降级，路线已隐藏。'),
    ];

    for (final testCase in cases) {
      source.emit(VerifiedCoachingEvent(testCase.event));
      await tester.pump();

      expect(find.text(testCase.message), findsOne);
      expect(find.text(testCase.event.textSummary), findsNothing);
    }
  });
}

final class _WidgetEventSource implements CoachingEventSource {
  final _controller = StreamController<CoachingEventSourceState>.broadcast(
    sync: true,
  );

  @override
  Stream<CoachingEventSourceState> get states => _controller.stream;

  void emit(CoachingEventSourceState state) => _controller.add(state);

  void emitRawError(Object error) => _controller.addError(error);

  bool get hasListener => _controller.hasListener;

  Future<void> close() => _controller.close();
}

RealtimeCoachingEventV2 _decodedFixture(String name) =>
    decodeRealtimeCoachingEventV2(
      File('test/fixtures/realtime_v2/$name').readAsBytesSync(),
    );
