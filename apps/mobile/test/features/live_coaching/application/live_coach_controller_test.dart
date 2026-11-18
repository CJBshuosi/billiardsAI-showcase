import 'dart:async';
import 'dart:io';

import 'package:billiards_ai_mobile/core/contracts/contract_failure.dart';
import 'package:billiards_ai_mobile/core/contracts/realtime_coaching_event_v2.dart';
import 'package:billiards_ai_mobile/features/live_coaching/application/coaching_event_source.dart';
import 'package:billiards_ai_mobile/features/live_coaching/application/live_coach_controller.dart';
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

  test('starts connecting without trusted guidance or overlay', () {
    final source = _ControlledCoachingEventSource();
    final controller = LiveCoachController(source: source);
    addTearDown(controller.dispose);
    addTearDown(source.close);

    expect(controller.state.mode, LiveCoachUiMode.connecting);
    expect(controller.state.trustedEventId, isNull);
    expect(controller.state.outcomeStatus, isNull);
    expect(controller.state.deliveryStatus, isNull);
    expect(controller.state.overlayVisible, isFalse);
  });

  test('normal verified guidance exposes only trusted UI values', () {
    final source = _ControlledCoachingEventSource();
    final controller = LiveCoachController(source: source);
    addTearDown(controller.dispose);
    addTearDown(source.close);

    source.emit(VerifiedCoachingEvent(baseEvent));

    expect(controller.state.mode, LiveCoachUiMode.verifiedGuidance);
    expect(controller.state.message, baseEvent.textSummary);
    expect(controller.state.trustedEventId, baseEvent.eventId);
    expect(controller.state.outcomeStatus, CoachingOutcomeStatus.success);
    expect(controller.state.deliveryStatus, CoachingDeliveryStatus.normal);
    expect(controller.state.overlayVisible, isTrue);
  });

  test('abstention and unsupported are not presented as user failure', () {
    final source = _ControlledCoachingEventSource();
    final controller = LiveCoachController(source: source);
    addTearDown(controller.dispose);
    addTearDown(source.close);

    source.emit(VerifiedCoachingEvent(abstainedEvent));
    expect(controller.state.mode, LiveCoachUiMode.abstained);
    expect(controller.state.message, contains('不计为失败'));
    expect(controller.state.overlayVisible, isFalse);

    source.emit(VerifiedCoachingEvent(unsupportedEvent));
    expect(controller.state.mode, LiveCoachUiMode.unsupported);
    expect(controller.state.message, contains('不代表用户失败'));
    expect(controller.state.overlayVisible, isFalse);
  });

  test('degraded delivery suppresses overlay', () {
    final source = _ControlledCoachingEventSource();
    final controller = LiveCoachController(source: source);
    addTearDown(controller.dispose);
    addTearDown(source.close);

    source.emit(VerifiedCoachingEvent(degradedEvent));

    expect(controller.state.mode, LiveCoachUiMode.degraded);
    expect(controller.state.message, contains('路线已隐藏'));
    expect(controller.state.overlayVisible, isFalse);
  });

  test('safe special states never expose contradictory upstream summary', () {
    final source = _ControlledCoachingEventSource();
    final controller = LiveCoachController(source: source);
    addTearDown(controller.dispose);
    addTearDown(source.close);

    final cases = [
      (event: abstainedEvent, mode: LiveCoachUiMode.abstained),
      (event: unsupportedEvent, mode: LiveCoachUiMode.unsupported),
      (event: degradedEvent, mode: LiveCoachUiMode.degraded),
    ];

    for (final testCase in cases) {
      source.emit(VerifiedCoachingEvent(testCase.event));

      expect(controller.state.mode, testCase.mode);
      expect(controller.state.message, isNot(testCase.event.textSummary));
      expect(controller.state.overlayVisible, isFalse);
    }
  });

  test(
    'raw stream error clears trusted guidance and terminates without detail',
    () async {
      final source = _ControlledCoachingEventSource();
      final controller = LiveCoachController(source: source);
      addTearDown(controller.dispose);
      addTearDown(source.close);

      source.emit(VerifiedCoachingEvent(baseEvent));
      source.emitRawError(StateError('RAW_STREAM_SECRET'));
      await Future<void>.delayed(Duration.zero);

      expect(controller.state.mode, LiveCoachUiMode.stopped);
      expect(controller.state.trustedEventId, isNull);
      expect(controller.state.outcomeStatus, isNull);
      expect(controller.state.deliveryStatus, isNull);
      expect(controller.state.overlayVisible, isFalse);
      expect(controller.state.message, isNot(contains('RAW_STREAM_SECRET')));
      expect(source.hasListener, isFalse);
    },
  );

  test('contract rejection clears prior trusted values and terminates', () {
    final source = _ControlledCoachingEventSource();
    final controller = LiveCoachController(source: source);
    addTearDown(controller.dispose);
    addTearDown(source.close);

    source.emit(VerifiedCoachingEvent(baseEvent));
    source.emit(
      const CoachingEventSourceFailed(
        CoachingEventSourceFailure(
          code: CoachingEventSourceFailureCode.contractRejected,
          message: 'raw rejected text must not escape',
          contractFailure: ContractFailure(
            ContractFailureCode.hashMismatch,
            'rejected payload overlay label must not escape',
          ),
        ),
      ),
    );

    expect(controller.state.mode, LiveCoachUiMode.contractRejected);
    expect(controller.state.trustedEventId, isNull);
    expect(controller.state.outcomeStatus, isNull);
    expect(controller.state.deliveryStatus, isNull);
    expect(controller.state.overlayVisible, isFalse);
    expect(controller.state.message, isNot(contains('raw rejected text')));

    source.emit(VerifiedCoachingEvent(baseEvent));
    expect(controller.state.mode, LiveCoachUiMode.contractRejected);
    expect(controller.state.trustedEventId, isNull);
  });

  test('gap and exhaustion stop guidance without fabricating failure', () {
    final source = _ControlledCoachingEventSource();
    final controller = LiveCoachController(source: source);
    addTearDown(controller.dispose);
    addTearDown(source.close);

    source.emit(VerifiedCoachingEvent(baseEvent));
    source.emit(
      const CoachingEventSourceFailed(
        CoachingEventSourceFailure(
          code: CoachingEventSourceFailureCode.sequenceGap,
          message: 'gap',
        ),
      ),
    );

    expect(controller.state.mode, LiveCoachUiMode.stopped);
    expect(controller.state.message, contains('事件流'));
    expect(controller.state.message, isNot(contains('训练失败')));
    expect(controller.state.overlayVisible, isFalse);

    final exhaustedSource = _ControlledCoachingEventSource();
    final exhaustedController = LiveCoachController(source: exhaustedSource);
    addTearDown(exhaustedController.dispose);
    addTearDown(exhaustedSource.close);
    exhaustedSource.emit(const CoachingEventSourceExhausted());

    expect(exhaustedController.state.mode, LiveCoachUiMode.stopped);
    expect(exhaustedController.state.message, contains('事件已结束'));
    expect(exhaustedController.state.message, isNot(contains('训练失败')));
  });

  test('an event stream disconnect clears guidance and stops safely', () async {
    final source = _ControlledCoachingEventSource();
    final controller = LiveCoachController(source: source);
    addTearDown(controller.dispose);

    source.emit(VerifiedCoachingEvent(baseEvent));
    await source.close();

    expect(controller.state.mode, LiveCoachUiMode.stopped);
    expect(controller.state.message, contains('连接已结束'));
    expect(controller.state.trustedEventId, isNull);
    expect(controller.state.overlayVisible, isFalse);
  });

  test('dispose cancels listeners and ignores later states safely', () {
    final source = _ControlledCoachingEventSource();
    final controller = LiveCoachController(source: source);
    var notifications = 0;
    controller.addListener(() => notifications += 1);

    controller.dispose();
    source.emit(VerifiedCoachingEvent(baseEvent));

    expect(notifications, 0);
    expect(source.hasListener, isFalse);
    source.close();
  });

  test(
    'synchronous terminal listen cancels returned subscription immediately',
    () {
      final source = _SynchronousTerminalEventSource();

      final controller = LiveCoachController(source: source);
      addTearDown(controller.dispose);

      expect(controller.state.mode, LiveCoachUiMode.contractRejected);
      expect(source.subscription.cancelCount, 1);
    },
  );
}

final class _ControlledCoachingEventSource implements CoachingEventSource {
  final _controller = StreamController<CoachingEventSourceState>.broadcast(
    sync: true,
  );

  @override
  Stream<CoachingEventSourceState> get states => _controller.stream;

  bool get hasListener => _controller.hasListener;

  void emit(CoachingEventSourceState state) => _controller.add(state);

  void emitRawError(Object error) => _controller.addError(error);

  Future<void> close() => _controller.close();
}

final class _SynchronousTerminalEventSource implements CoachingEventSource {
  final subscription = _TrackableSubscription<CoachingEventSourceState>();

  @override
  late final Stream<CoachingEventSourceState> states =
      _SynchronousTerminalStream(subscription);
}

final class _SynchronousTerminalStream
    extends Stream<CoachingEventSourceState> {
  _SynchronousTerminalStream(this.subscription);

  final _TrackableSubscription<CoachingEventSourceState> subscription;

  @override
  StreamSubscription<CoachingEventSourceState> listen(
    void Function(CoachingEventSourceState event)? onData, {
    Function? onError,
    void Function()? onDone,
    bool? cancelOnError,
  }) {
    onData?.call(
      const CoachingEventSourceFailed(
        CoachingEventSourceFailure(
          code: CoachingEventSourceFailureCode.contractRejected,
          message: 'terminal during listen',
        ),
      ),
    );
    onData?.call(const CoachingEventSourceExhausted());
    onDone?.call();
    return subscription;
  }
}

final class _TrackableSubscription<T> implements StreamSubscription<T> {
  int cancelCount = 0;
  bool _isPaused = false;

  @override
  Future<void> cancel() async {
    cancelCount += 1;
  }

  @override
  void onData(void Function(T data)? handleData) {}

  @override
  void onError(Function? handleError) {}

  @override
  void onDone(void Function()? handleDone) {}

  @override
  void pause([Future<void>? resumeSignal]) {
    _isPaused = true;
  }

  @override
  void resume() {
    _isPaused = false;
  }

  @override
  bool get isPaused => _isPaused;

  @override
  Future<E> asFuture<E>([E? futureValue]) => Completer<E>().future;
}

RealtimeCoachingEventV2 _decodedFixture(String name) =>
    decodeRealtimeCoachingEventV2(
      File('test/fixtures/realtime_v2/$name').readAsBytesSync(),
    );
