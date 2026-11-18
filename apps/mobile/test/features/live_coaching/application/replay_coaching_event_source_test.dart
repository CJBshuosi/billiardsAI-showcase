import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:billiards_ai_mobile/core/contracts/contract_failure.dart';
import 'package:billiards_ai_mobile/features/live_coaching/application/coaching_event_source.dart';
import 'package:billiards_ai_mobile/features/live_coaching/application/replay_coaching_event_source.dart';
import 'package:flutter_test/flutter_test.dart';

Uint8List _fixture(String name) =>
    File('test/fixtures/$name').readAsBytesSync();

void main() {
  final sequence9 = _fixture('realtime_v2/post_shot_success.json');
  final sequence10 = _fixture('replay_v2/sequence_10.json');
  final sequence11 = _fixture('replay_v2/sequence_11.json');
  final conflictingSequence9 = _fixture(
    'replay_v2/conflicting_sequence_9.json',
  );

  test(
    'a valid finite replay emits one verified event then exhausted',
    () async {
      final states = await ReplayCoachingEventSource.fromPayloads([
        sequence9,
      ]).states.toList();

      expect(states, hasLength(2));
      expect(states.first, isA<VerifiedCoachingEvent>());
      expect(
        (states.first as VerifiedCoachingEvent).event.sequence,
        BigInt.from(9),
      );
      expect(states.last, isA<CoachingEventSourceExhausted>());
    },
  );

  test('identical idempotency replay is suppressed', () async {
    final states = await ReplayCoachingEventSource.fromPayloads([
      sequence9,
      sequence9,
    ]).states.toList();

    expect(states.whereType<VerifiedCoachingEvent>(), hasLength(1));
    expect(states.last, isA<CoachingEventSourceExhausted>());
  });

  test('distinct contiguous events are emitted in authority order', () async {
    final states = await ReplayCoachingEventSource.fromPayloads([
      sequence9,
      sequence10,
      sequence11,
    ]).states.toList();

    expect(
      states.whereType<VerifiedCoachingEvent>().map(
        (state) => state.event.sequence,
      ),
      [BigInt.from(9), BigInt.from(10), BigInt.from(11)],
    );
    expect(states.last, isA<CoachingEventSourceExhausted>());
  });

  test(
    'a sequence gap emits failure and never exposes later payloads',
    () async {
      final states = await ReplayCoachingEventSource.fromPayloads([
        sequence9,
        sequence11,
        sequence10,
      ]).states.toList();

      expect(states.whereType<VerifiedCoachingEvent>(), hasLength(1));
      expect(states, hasLength(2));
      final failure = (states.last as CoachingEventSourceFailed).failure;
      expect(failure.code, CoachingEventSourceFailureCode.sequenceGap);
    },
  );

  test('a sequence regression fails closed', () async {
    final states = await ReplayCoachingEventSource.fromPayloads([
      sequence10,
      sequence9,
    ]).states.toList();

    expect(states.whereType<VerifiedCoachingEvent>(), hasLength(1));
    expect(states, hasLength(2));
    final failure = (states.last as CoachingEventSourceFailed).failure;
    expect(failure.code, CoachingEventSourceFailureCode.sequenceRegression);
  });

  test('conflicting reuse of an idempotency key fails closed', () async {
    final states = await ReplayCoachingEventSource.fromPayloads([
      sequence9,
      conflictingSequence9,
      sequence10,
    ]).states.toList();

    expect(states.whereType<VerifiedCoachingEvent>(), hasLength(1));
    expect(states, hasLength(2));
    final failure = (states.last as CoachingEventSourceFailed).failure;
    expect(
      failure.code,
      CoachingEventSourceFailureCode.conflictingIdempotencyKey,
    );
  });

  test('contract rejection emits its typed failure and stops', () async {
    final malformed = Uint8List.fromList(sequence9)..[0] = 0x5b;
    final states = await ReplayCoachingEventSource.fromPayloads([
      malformed,
      sequence9,
    ]).states.toList();

    expect(states, hasLength(1));
    final failure = (states.single as CoachingEventSourceFailed).failure;
    expect(failure.code, CoachingEventSourceFailureCode.contractRejected);
    expect(failure.contractFailure, isA<ContractFailure>());
  });

  test('the replay copies retained bytes before caller mutation', () async {
    final mutable = Uint8List.fromList(sequence9);
    final source = ReplayCoachingEventSource.fromPayloads([mutable]);
    mutable.fillRange(0, mutable.length, 0);

    final states = await source.states.toList();

    expect(states.first, isA<VerifiedCoachingEvent>());
    expect(states.last, isA<CoachingEventSourceExhausted>());
  });

  test(
    'captured payload survives same-turn mutation before async delivery',
    () async {
      final mutable = Uint8List.fromList(sequence9);
      final controller = StreamController<ReplayCoachingPayload>();
      final source = ReplayCoachingEventSource.fromStream(controller.stream);
      final captured = ReplayCoachingPayload.capture(mutable);

      final statesFuture = source.states.toList();
      controller.add(captured);
      mutable.fillRange(0, mutable.length, 0);
      await controller.close();
      final states = await statesFuture;

      expect(states.first, isA<VerifiedCoachingEvent>());
      expect(states.last, isA<CoachingEventSourceExhausted>());
    },
  );

  test(
    'upstream stream error becomes one safe typed terminal failure',
    () async {
      final controller = StreamController<ReplayCoachingPayload>();
      final source = ReplayCoachingEventSource.fromStream(controller.stream);
      final statesFuture = source.states.toList();

      controller.addError(StateError('raw upstream detail must not escape'));
      await controller.close();
      final states = await statesFuture;

      expect(states, hasLength(1));
      final failure = (states.single as CoachingEventSourceFailed).failure;
      expect(failure.code, CoachingEventSourceFailureCode.sourceFailure);
      expect(failure.message, 'replay source failed before exhaustion');
      expect(failure.contractFailure, isNull);
    },
  );

  test('exhaustion releases the upstream listener', () async {
    final input = _TrackedReplayInput();
    final statesFuture = input.source.states.toList();

    input.add(sequence9);
    await input.close();
    final states = await statesFuture;

    expect(states.last, isA<CoachingEventSourceExhausted>());
    expect(input.hasListener, isFalse);
    expect(input.cancelObserved, isTrue);
  });

  test('contract rejection cancels an upstream that stays open', () async {
    final input = _TrackedReplayInput();
    final statesFuture = input.source.states.toList();
    final malformed = Uint8List.fromList(sequence9)..[0] = 0x5b;

    input.add(malformed);
    final states = await statesFuture;

    expect(states.single, isA<CoachingEventSourceFailed>());
    expect(input.hasListener, isFalse);
    expect(input.cancelObserved, isTrue);
    await input.close();
  });

  test('idempotency conflict cancels an upstream that stays open', () async {
    final input = _TrackedReplayInput();
    final statesFuture = input.source.states.toList();

    input.add(sequence9);
    input.add(conflictingSequence9);
    final states = await statesFuture;

    expect(
      (states.last as CoachingEventSourceFailed).failure.code,
      CoachingEventSourceFailureCode.conflictingIdempotencyKey,
    );
    expect(input.hasListener, isFalse);
    expect(input.cancelObserved, isTrue);
    await input.close();
  });

  test('sequence gap cancels an upstream that stays open', () async {
    final input = _TrackedReplayInput();
    final statesFuture = input.source.states.toList();

    input.add(sequence9);
    input.add(sequence11);
    final states = await statesFuture;

    expect(
      (states.last as CoachingEventSourceFailed).failure.code,
      CoachingEventSourceFailureCode.sequenceGap,
    );
    expect(input.hasListener, isFalse);
    expect(input.cancelObserved, isTrue);
    await input.close();
  });

  test('sequence regression cancels an upstream that stays open', () async {
    final input = _TrackedReplayInput();
    final statesFuture = input.source.states.toList();

    input.add(sequence10);
    input.add(sequence9);
    final states = await statesFuture;

    expect(
      (states.last as CoachingEventSourceFailed).failure.code,
      CoachingEventSourceFailureCode.sequenceRegression,
    );
    expect(input.hasListener, isFalse);
    expect(input.cancelObserved, isTrue);
    await input.close();
  });

  test('upstream error releases the listener and hides raw detail', () async {
    final input = _TrackedReplayInput();
    final statesFuture = input.source.states.toList();

    input.addError(StateError('RAW_REPLAY_SECRET'));
    final states = await statesFuture;

    final failure = (states.single as CoachingEventSourceFailed).failure;
    expect(failure.code, CoachingEventSourceFailureCode.sourceFailure);
    expect(failure.message, isNot(contains('RAW_REPLAY_SECRET')));
    expect(input.hasListener, isFalse);
    expect(input.cancelObserved, isTrue);
    await input.close();
  });

  test(
    'downstream cancellation releases an upstream that stays open',
    () async {
      final input = _TrackedReplayInput();
      final firstVerified = Completer<void>();
      late final StreamSubscription<CoachingEventSourceState> subscription;
      subscription = input.source.states.listen((state) {
        if (state is VerifiedCoachingEvent && !firstVerified.isCompleted) {
          firstVerified.complete();
        }
      });

      input.add(sequence9);
      await firstVerified.future;
      await subscription.cancel();

      expect(input.hasListener, isFalse);
      expect(input.cancelObserved, isTrue);
      await input.close();
    },
  );
}

final class _TrackedReplayInput {
  _TrackedReplayInput() {
    _controller = StreamController<ReplayCoachingPayload>(
      onCancel: () {
        cancelObserved = true;
      },
    );
    source = ReplayCoachingEventSource.fromStream(_controller.stream);
  }

  late final StreamController<ReplayCoachingPayload> _controller;
  late final ReplayCoachingEventSource source;
  bool cancelObserved = false;

  bool get hasListener => _controller.hasListener;

  void add(Uint8List bytes) =>
      _controller.add(ReplayCoachingPayload.capture(bytes));

  void addError(Object error) => _controller.addError(error);

  Future<void> close() => _controller.close();
}
