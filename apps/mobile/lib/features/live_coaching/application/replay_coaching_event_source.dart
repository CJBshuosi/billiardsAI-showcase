import 'dart:async';
import 'dart:typed_data';

import '../../../core/contracts/contract_failure.dart';
import '../../../core/contracts/realtime_coaching_event_v2.dart';
import 'coaching_event_source.dart';

final class ReplayCoachingPayload {
  ReplayCoachingPayload._(this._bytes);

  factory ReplayCoachingPayload.capture(Uint8List bytes) =>
      ReplayCoachingPayload._(Uint8List.fromList(bytes));

  final Uint8List _bytes;

  Uint8List copyBytes() => Uint8List.fromList(_bytes);
}

final class ReplayCoachingEventSource implements CoachingEventSource {
  ReplayCoachingEventSource.fromPayloads(Iterable<Uint8List> payloads)
    : _payloads = Stream<ReplayCoachingPayload>.fromIterable(
        List<ReplayCoachingPayload>.unmodifiable(
          payloads.map(ReplayCoachingPayload.capture),
        ),
      );

  ReplayCoachingEventSource.fromStream(Stream<ReplayCoachingPayload> payloads)
    : _payloads = payloads;

  final Stream<ReplayCoachingPayload> _payloads;

  @override
  Stream<CoachingEventSourceState> get states => _decode();

  Stream<CoachingEventSourceState> _decode() async* {
    final acceptedContentByIdempotency = <String, String>{};
    BigInt? lastAcceptedSequence;
    final iterator = StreamIterator(_payloads);

    try {
      while (true) {
        final bool hasNext;
        try {
          hasNext = await iterator.moveNext();
        } on Object {
          yield const CoachingEventSourceFailed(
            CoachingEventSourceFailure(
              code: CoachingEventSourceFailureCode.sourceFailure,
              message: 'replay source failed before exhaustion',
            ),
          );
          return;
        }
        if (!hasNext) break;

        final RealtimeCoachingEventV2 event;
        try {
          event = decodeRealtimeCoachingEventV2(iterator.current.copyBytes());
        } on ContractFailure catch (failure) {
          yield CoachingEventSourceFailed(
            CoachingEventSourceFailure(
              code: CoachingEventSourceFailureCode.contractRejected,
              message: 'replay payload failed the realtime contract',
              contractFailure: failure,
            ),
          );
          return;
        }

        final priorContent = acceptedContentByIdempotency[event.idempotencyKey];
        if (priorContent != null) {
          if (priorContent == event.contentSha256) {
            continue;
          }
          yield const CoachingEventSourceFailed(
            CoachingEventSourceFailure(
              code: CoachingEventSourceFailureCode.conflictingIdempotencyKey,
              message:
                  'replay idempotency identity was reused with new content',
            ),
          );
          return;
        }

        if (lastAcceptedSequence != null) {
          final expected = lastAcceptedSequence + BigInt.one;
          if (event.sequence != expected) {
            final code = event.sequence <= lastAcceptedSequence
                ? CoachingEventSourceFailureCode.sequenceRegression
                : CoachingEventSourceFailureCode.sequenceGap;
            yield CoachingEventSourceFailed(
              CoachingEventSourceFailure(
                code: code,
                message: 'replay sequence is not strictly contiguous',
              ),
            );
            return;
          }
        }

        acceptedContentByIdempotency[event.idempotencyKey] =
            event.contentSha256;
        lastAcceptedSequence = event.sequence;
        yield VerifiedCoachingEvent(event);
      }

      yield const CoachingEventSourceExhausted();
    } finally {
      try {
        await iterator.cancel();
      } on Object {
        // The public stream has already emitted its terminal state, or its
        // downstream cancelled. Upstream cancellation detail stays private.
      }
    }
  }
}
