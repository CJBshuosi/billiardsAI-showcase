import '../../../core/contracts/contract_failure.dart';
import '../../../core/contracts/realtime_coaching_event_v2.dart';

enum CoachingEventSourceFailureCode {
  contractRejected,
  sourceFailure,
  sequenceGap,
  sequenceRegression,
  conflictingIdempotencyKey,
}

final class CoachingEventSourceFailure {
  const CoachingEventSourceFailure({
    required this.code,
    required this.message,
    this.contractFailure,
  });

  final CoachingEventSourceFailureCode code;
  final String message;
  final ContractFailure? contractFailure;
}

sealed class CoachingEventSourceState {
  const CoachingEventSourceState();
}

final class VerifiedCoachingEvent extends CoachingEventSourceState {
  const VerifiedCoachingEvent(this.event);

  final RealtimeCoachingEventV2 event;
}

final class CoachingEventSourceExhausted extends CoachingEventSourceState {
  const CoachingEventSourceExhausted();
}

final class CoachingEventSourceFailed extends CoachingEventSourceState {
  const CoachingEventSourceFailed(this.failure);

  final CoachingEventSourceFailure failure;
}

abstract interface class CoachingEventSource {
  Stream<CoachingEventSourceState> get states;
}
