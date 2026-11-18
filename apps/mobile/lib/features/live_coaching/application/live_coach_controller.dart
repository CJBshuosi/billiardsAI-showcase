import 'dart:async';

import 'package:flutter/foundation.dart';

import '../../../core/contracts/realtime_coaching_event_v2.dart';
import 'coaching_event_source.dart';

enum LiveCoachUiMode {
  connecting,
  verifiedGuidance,
  abstained,
  unsupported,
  degraded,
  contractRejected,
  stopped,
}

@immutable
final class LiveCoachUiState {
  const LiveCoachUiState({
    required this.mode,
    required this.message,
    required this.trustedEventId,
    required this.outcomeStatus,
    required this.deliveryStatus,
    required this.overlayVisible,
  });

  const LiveCoachUiState.connecting()
    : mode = LiveCoachUiMode.connecting,
      message = '正在等待已验证的教练事件',
      trustedEventId = null,
      outcomeStatus = null,
      deliveryStatus = null,
      overlayVisible = false;

  final LiveCoachUiMode mode;
  final String message;
  final String? trustedEventId;
  final CoachingOutcomeStatus? outcomeStatus;
  final CoachingDeliveryStatus? deliveryStatus;
  final bool overlayVisible;
}

final class LiveCoachController extends ChangeNotifier {
  LiveCoachController({required CoachingEventSource source})
    : _state = const LiveCoachUiState.connecting() {
    final subscription = source.states.listen(
      _acceptSourceState,
      onError: _acceptSourceError,
      onDone: _acceptSourceDone,
    );
    _subscription = subscription;
    if (_terminal || _disposed) unawaited(subscription.cancel());
  }

  LiveCoachUiState _state;
  StreamSubscription<CoachingEventSourceState>? _subscription;
  bool _terminal = false;
  bool _disposed = false;

  LiveCoachUiState get state => _state;

  void _acceptSourceState(CoachingEventSourceState sourceState) {
    if (_disposed || _terminal) return;

    switch (sourceState) {
      case VerifiedCoachingEvent(:final event):
        _publish(_fromVerified(event));
      case CoachingEventSourceExhausted():
        _stop(
          const LiveCoachUiState(
            mode: LiveCoachUiMode.stopped,
            message: '教练事件已结束；视频可继续，但不会产生新指导。',
            trustedEventId: null,
            outcomeStatus: null,
            deliveryStatus: null,
            overlayVisible: false,
          ),
        );
      case CoachingEventSourceFailed(:final failure):
        final contractRejected =
            failure.code == CoachingEventSourceFailureCode.contractRejected;
        _stop(
          LiveCoachUiState(
            mode: contractRejected
                ? LiveCoachUiMode.contractRejected
                : LiveCoachUiMode.stopped,
            message: contractRejected
                ? '指导数据未通过安全校验，已停止展示。'
                : '教练事件流不连续或已中断，已停止新指导。',
            trustedEventId: null,
            outcomeStatus: null,
            deliveryStatus: null,
            overlayVisible: false,
          ),
        );
    }
  }

  void _acceptSourceDone() {
    if (_disposed || _terminal) return;
    _stop(
      const LiveCoachUiState(
        mode: LiveCoachUiMode.stopped,
        message: '教练事件连接已结束；视频可继续，但不会产生新指导。',
        trustedEventId: null,
        outcomeStatus: null,
        deliveryStatus: null,
        overlayVisible: false,
      ),
    );
  }

  void _acceptSourceError(Object _, StackTrace stackTrace) {
    if (_disposed || _terminal) return;
    _stop(
      const LiveCoachUiState(
        mode: LiveCoachUiMode.stopped,
        message: '教练事件连接异常，已停止新指导。',
        trustedEventId: null,
        outcomeStatus: null,
        deliveryStatus: null,
        overlayVisible: false,
      ),
    );
  }

  LiveCoachUiState _fromVerified(RealtimeCoachingEventV2 event) {
    if (event.deliveryStatus != CoachingDeliveryStatus.normal) {
      final outcomePrefix = switch (event.outcomeStatus) {
        CoachingOutcomeStatus.abstained => '本杆已弃权，不计为失败；',
        CoachingOutcomeStatus.unsupported => '当前指导暂不支持，不代表用户失败；',
        _ => '',
      };
      return _verifiedState(
        event,
        mode: LiveCoachUiMode.degraded,
        message: '$outcomePrefix实时指导已降级，路线已隐藏。',
        overlayVisible: false,
      );
    }

    return switch (event.outcomeStatus) {
      CoachingOutcomeStatus.abstained => _verifiedState(
        event,
        mode: LiveCoachUiMode.abstained,
        message: '本杆已弃权，不计为失败。',
        overlayVisible: false,
      ),
      CoachingOutcomeStatus.unsupported => _verifiedState(
        event,
        mode: LiveCoachUiMode.unsupported,
        message: '当前指导暂不支持，这不代表用户失败。',
        overlayVisible: false,
      ),
      _ => _verifiedState(
        event,
        mode: LiveCoachUiMode.verifiedGuidance,
        message: event.textSummary,
        overlayVisible: event.overlay != null,
      ),
    };
  }

  LiveCoachUiState _verifiedState(
    RealtimeCoachingEventV2 event, {
    required LiveCoachUiMode mode,
    required String message,
    required bool overlayVisible,
  }) => LiveCoachUiState(
    mode: mode,
    message: message,
    trustedEventId: event.eventId,
    outcomeStatus: event.outcomeStatus,
    deliveryStatus: event.deliveryStatus,
    overlayVisible: overlayVisible,
  );

  void _publish(LiveCoachUiState nextState) {
    if (_disposed || _terminal) return;
    _state = nextState;
    notifyListeners();
  }

  void _stop(LiveCoachUiState terminalState) {
    if (_disposed || _terminal) return;
    _terminal = true;
    _state = terminalState;
    notifyListeners();
    final subscription = _subscription;
    if (subscription != null) unawaited(subscription.cancel());
  }

  @override
  void dispose() {
    if (_disposed) return;
    _disposed = true;
    final subscription = _subscription;
    if (subscription != null) unawaited(subscription.cancel());
    super.dispose();
  }
}
