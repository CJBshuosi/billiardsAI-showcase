import 'package:flutter/material.dart';

import '../application/live_coach_controller.dart';

final class LiveCoachPage extends StatelessWidget {
  const LiveCoachPage({required this.controller, super.key});

  final LiveCoachController controller;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('实时 AI 教练')),
      body: SafeArea(
        child: AnimatedBuilder(
          animation: controller,
          builder: (context, _) => _LiveCoachStatus(state: controller.state),
        ),
      ),
    );
  }
}

final class _LiveCoachStatus extends StatelessWidget {
  const _LiveCoachStatus({required this.state});

  final LiveCoachUiState state;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Semantics(
        container: true,
        liveRegion: true,
        label: state.message,
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            key: _statusKey(state.mode),
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                _title(state.mode),
                style: Theme.of(context).textTheme.headlineSmall,
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 12),
              Text(state.message, textAlign: TextAlign.center),
            ],
          ),
        ),
      ),
    );
  }
}

Key _statusKey(LiveCoachUiMode mode) => Key(switch (mode) {
  LiveCoachUiMode.connecting => 'live-coach-status-connecting',
  LiveCoachUiMode.verifiedGuidance => 'live-coach-status-verified',
  LiveCoachUiMode.abstained => 'live-coach-status-abstained',
  LiveCoachUiMode.unsupported => 'live-coach-status-unsupported',
  LiveCoachUiMode.degraded => 'live-coach-status-degraded',
  LiveCoachUiMode.contractRejected => 'live-coach-status-contract-rejected',
  LiveCoachUiMode.stopped => 'live-coach-status-stopped',
});

String _title(LiveCoachUiMode mode) => switch (mode) {
  LiveCoachUiMode.connecting => '连接中',
  LiveCoachUiMode.verifiedGuidance => '已验证指导',
  LiveCoachUiMode.abstained => '本杆弃权',
  LiveCoachUiMode.unsupported => '当前不支持',
  LiveCoachUiMode.degraded => '安全降级',
  LiveCoachUiMode.contractRejected => '数据已拒绝',
  LiveCoachUiMode.stopped => '指导已停止',
};
