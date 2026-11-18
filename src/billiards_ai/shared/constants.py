"""两端共享的常量。感知层的两种实现必须引用同一份，否则切分结果不可复现（AD-13c）。"""

from datetime import timedelta

# 台面尺寸（毫米）。中式八球标准 9 尺台内沿。
# TODO: 以 Dilan 实测为准，安装标定时确认
TABLE_LENGTH_MM = 2540
TABLE_WIDTH_MM = 1270

# 球编号约定：0 = 母球，1-15 = 目标球
CUE_BALL = 0
BALL_IDS = range(0, 16)

# 静止判定：球心位移低于此值持续 STILL_DURATION_MS 视为静止。
# 这两个值定义了 shot 的边界，两种感知实现必须用同一组（AD-13c）。
STILL_THRESHOLD_MM = 3
STILL_DURATION_MS = 400

# 一杆的最长时长，超过视为切分失败
MAX_SHOT_DURATION_MS = 30_000

# 单球或连续帧检测缺口的最大容忍时长；更长缺口不得猜测 shot 边界。
MAX_DETECTION_GAP_MS = 500

# 置信度门限。低于 ABSTAIN 阈值触发弃权（AD-5），弃权是硬约束。
CONFIDENCE_HIGH = 0.85
CONFIDENCE_MEDIUM = 0.65
CONFIDENCE_ABSTAIN_BELOW = 0.50

# 会话空闲判定（AD-15）：台面无运动超过此时长，边缘上报 idle
SESSION_IDLE_TIMEOUT_MS = 20 * 60 * 1000

# 边缘时钟偏移超过此值拒绝截取（AD-15）
MAX_CLOCK_SKEW_MS = 2000

# 隐私门人工审计与素材声明允许的最大未来时钟偏差。
PRIVACY_MAX_FUTURE_CLOCK_SKEW = timedelta(minutes=5)

# 环形缓冲保留时长（AD-15）
RING_BUFFER_RETENTION_HOURS = 48

# 云端片段保留时长，交付且复核完成后起算（AD-17）
CLIP_RETENTION_DAYS = 30
RETENTION_CLOCK_SKEW_MINUTES = 5

# 复核租约（AD-16）
REVIEW_LEASE_MINUTES = 30
REVIEW_ESCALATE_HOURS = 24
