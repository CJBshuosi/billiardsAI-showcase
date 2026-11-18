"""Shot — 感知层与上层之间的契约（AD-13）。

感知层唯一的职责是填充这个模型。两种实现（async 的 VLM、realtime 的 YOLO）
必须产出语义等价的 Shot，上层才能在两者之间互换而不改代码。

字段分两组，边界是硬约束：

  上层可消费   struck_ball / potted_balls / cue_ball_end / cushion_count /
               travel_distance_mm / is_foul / deviation_mm / confidence
  上层禁读     trajectory（仅可视化）/ raw_observation（仅人工复核与调试）

新增字段前必须回答：另一种实现能否填充它？填出的语义是否等价？
任一为否，该字段属于 raw_observation，不进模型主体。
"""

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, Field

from billiards_ai.shared.models.calibration import CalibrationLineage
from billiards_ai.shared.models.geometry import Point, Zone
from billiards_ai.shared.models.quality import ShotAnalysisStatus, ShotFieldQuality


class Confidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    ABSTAIN = "abstain"
    """弃权。子集外或置信不足，系统不得给出走位反馈，且必须对用户可见（AD-5）。"""


class TargetZoneSource(StrEnum):
    """走位目标区的来源。

    两个变体的来源根本不同：realtime 由训练场景预设（因果可得），
    async 只能从后续球局倒推。上层禁止跨来源比较或合并统计——
    倒推出的目标区和预设的不是同一种量（AD-13d）。
    """

    SCENE_PRESET = "scene_preset"
    POST_HOC_INFERRED = "post_hoc_inferred"
    NONE = "none"


class ReviewStatus(StrEnum):
    PENDING = "pending_review"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    CORRECTED = "corrected"
    REJECTED = "rejected"


class TrajectoryUnavailableReason(StrEnum):
    """Why a calibrated trajectory attempt did not produce usable geometry."""

    NOT_OBSERVED = "NOT_OBSERVED"
    INSUFFICIENT_POINTS = "INSUFFICIENT_POINTS"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"


class Shot(BaseModel):
    id: UUID
    session_id: UUID
    sequence: int = Field(ge=1, description="会话内第几杆，从 1 起")

    t_start_ms: int = Field(ge=0, description="相对会话起点")
    t_end_ms: int = Field(ge=0)

    ball_positions_before: dict[int, Point] | None
    ball_positions_after: dict[int, Point] | None
    cue_ball_end: Point | None

    struck_ball: int | None = Field(default=None, description="击中的第一个目标球")
    potted_balls: list[int] | None = Field(default_factory=list)
    cushion_count: int | None = Field(default=0, ge=0)
    travel_distance_mm: int | None = Field(default=0, ge=0)
    is_foul: bool | None = False

    target_zone: Zone | None = None
    target_zone_source: TargetZoneSource = TargetZoneSource.NONE
    deviation_mm: int | None = None

    confidence: Confidence
    field_quality: ShotFieldQuality | None = Field(
        default=None,
        description="最终业务字段的来源与置信度；由 finalizer 完整填充。",
    )
    analysis_status: ShotAnalysisStatus = ShotAnalysisStatus.PENDING

    trajectory: list[Point] | None = Field(
        default=None,
        description="上层禁读。采样密度因实现而异，仅供可视化与复核。",
    )
    trajectory_calibration: CalibrationLineage | None = Field(
        default=None,
        description="尝试观测 trajectory 时使用的不可变标定 lineage，仅供复核与可视化。",
    )
    trajectory_unavailable_reason: TrajectoryUnavailableReason | None = Field(
        default=None,
        description="未形成可用 trajectory 时的类型化原因；仅供复核与可视化。",
    )
    raw_observation: dict = Field(
        default_factory=dict,
        description="上层禁读。VLM 自由文本、YOLO 逐帧框等实现特有产物。",
    )

    review_status: ReviewStatus = ReviewStatus.PENDING
    created_at: datetime

    @property
    def duration_ms(self) -> int:
        return self.t_end_ms - self.t_start_ms

    @property
    def is_abstained(self) -> bool:
        return self.confidence is Confidence.ABSTAIN

    @property
    def has_positional_verdict(self) -> bool:
        """能否对这一杆下走位结论。弃权时必须为 False（AD-5）。"""
        return (
            not self.is_abstained
            and self.deviation_mm is not None
            and self.target_zone_source is not TargetZoneSource.NONE
        )
