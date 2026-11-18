"""Session / Recording / ReviewTask。

async 变体下会话由云端创建并全程拥有——边缘不知道用户是谁，
它对会话的认知仅限于一个 id 和一对时间戳（AD-11 变体条款）。
"""

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, Field

from billiards_ai.shared.models.review import (
    ReviewCorrection as _ReviewCorrection,
)
from billiards_ai.shared.models.review import (
    ReviewTask as ReviewTask,
)

Correction = _ReviewCorrection


class SessionStatus(StrEnum):
    RECORDING = "recording"
    PROCESSING = "processing"
    REVIEWING = "reviewing"
    DELIVERED = "delivered"
    REJECTED = "rejected"


class EndReason(StrEnum):
    STAFF = "staff"
    IDLE_TIMEOUT = "idle_timeout"
    MANUAL_BACKFILL = "manual_backfill"


class UploadStatus(StrEnum):
    QUEUED = "queued"
    UPLOADING = "uploading"
    DONE = "done"
    FAILED = "failed"


class FourDimScores(BaseModel):
    """等级测试与训练报告共用的四维度。0-100。"""

    potting: int = Field(ge=0, le=100, description="进球")
    positioning: int = Field(ge=0, le=100, description="走位")
    power_control: int = Field(ge=0, le=100, description="力度控制")
    spin_usage: int = Field(ge=0, le=100, description="旋转运用")


class Session(BaseModel):
    id: UUID
    user_id: UUID
    device_id: str = Field(description="球厅 + 台号")

    start_ts: datetime = Field(description="服务器时间，绝不采用客户端上报（AD-15）")
    end_ts: datetime | None = None
    end_reason: EndReason | None = None

    recording_id: UUID | None = None
    shot_count: int = 0
    scores: FourDimScores | None = None
    status: SessionStatus = SessionStatus.RECORDING


class Recording(BaseModel):
    """async 上传对象；留存权威状态由 retention service 管理。"""

    id: UUID
    device_id: str
    start_ts: datetime
    end_ts: datetime
    oss_key: str
    duration_ms: int = Field(ge=0)
    size_bytes: int = Field(ge=0)
    upload_status: UploadStatus = UploadStatus.QUEUED
    retention_asset_id: UUID | None = Field(default=None, description="关联权威 RetainedMediaAsset")
    delete_after: datetime | None = Field(
        default=None,
        deprecated=True,
        description="仅供旧客户端读的 retention_expires_at 投影，不是写入权威来源",
    )
