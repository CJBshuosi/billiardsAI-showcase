"""台面平面坐标系（AD-13a）。

原点在左下袋口中心，X 沿长边向右，Y 沿短边向上，单位毫米整数。
图像像素坐标不得进入本模块以外的任何数据结构——像素到台面的映射由
shared.calibration 负责（AD-18）。
"""

from pydantic import BaseModel, ConfigDict, Field, StrictInt

from billiards_ai.shared.constants import TABLE_LENGTH_MM, TABLE_WIDTH_MM


class Point(BaseModel):
    """台面上的一个位置，毫米整数。"""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    x_mm: StrictInt
    y_mm: StrictInt

    def distance_to(self, other: "Point") -> int:
        dx = self.x_mm - other.x_mm
        dy = self.y_mm - other.y_mm
        return round((dx * dx + dy * dy) ** 0.5)

    def is_on_table(self) -> bool:
        return 0 <= self.x_mm <= TABLE_LENGTH_MM and 0 <= self.y_mm <= TABLE_WIDTH_MM


class Zone(BaseModel):
    """台面上的一块矩形区域，用于表达走位目标区。"""

    center: Point
    radius_mm: int = Field(gt=0)

    def contains(self, p: Point) -> bool:
        return self.center.distance_to(p) <= self.radius_mm

    def deviation_of(self, p: Point) -> int:
        """点到区域边界的距离；区域内为 0。"""
        return max(0, self.center.distance_to(p) - self.radius_mm)
