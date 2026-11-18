"""Typed report-delivery failures without draft payload disclosure."""

from billiards_ai.shared.models.review import ReportNotDeliverableReason


class ReportNotDeliverableError(RuntimeError):
    def __init__(self, reason: ReportNotDeliverableReason):
        self.reason = reason
        super().__init__(reason.value)
