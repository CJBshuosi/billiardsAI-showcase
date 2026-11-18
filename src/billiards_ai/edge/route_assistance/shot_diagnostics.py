"""Pure classification of retained nonmaterialized capture-loss evidence."""

from billiards_ai.edge.route_assistance.records import FieldShotRecord


def is_capture_interruption(shot: FieldShotRecord) -> bool:
    """Recognize exact capture loss and the retained legacy freeze diagnostic."""
    epoch = shot.media.source_epoch_id
    return (
        shot.settled_ms is not None
        and shot.media.source_kind == "usb_live"
        and bool(epoch)
        and shot.live_evidence_digest is not None
        and shot.live_job_id is not None
        and shot.live_job_id
        in {
            f"abstained:capture_interrupted:{epoch}:{shot.shot_id}",
            f"capture-interrupted:{epoch}:{shot.shot_id}",
            # Retained-data compatibility, not a claim of current producer output.
            f"abstained:incomplete_freeze:{epoch}:{shot.shot_id}",
        }
    )
