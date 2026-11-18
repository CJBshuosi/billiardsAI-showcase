"""Prepare exact ball crops and launch the resumable loopback review UI."""

from __future__ import annotations

import json
import webbrowser
from collections.abc import Callable
from pathlib import Path
from typing import Any

from billiards_ai.lab.training.identity_crop_export import export_identity_crop_candidates
from billiards_ai.lab.training.identity_review import IdentityReviewSession
from billiards_ai.lab.training.identity_review_server import make_identity_review_server


def run_identity_review_cli(
    *,
    bootstrap_dataset: Path,
    output_root: Path,
    port: int,
    open_browser: bool,
    server_factory: Callable[..., Any] = make_identity_review_server,
) -> int:
    candidates = export_identity_crop_candidates(
        bootstrap_dataset=bootstrap_dataset,
        output_root=output_root / "candidates",
    )
    return run_existing_identity_review_cli(
        candidate_artifact=candidates.path,
        output_root=output_root,
        port=port,
        open_browser=open_browser,
        server_factory=server_factory,
    )


def run_existing_identity_review_cli(
    *,
    candidate_artifact: Path,
    output_root: Path,
    port: int,
    open_browser: bool,
    server_factory: Callable[..., Any] = make_identity_review_server,
) -> int:
    candidate_artifact = candidate_artifact.resolve()
    state_path = output_root / "states" / f"{candidate_artifact.name}.json"
    session = IdentityReviewSession.open(
        candidate_artifact=candidate_artifact,
        state_path=state_path,
    )
    server = server_factory(
        session=session,
        export_root=output_root / "verdicts",
        port=port,
    )
    url = f"http://127.0.0.1:{server.server_port}/"
    print(
        json.dumps(
            {
                "status": "READY",
                "url": url,
                "candidate_artifact": str(candidate_artifact),
                "candidate_id": session.candidate_id,
                "crop_count": len(session.crop_ids),
                "reviewed_count": session.reviewed_count,
                "state_path": str(state_path),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
