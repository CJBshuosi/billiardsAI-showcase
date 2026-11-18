from __future__ import annotations

import json
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from billiards_ai.lab.training.identity_review import IdentityReviewSession
from billiards_ai.lab.training.identity_review_server import make_identity_review_server
from tests.unit.lab.training.test_identity_review import _session


def _json_request(url: str, *, method: str = "GET", payload: object | None = None) -> object:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        method=method,
        headers={"Content-Type": "application/json"} if body else {},
    )
    with urlopen(request, timeout=3) as response:  # noqa: S310 - loopback test server
        return json.loads(response.read().decode("utf-8"))


def test_loopback_review_server_records_decisions_serves_media_and_exports(
    tmp_path: Path,
) -> None:
    session = _session(tmp_path)
    server = make_identity_review_server(
        session=session,
        export_root=tmp_path / "verdicts",
        port=0,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        state = _json_request(f"{base}/api/session")
        assert state["total_count"] == 2
        crop_ids = state["crop_ids"]
        with urlopen(f"{base}/media/{crop_ids[0]}.png", timeout=3) as response:  # noqa: S310
            assert response.headers["Content-Type"] == "image/png"
            assert response.read().startswith(b"\x89PNG")
        for crop_id in crop_ids:
            state = _json_request(
                f"{base}/api/decision",
                method="POST",
                payload={
                    "crop_id": crop_id,
                    "group": "solid",
                    "exact_identity": None,
                    "exact_evidence": None,
                    "visibility": "visible",
                    "orientation": "number_hidden",
                    "note": None,
                },
            )
        assert state["reviewed_count"] == 2
        exported = _json_request(f"{base}/api/export", method="POST")
        assert Path(exported["path"]).joinpath("verdict.json").is_file()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)

    reopened = IdentityReviewSession.open(
        candidate_artifact=session.candidate_artifact,
        state_path=session.state_path,
    )
    assert reopened.reviewed_count == 2
    assert all(decision.group == "solid" for decision in reopened.decisions.values())
    assert all(decision.exact_identity is None for decision in reopened.decisions.values())


def test_review_server_rejects_non_loopback_and_unknown_media(tmp_path: Path) -> None:
    session = _session(tmp_path)
    with pytest.raises(ValueError, match="loopback"):
        make_identity_review_server(
            session=session,
            export_root=tmp_path / "verdicts",
            port=0,
            host="0.0.0.0",
        )

    server = make_identity_review_server(
        session=session,
        export_root=tmp_path / "verdicts",
        port=0,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(HTTPError) as error:
            urlopen(  # noqa: S310 - loopback test server
                f"http://127.0.0.1:{server.server_port}/media/not-a-crop.png",
                timeout=3,
            )
        assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_review_payload_exposes_error_context_and_unreviewed_suggestion(
    tmp_path: Path,
) -> None:
    original = _session(tmp_path)
    crop_id = original.crop_ids[0]
    original.record(
        crop_id=crop_id,
        group="cue",
        exact_identity="cue",
        exact_evidence="group_intrinsic",
        visibility="visible",
        orientation="number_visible",
    )
    subset = type(original).open(
        candidate_artifact=original.candidate_artifact,
        state_path=tmp_path / "subset.json",
        crop_ids=(crop_id,),
        suggestions={crop_id: original.decisions[crop_id]},
        contexts={crop_id: {"truth": "cue", "predicted": "stripe"}},
    )
    server = make_identity_review_server(
        session=subset,
        export_root=tmp_path / "verdicts",
        port=0,
    )
    try:
        payload = server.session_payload()
    finally:
        server.server_close()

    assert payload["reviewed_count"] == 0
    assert payload["suggestions"][crop_id]["group"] == "cue"
    assert payload["contexts"][crop_id]["predicted"] == "stripe"
