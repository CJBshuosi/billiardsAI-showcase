from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from uuid import UUID

import pytest

from billiards_ai.cloud.privacy.access_identity import FileReviewerIdentityProvider
from billiards_ai.cloud.review.cli import main
from billiards_ai.cloud.table_binding.repository import JsonTableBindingRepository

TASK_ID = "20000000-0000-4000-8000-000000000001"
REPORT_ID = "20000000-0000-4000-8000-000000000002"
DATABASE_URL = "postgresql+asyncpg://quality-gate@127.0.0.1/quality_gate"


class _Service:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    async def list_tasks(self, *, cursor=None, limit=50):
        self.calls.append(("list", (cursor, limit)))
        return {"items": [], "next_cursor": None}

    async def claim(self, command):
        self.calls.append(("claim", command))
        return {"task_id": str(command.task.task_id), "status": "IN_REVIEW"}

    async def show(self, task):
        self.calls.append(("show", task))
        return {"task_id": str(task.task_id), "status": "IN_REVIEW"}

    async def decide(self, command):
        self.calls.append(("decide", command))
        return {"task_id": str(command.task.task_id), "status": "VALID"}

    async def finalize(self, command):
        self.calls.append(("finalize", command))
        return {"task_id": str(command.task.task_id), "status": "APPROVED"}

    async def reject(self, command):
        self.calls.append(("reject", command))
        return {"task_id": str(command.task.task_id), "status": "REJECTED"}

    async def result(self, task):
        self.calls.append(("result", task))
        return {"task_id": str(task.task_id), "status": "APPROVED"}


def _environment(tmp_path: Path) -> dict[str, str]:
    return {
        "QUALITY_GATE_DATABASE_URL": DATABASE_URL,
        "QUALITY_GATE_TABLE_BINDING_STORE": str(tmp_path / "private" / "bindings.json"),
        "QUALITY_GATE_CALIBRATION_STORE": str(tmp_path / "private" / "calibrations.json"),
        "QUALITY_GATE_CALIBRATION_DEVICE_ID": "overhead-camera-test-1",
    }


def _task_args() -> list[str]:
    return [
        "--task-id",
        TASK_ID,
        "--expected-lock-version",
        "3",
        "--source-report-id",
        REPORT_ID,
        "--source-report-revision",
        "1",
    ]


def test_cli_list_uses_dedicated_database_and_trusted_local_providers(tmp_path: Path) -> None:
    service = _Service()
    captured = {}

    def factory(**kwargs):
        captured.update(kwargs)
        return service

    stdout, stderr = StringIO(), StringIO()
    exit_code = main(
        ["list", "--limit", "12"],
        service_factory=factory,
        environ=_environment(tmp_path),
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 0
    assert stderr.getvalue() == ""
    assert json.loads(stdout.getvalue()) == {
        "ok": True,
        "data": {"items": [], "next_cursor": None},
        "error": None,
    }
    assert captured["database_url"] == DATABASE_URL
    assert isinstance(captured["identity_provider"], FileReviewerIdentityProvider)
    assert isinstance(captured["table_binding_repository"], JsonTableBindingRepository)
    assert service.calls == [("list", (None, 12))]


@pytest.mark.parametrize(
    "missing",
    (
        "QUALITY_GATE_DATABASE_URL",
        "QUALITY_GATE_TABLE_BINDING_STORE",
        "QUALITY_GATE_CALIBRATION_STORE",
        "QUALITY_GATE_CALIBRATION_DEVICE_ID",
    ),
)
def test_cli_missing_dedicated_configuration_fails_closed(tmp_path: Path, missing: str) -> None:
    environment = _environment(tmp_path)
    environment.pop(missing)
    environment["DATABASE_URL"] = DATABASE_URL
    stdout, stderr = StringIO(), StringIO()

    exit_code = main(
        ["list"],
        service_factory=lambda **_kwargs: pytest.fail("factory must not be called"),
        environ=environment,
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 6
    assert stdout.getvalue() == ""
    assert json.loads(stderr.getvalue()) == {
        "ok": False,
        "data": None,
        "error": {"code": "DEPENDENCY_UNAVAILABLE", "message": "operation could not be completed"},
    }


def test_cli_rejects_non_asyncpg_database_url(tmp_path: Path) -> None:
    environment = _environment(tmp_path) | {
        "QUALITY_GATE_DATABASE_URL": "postgresql://quality-gate@localhost/db"
    }
    stderr = StringIO()

    exit_code = main(
        ["list"],
        service_factory=lambda **_kwargs: pytest.fail("factory must not be called"),
        environ=environment,
        stdout=StringIO(),
        stderr=stderr,
    )

    assert exit_code == 6
    assert json.loads(stderr.getvalue())["error"]["code"] == "DEPENDENCY_UNAVAILABLE"


@pytest.mark.parametrize("forbidden", ("--reviewer", "--credential", "--user", "--time", "--media"))
def test_cli_parser_rejects_forged_authority_flags(tmp_path: Path, forbidden: str) -> None:
    stderr = StringIO()
    exit_code = main(
        ["show", *_task_args(), forbidden, "forged"],
        service_factory=lambda **_kwargs: _Service(),
        environ=_environment(tmp_path),
        stdout=StringIO(),
        stderr=stderr,
    )

    assert exit_code == 2
    error = json.loads(stderr.getvalue())
    assert error["error"]["code"] == "INVALID_CONTRACT"
    assert "forged" not in stderr.getvalue()


def test_cli_claim_builds_strict_task_ref_and_idempotency_command(tmp_path: Path) -> None:
    service = _Service()
    stdout = StringIO()
    exit_code = main(
        ["claim", *_task_args(), "--idempotency-key", "qg-cli-claim-000-0123456789abcdef"],
        service_factory=lambda **_kwargs: service,
        environ=_environment(tmp_path),
        stdout=stdout,
        stderr=StringIO(),
    )

    assert exit_code == 0
    name, command = service.calls[-1]
    assert name == "claim"
    assert command.task.task_id == UUID(TASK_ID)
    assert command.idempotency_key == "qg-cli-claim-000-0123456789abcdef"
    assert json.loads(stdout.getvalue())["ok"] is True


def _write_review_inputs(tmp_path: Path) -> tuple[Path, Path]:
    decisions = tmp_path / "decisions.json"
    metrics = tmp_path / "metrics.json"
    decisions.write_text(
        json.dumps(
            [
                {
                    "shot_id": "20000000-0000-4000-8000-000000000010",
                    "outcome": "approved",
                    "trajectory_reviewed": True,
                    "corrections": [],
                }
            ]
        )
    )
    unsupported_states = {
        name: "UNSUPPORTED"
        for name in (
            "geometry",
            "intent",
            "pot_attempt",
            "pot_result",
            "foul_result",
            "target_zone",
            "cue_ball_end",
        )
    }
    metrics.write_text(
        json.dumps(
            [
                {
                    "shot_id": "20000000-0000-4000-8000-000000000010",
                    "cue_ball_start_mm": None,
                    "object_ball_start_mm": None,
                    "intended_object_ball": None,
                    "intended_target_pocket": None,
                    "pot_attempt_status": "UNSUPPORTED",
                    "pot_result": None,
                    "foul_result": "UNSUPPORTED",
                    "target_zone_source": "none",
                    "target_zone": None,
                    "cue_ball_end_mm": None,
                    "target_zone_evaluation_status": "UNSUPPORTED",
                    "review_states": unsupported_states,
                }
            ]
        )
    )
    return decisions, metrics


def test_cli_dispatches_show_decide_finalize_reject_and_result(tmp_path: Path) -> None:
    service = _Service()
    decisions, metrics = _write_review_inputs(tmp_path)
    commands = (
        ["show", *_task_args()],
        ["decide", *_task_args(), "--decisions", str(decisions), "--metric-shots", str(metrics)],
        [
            "finalize",
            *_task_args(),
            "--decisions",
            str(decisions),
            "--metric-shots",
            str(metrics),
            "--idempotency-key",
            "qg-cli-finalize-0123456789abcdef",
        ],
        [
            "reject",
            *_task_args(),
            "--reason",
            "occluded",
            "--idempotency-key",
            "qg-cli-reject-00-0123456789abcdef",
        ],
        ["result", *_task_args()],
    )

    for command in commands:
        assert (
            main(
                command,
                service_factory=lambda **_kwargs: service,
                environ=_environment(tmp_path),
                stdout=StringIO(),
                stderr=StringIO(),
            )
            == 0
        )

    assert tuple(name for name, _value in service.calls) == (
        "show",
        "decide",
        "finalize",
        "reject",
        "result",
    )


def test_cli_decide_rejects_idempotency_key(tmp_path: Path) -> None:
    decisions, metrics = _write_review_inputs(tmp_path)
    stderr = StringIO()
    exit_code = main(
        [
            "decide",
            *_task_args(),
            "--decisions",
            str(decisions),
            "--metric-shots",
            str(metrics),
            "--idempotency-key",
            "must-not-be-accepted",
        ],
        service_factory=lambda **_kwargs: _Service(),
        environ=_environment(tmp_path),
        stdout=StringIO(),
        stderr=stderr,
    )

    assert exit_code == 2
    assert json.loads(stderr.getvalue())["error"]["code"] == "INVALID_CONTRACT"


def test_async_cli_factory_and_service_share_one_event_loop(tmp_path: Path) -> None:
    class LoopBoundService(_Service):
        def __init__(self, loop) -> None:
            super().__init__()
            self.loop = loop

        async def list_tasks(self, *, cursor=None, limit=50):
            import asyncio

            assert asyncio.get_running_loop() is self.loop
            return await super().list_tasks(cursor=cursor, limit=limit)

    async def factory(**_kwargs):
        import asyncio

        return LoopBoundService(asyncio.get_running_loop())

    assert (
        main(
            ["list"],
            service_factory=factory,
            environ=_environment(tmp_path),
            stdout=StringIO(),
            stderr=StringIO(),
        )
        == 0
    )


def test_cli_unhandled_exception_cannot_forge_a_public_error_code(tmp_path: Path) -> None:
    class ForgedCodeError(RuntimeError):
        code = "REVIEW_ACCESS_DENIED"

    class FailingService(_Service):
        async def list_tasks(self, *, cursor=None, limit=50):
            raise ForgedCodeError("not an application error")

    stderr = StringIO()
    exit_code = main(
        ["list"],
        service_factory=lambda **_kwargs: FailingService(),
        environ=_environment(tmp_path),
        stdout=StringIO(),
        stderr=stderr,
    )

    assert exit_code == 70
    assert json.loads(stderr.getvalue())["error"]["code"] == "INTERNAL_ERROR"


@pytest.mark.parametrize("key", ("a" * 32, "abcd" * 8, "0123456789abcdef0123456789abcde"))
def test_cli_rejects_low_entropy_shaped_idempotency_keys(tmp_path: Path, key: str) -> None:
    service = _Service()
    stderr = StringIO()
    exit_code = main(
        ["claim", *_task_args(), "--idempotency-key", key],
        service_factory=lambda **_kwargs: service,
        environ=_environment(tmp_path),
        stdout=StringIO(),
        stderr=stderr,
    )

    assert exit_code == 2
    assert json.loads(stderr.getvalue())["error"]["code"] == "INVALID_CONTRACT"
    assert service.calls == []
