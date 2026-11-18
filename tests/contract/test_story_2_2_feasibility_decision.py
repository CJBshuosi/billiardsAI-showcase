from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = (
    PROJECT_ROOT / "_bmad-output" / "implementation-artifacts" / "feasibility" / "story-2.2"
)
SOURCE_ROOT = (
    PROJECT_ROOT / "_bmad-output" / "implementation-artifacts" / "feasibility" / "story-2.1"
)
EXPECTED_ARTIFACTS = {
    "benchmark-plan.md",
    "benchmark-results.json",
    "dataset-manifest.json",
    "decision-record.json",
    "failure-matrix.json",
    "failure-matrix.md",
    "feasibility-report.md",
    "ground-truth-schema.md",
    "thresholds.json",
    "verify-decision.py",
}
EXPECTED_CANDIDATES = {
    "Interactive-Pool": (
        "479508d7504a9b540165482cf477220c84c7cd65",
        "REPRODUCTION_FAILED",
        "MIT_CONFIRMED",
    ),
    "pool-telemetry-web": (
        "83ec155add823bebb3a31795461aba57e9078a97",
        "REPRODUCTION_FAILED",
        "MIT_CONFIRMED",
    ),
    "CueDetat": (
        "240a9d5f693f4712c670e280102495267ffeb6ef",
        "REPRODUCTION_FAILED",
        "MIT_CONFIRMED",
    ),
    "DEADPOOL_Project": (
        "2e974916fb888f212e57d5baae3ec316d71825dd",
        "REJECTED_LICENSE",
        "NO_LICENSE_FOUND",
    ),
    "AISnooker": (
        "4a2df91bbe446fb4a2021317d66997f7f0f6e97f",
        "REJECTED_LICENSE",
        "NO_LICENSE_FOUND",
    ),
}


def _load_json(name: str) -> dict:
    path = ARTIFACT_ROOT / name
    assert path.is_file(), f"missing Story 2.2 artifact: {path.relative_to(PROJECT_ROOT)}"
    return json.loads(path.read_text(encoding="utf-8"))


def _run_verifier(artifact_root: Path = ARTIFACT_ROOT) -> subprocess.CompletedProcess[str]:
    script = artifact_root / "verify-decision.py"
    assert script.is_file(), "Story 2.2 decision verifier is missing"
    return subprocess.run(
        [
            sys.executable,
            str(script),
            "--artifact-root",
            str(artifact_root),
            "--story-2-1-root",
            str(SOURCE_ROOT),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_story_2_2_bundle_is_complete_and_contains_only_safe_small_documents() -> None:
    assert ARTIFACT_ROOT.is_dir(), "Story 2.2 feasibility artifact directory is missing"
    files = {path.name for path in ARTIFACT_ROOT.iterdir() if path.is_file()}
    assert files == EXPECTED_ARTIFACTS

    forbidden_suffixes = {
        ".avi",
        ".env",
        ".key",
        ".mkv",
        ".mov",
        ".mp4",
        ".onnx",
        ".pem",
        ".pt",
        ".tflite",
        ".weights",
    }
    for path in ARTIFACT_ROOT.rglob("*"):
        if path.is_file():
            assert path.suffix.lower() not in forbidden_suffixes
            assert path.stat().st_size < 1_000_000

    combined_text = "\n".join(
        path.read_text(encoding="utf-8") for path in ARTIFACT_ROOT.rglob("*") if path.is_file()
    )
    for pattern in (
        r"AIza[0-9A-Za-z_-]{20,}",
        r"sk-ant-[0-9A-Za-z_-]{20,}",
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    ):
        assert re.search(pattern, combined_text) is None


def test_empty_benchmark_is_explicitly_not_executed_and_cannot_claim_metrics() -> None:
    manifest = _load_json("dataset-manifest.json")
    assert manifest["schema_version"] == "1.0"
    assert manifest["story_id"] == "2.2"
    assert manifest["revision"] == "story-2.2-r0"
    assert manifest["status"] == "NOT_EXECUTED"
    assert manifest["real_video_gate"]["required_status"] == "APPROVED"
    assert manifest["real_video_gate"]["current_status"] == "UNAVAILABLE"
    assert manifest["real_video_used"] is False
    assert manifest["assets"] == []
    assert manifest["shot_count"] == 0

    results = _load_json("benchmark-results.json")
    assert results["schema_version"] == "1.0"
    assert results["benchmark_revision"] == "story-2.2-r0"
    assert results["status"] == "NOT_EXECUTED"
    assert set(results["reason_codes"]) == {
        "NO_RUNNABLE_CANDIDATE",
        "REAL_VIDEO_GATE_DATASET_UNAVAILABLE",
        "THRESHOLDS_NOT_APPROVED_PRE_RESULT",
    }
    assert results["candidate_runs"] == []
    assert results["metric_results"] == []
    assert results["passed"] is None
    assert results["direct_adoption_qualified"] is None
    assert results["metrics_fabricated"] is False


def test_future_protocol_freezes_required_metrics_but_remains_unapproved_for_execution() -> None:
    thresholds = _load_json("thresholds.json")
    assert thresholds["schema_version"] == "1.0"
    assert thresholds["benchmark_revision"] == "story-2.2-r0"
    assert thresholds["frozen"] is True
    assert thresholds["execution_authorized"] is False
    assert thresholds["approval"] == {
        "required_approver": "Boyu",
        "status": "PENDING_PRE_RESULT_APPROVAL",
        "approved_at": None,
        "result_visibility_required_at_approval": "NO_RESULTS_VISIBLE",
    }
    metric_by_id = {metric["id"]: metric for metric in thresholds["metrics"]}
    assert {
        "shot_boundary",
        "ball_localization",
        "ball_classification",
        "struck_ball_recognition",
        "potted_ball_recognition",
        "cue_ball_endpoint",
        "cushion_contact",
        "spin_support",
        "coverage",
        "runtime",
        "cost",
    } <= set(metric_by_id)
    assert metric_by_id["ball_localization"]["direct_adoption_reject_if"]["operator"] == ">"
    assert metric_by_id["ball_localization"]["direct_adoption_reject_if"]["value"] == 20
    assert metric_by_id["ball_localization"]["unit"] == "mm"
    struck_ball = metric_by_id["struck_ball_recognition"]
    assert struck_ball["direct_adoption_reject_if"] == {
        "operator": "<",
        "value": 0.7,
    }


def test_failure_matrix_matches_every_fixed_story_2_1_candidate_and_evidence() -> None:
    matrix = _load_json("failure-matrix.json")
    assert matrix["schema_version"] == "1.0"
    assert matrix["story_id"] == "2.2"
    assert matrix["source_story"] == "2.1"
    assert matrix["runnable_candidate_count"] == 0
    assert matrix["benchmark_eligible_candidate_count"] == 0
    assert matrix["benchmark_status"] == "NOT_EXECUTED"

    candidates = {candidate["name"]: candidate for candidate in matrix["candidates"]}
    assert set(candidates) == set(EXPECTED_CANDIDATES)
    for name, (commit, status, license_status) in EXPECTED_CANDIDATES.items():
        candidate = candidates[name]
        assert candidate["commit"] == commit
        assert candidate["story_2_1_status"] == status
        assert candidate["license"]["status"] == license_status
        assert candidate["benchmark_eligibility"] == "NOT_ELIGIBLE"
        assert candidate["benchmark_exclusion_reasons"]
        assert candidate["reproduction_evidence"]["result_ref"].startswith(
            "../story-2.1/reproduction-results.json#"
        )
        assert candidate["reproduction_evidence"]["commands"]
        assert candidate["license"]["evidence_ref"].startswith("../story-2.1/license-review.md#")
        assert {
            "MILLIMETRE_COORDINATES_UNVERIFIED",
            "REPRODUCIBLE_SHOT_BOUNDARIES_UNVERIFIED",
            "LAYERED_OBSERVATION_CONTRACT_UNVERIFIED",
            "EXPLICIT_UNKNOWN_AND_ABSTENTION_UNVERIFIED",
            "NORMALIZED_MEDIA_CONTRACT_UNVERIFIED",
        } <= set(candidate["ad_13_gaps"])


def test_build_authorization_is_narrow_ordered_and_does_not_bypass_gates() -> None:
    decision = _load_json("decision-record.json")
    assert decision["schema_version"] == "1.0"
    assert decision["story_id"] == "2.2"
    assert decision["decision"] == "BUILD"
    assert decision["decision_basis"] == "APPROVED_STORY_2_1_FAILURE_MATRIX"
    assert decision["benchmark_status"] == "NOT_EXECUTED"
    assert decision["approval"]["approver"] == "Boyu"
    assert decision["approval"]["approved_on"] == "2026-07-28"
    assert decision["authorization"]["dependency_order"] == [
        "2.3",
        "2.4",
        "2.5",
        "2.6",
        "2.7",
    ]
    assert decision["authorization"]["parallel_or_early_start_allowed"] is False
    assert set(decision["authorization"]["does_not_authorize"]) == {
        "REAL_VIDEO_GATE_BYPASS",
        "EXTERNAL_VLM_OR_VENDOR_CALLS",
        "API_KEYS_OR_SECRETS",
        "MODEL_TRAINING",
        "SKIP_CALIBRATION",
        "SKIP_NORMALIZED_MEDIA",
        "EARLY_OR_OUT_OF_ORDER_STORY_DEVELOPMENT",
    }
    assert decision["external_vlm"] == {
        "status": "UNVERIFIED_NOT_APPLICABLE",
        "provider_selected": False,
        "data_sent": False,
        "api_key_used": False,
    }


def test_verifier_cross_checks_sources_and_fails_closed_on_fabricated_execution(
    tmp_path: Path,
) -> None:
    verified = _run_verifier()
    assert verified.returncode == 0, verified.stderr
    summary = json.loads(verified.stdout)
    assert summary == {
        "benchmark_status": "NOT_EXECUTED",
        "candidate_count": 5,
        "decision": "BUILD",
        "status": "VERIFIED",
    }

    tampered_root = tmp_path / "story-2.2"
    shutil.copytree(ARTIFACT_ROOT, tampered_root)
    results_path = tampered_root / "benchmark-results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    results["status"] = "PASSED"
    results_path.write_text(json.dumps(results), encoding="utf-8")

    rejected = _run_verifier(tampered_root)
    assert rejected.returncode == 1
    assert "benchmark status must be NOT_EXECUTED" in rejected.stderr
