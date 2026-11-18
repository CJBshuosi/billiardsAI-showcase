"""Synthetic, non-inference identity inventory for development policy tests."""

import hashlib
import json


def write_identity_assets(root, *, gate_status="FAIL", gate_violations=None):
    root.mkdir(parents=True, exist_ok=True)
    (root / "group-head.onnx").write_bytes(b"synthetic inventory only; not an ONNX model")
    (root / "metrics.json").write_text(
        json.dumps(
            {
                "unknown_policy": {
                    "confidence_min": 0.55,
                    "margin_min": 0.10,
                    "known_probability_min": 0.30,
                },
                "visual_evidence_policy": {"weight": 0.5},
            }
        )
    )
    manifest = {
        "run_id": root.name,
        "status": "PASS",
        "gate_status": gate_status,
        "gate_violations": ["solid_f1 below threshold"]
        if gate_violations is None
        else gate_violations,
        "backend_configuration": {"architecture": "hierarchical_fused_evidence_mlp"},
        "inventory": {
            name: {"sha256": hashlib.sha256((root / name).read_bytes()).hexdigest()}
            for name in ("metrics.json", "group-head.onnx")
        },
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    return {
        "mode": "development_model_acceptance",
        "identity_gate_status": "FAIL",
        "identity_training_package_id": root.name,
        "identity_training_package_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "gate_violations": manifest["gate_violations"],
        "formal_model_qualified": False,
        "formal_field_acceptance": "NOT_ASSERTED",
    }
