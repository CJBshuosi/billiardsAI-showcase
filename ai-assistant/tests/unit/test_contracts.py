import copy

import pytest
from pydantic import ValidationError

from cueorbit_replay.contracts import ReplayBundleV1, ToolCall
from cueorbit_replay.fixtures import load_builtin_bundles


@pytest.mark.parametrize("bad", [True, "1", 1.2, float("inf")])
def test_strict_revision(bad):
    obj = load_builtin_bundles()[0].model_dump(mode="json")
    obj["revision"] = bad
    with pytest.raises(ValidationError):
        ReplayBundleV1.model_validate(obj)


@pytest.mark.parametrize(
    "mutation",
    ["duplicate", "foreign", "dangling", "time", "duration", "path", "evidence"],
)
def test_reject_invalid_bundle(mutation):
    obj = copy.deepcopy(load_builtin_bundles()[0].model_dump(mode="json"))
    if mutation == "duplicate":
        obj["shots"][1]["sequence"] = 1
    if mutation == "foreign":
        obj["shots"][0]["session_id"] = "other"
    if mutation == "dangling":
        obj["shots"][0]["media_id"] = "missing"
    if mutation == "time":
        obj["shots"][0]["end_ms"] = 0
    if mutation == "duration":
        obj["shots"][0]["end_ms"] = 999999
    if mutation == "path":
        obj["media"][0]["filename"] = "../secret"
    if mutation == "evidence":
        obj["shots"][1]["evidence_id"] = obj["shots"][0]["evidence_id"]
    with pytest.raises(ValidationError):
        ReplayBundleV1.model_validate(obj)


def test_boundary_json_immutable_and_finite():
    call = ToolCall(
        call_id="a", name="get_shot", arguments={"sequence": 1, "nested": [1]}
    )
    with pytest.raises(TypeError):
        call.arguments["sequence"] = 2
    with pytest.raises(TypeError):
        call.arguments["nested"].append(2)
    with pytest.raises(ValidationError):
        ToolCall(call_id="a", name="get_shot", arguments={"x": float("nan")})


def test_imported_media_duration_cannot_expand_authoritative_clip():
    obj = load_builtin_bundles()[0].model_dump(mode="json")
    obj["media"][0]["duration_ms"] = 100000
    obj["shots"][0]["start_ms"] = 89000
    obj["shots"][0]["end_ms"] = 90000
    with pytest.raises(ValidationError):
        ReplayBundleV1.model_validate(obj)
