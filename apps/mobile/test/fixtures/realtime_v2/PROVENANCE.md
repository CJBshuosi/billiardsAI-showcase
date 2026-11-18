# Realtime coaching V2 fixture provenance

`post_shot_success.json` is a temporary, reviewed contract fixture because the
team integration branch did not yet contain a formal `RealtimeCoachingEventV2`
integration when App A2.1 Task 3 began. Its reviewed authority is the exact Git
commit below; the fixture must be replaced or re-approved when `dev` gains its
formal V2 contract.

## Reviewed authority

- Upstream commit: `6825068e85e4eecfe89a19b8a8854fb4c2f454dd`
- Source model: `src/billiards_ai/shared/models/realtime.py`
- Source test: `tests/unit/shared/models/test_realtime_coaching_v2.py`
| Fixture | State | Bytes | SHA-256 |
|---|---|---:|---|
| `post_shot_success.json` | normal success | 2113 | `a90a29dcd7602d8f549ea36de268c3925435a371fdee9cf8caaa3085a6440bc4` |
| `abstained.json` | abstained | 1558 | `c1d5e0d07f329ab367b3092168a375bae46b06fa666b5f19afa3407cbbb7489c` |
| `unsupported.json` | unsupported | 1562 | `cc3f2a1b3b6d8576c66d952472335703031a4d70cd0f7106c19d87e0e7834684` |
| `degraded.json` | degraded | 1549 | `0fa158834aad0a373782a0e2b4a4480133e0b47380a0d13d73746158d2c2d22a` |

## Exact generation invocation

The following command is run from the repository root. It expands the exact
reviewed tree into an isolated temporary directory and writes all four states
through the upstream issuer and canonical serializer. The repository's pinned
Python 3.12.13 interpreter is selected through `UV_PYTHON`.

```bash
APP_REPO_ROOT="$(pwd -P)"
UPSTREAM_COMMIT="6825068e85e4eecfe89a19b8a8854fb4c2f454dd"
UPSTREAM_DIR="$(mktemp -d /private/tmp/billiardsAI-realtime-v2-6825068.XXXXXX)"
git archive "$UPSTREAM_COMMIT" | tar -x -C "$UPSTREAM_DIR"
UV_PYTHON="$APP_REPO_ROOT/.venv/bin/python3" uv sync --project "$UPSTREAM_DIR" --frozen --extra dev
PYTHONPATH="$UPSTREAM_DIR/src" uv run --project "$UPSTREAM_DIR" --frozen python -c 'import runpy,sys; from pathlib import Path; from billiards_ai.shared.models.realtime import CoachingDeliveryStatus as D,CoachingEventType as E,CoachingOutcomeStatus as O,SpeechCueV1,canonical_realtime_coaching_event_bytes as cb,issue_realtime_coaching_event_v2 as issue; n=runpy.run_path(sys.argv[1]); out=Path(sys.argv[2]); out.mkdir(parents=True,exist_ok=True); emit=lambda name,event:(out/name).write_bytes(cb(event)); base=n["_event"](); emit("post_shot_success.json",base); make=lambda sequence,event_type,outcome,delivery,text,monotonic:issue(sequence=sequence,event_type=event_type,outcome_status=outcome,delivery_status=delivery,lineage=n["_lineage"](),overlay=None,speech=SpeechCueV1(cue_code="app-a2-safe-state",outcome_status=outcome,delivery_status=delivery,text=text),text_summary=text,created_monotonic_ms=monotonic); emit("abstained.json",make(12,E.ABSTENTION,O.ABSTAINED,D.NORMAL,"本杆证据不足，系统选择弃权。",7600)); emit("unsupported.json",make(13,E.ABSTENTION,O.UNSUPPORTED,D.NORMAL,"当前动作不在已验证支持范围。",7700)); emit("degraded.json",make(14,E.SYSTEM_STATUS,O.INFORMATIONAL,D.DEGRADED,"事件时间线暂不可用。",7800))' "$UPSTREAM_DIR/tests/unit/shared/models/test_realtime_coaching_v2.py" "$APP_REPO_ROOT/apps/mobile/test/fixtures/realtime_v2"
```

The generated files are canonical UTF-8 JSON and intentionally have no
trailing newline. Do not pretty-print, reorder or hand-edit them.
