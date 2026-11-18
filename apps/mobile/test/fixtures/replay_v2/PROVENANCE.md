# Task 5 replay vector provenance

All three files are exact canonical bytes issued by the authoritative
`issue_realtime_coaching_event_v2` factory and serialized by
`canonical_realtime_coaching_event_bytes` at reviewed commit
`6825068e85e4eecfe89a19b8a8854fb4c2f454dd`.

## Frozen vectors

| File | Bytes | SHA-256 |
|---|---:|---|
| `sequence_10.json` | 2114 | `702095ca7083a58196ebc14a07ec0898e11dae7e735b0ff6f4282ca370f3774c` |
| `sequence_11.json` | 2114 | `11378d9c3b1bbc89bd253e8baf5a77d590eb3abf84356f638752b76f25dc0b29` |
| `conflicting_sequence_9.json` | 2099 | `98dcd9f867b022695d73a98182bac881459df61864a2de22ba206184dca72366` |

Every file intentionally has no trailing newline. Sequence 10 and 11 retain
the source test factory's event material and use monotonic times 8000 and 8500.
The conflict vector retains sequence 9's identity material and uses an
authority-validated alternate speech cue and summary. No digest is edited.

## Exact reproduction invocation

Run this single block from the App repository root with its Python 3.12.13
environment available at `.venv`. It archives the exact authority tree,
installs that archived frozen uv project and reproduces all three files.

```bash
APP_REPO_ROOT="$(pwd -P)"
UPSTREAM_COMMIT="6825068e85e4eecfe89a19b8a8854fb4c2f454dd"
UPSTREAM_DIR="$(mktemp -d /private/tmp/billiardsAI-replay-v2-authority.XXXXXX)"
git archive "$UPSTREAM_COMMIT" | tar -x -C "$UPSTREAM_DIR"
UV_PYTHON="$APP_REPO_ROOT/.venv/bin/python3" uv sync --project "$UPSTREAM_DIR" --frozen --extra dev
PYTHONPATH="$UPSTREAM_DIR/src" uv run --project "$UPSTREAM_DIR" --frozen python -c 'import runpy,sys; from pathlib import Path; from billiards_ai.shared.models.realtime import SpeechCueV1,canonical_realtime_coaching_event_bytes as cb,issue_realtime_coaching_event_v2 as issue; n=runpy.run_path(sys.argv[1]); out=Path(sys.argv[2]); out.mkdir(parents=True,exist_ok=True); base=n["_event"](); emit=lambda name,event:(out/name).write_bytes(cb(event)); emit("sequence_10.json",issue(sequence=10,event_type=base.event_type,outcome_status=base.outcome_status,delivery_status=base.delivery_status,lineage=base.lineage,overlay=base.overlay,speech=base.speech,text_summary=base.text_summary,created_monotonic_ms=8000)); emit("sequence_11.json",issue(sequence=11,event_type=base.event_type,outcome_status=base.outcome_status,delivery_status=base.delivery_status,lineage=base.lineage,overlay=base.overlay,speech=base.speech,text_summary=base.text_summary,created_monotonic_ms=8500)); speech=SpeechCueV1(cue_code="shot-success-alternate",outcome_status=base.outcome_status,delivery_status=base.delivery_status,text="另一条合法反馈。"); emit("conflicting_sequence_9.json",issue(sequence=9,event_type=base.event_type,outcome_status=base.outcome_status,delivery_status=base.delivery_status,lineage=base.lineage,overlay=base.overlay,speech=speech,text_summary=speech.text,created_monotonic_ms=base.created_monotonic_ms))' "$UPSTREAM_DIR/tests/unit/shared/models/test_realtime_coaching_v2.py" "$APP_REPO_ROOT/apps/mobile/test/fixtures/replay_v2"
```
