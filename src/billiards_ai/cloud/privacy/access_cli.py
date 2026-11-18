"""Restricted Phase 0 CLI for media grants, access decisions, and audit review."""

from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal
from pathlib import Path
from uuid import UUID

from billiards_ai.shared.models import AccessAction, AssetContext, MaterialUsePurpose

from .access_identity import FileReviewerIdentityProvider
from .access_repository import AccessError, JsonAccessRepository
from .access_service import AccessDeniedError, MediaAccessService
from .repository import JsonGateRepository
from .retention_repository import JsonRetentionRepository, RetentionError
from .retention_service import MediaRetentionService


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="billiards-media-access")
    parser.add_argument("--access-store", type=Path, required=True)
    parser.add_argument("--retention-store", type=Path, required=True)
    parser.add_argument("--gate-store", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)

    grant = sub.add_parser("grant")
    grant.add_argument("--session-id", type=UUID, required=True)
    grant.add_argument("--reason", required=True)

    revoke = sub.add_parser("revoke")
    revoke.add_argument("--grant-id", type=UUID, required=True)
    revoke.add_argument("--reason", required=True)

    check = sub.add_parser("check")
    check.add_argument("--session-id", type=UUID, required=True)

    access = sub.add_parser("controlled-view")
    access.add_argument("--asset-id", type=UUID, required=True)
    access.add_argument("--gate-revision-id", type=UUID, required=True)
    access.add_argument("--asset-context", type=Path, required=True)
    access.add_argument("--reason", required=True)

    audit = sub.add_parser("audit")
    audit.add_argument("--session-id", type=UUID)

    approve = sub.add_parser("provider-approve")
    approve.add_argument("--provider-id", required=True)
    approve.add_argument("--endpoint-region", required=True)
    approve.add_argument("--purpose", type=MaterialUsePurpose, required=True)
    approve.add_argument("--cross-border", action="store_true")
    approve.add_argument("--cross-border-assessment", required=True)
    approve.add_argument("--max-cost-cny", type=Decimal, required=True)
    approve.add_argument("--evidence-sha256", required=True)

    provider_revoke = sub.add_parser("provider-revoke")
    provider_revoke.add_argument("--approval-id", type=UUID, required=True)
    provider_revoke.add_argument("--reason", required=True)
    return parser


def _write(payload: dict | list) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _service(args: argparse.Namespace) -> MediaAccessService:
    retention = MediaRetentionService(
        JsonRetentionRepository(args.retention_store),
        gate_repository=JsonGateRepository(args.gate_store),
    )
    return MediaAccessService(
        access_repository=JsonAccessRepository(args.access_store),
        retention_service=retention,
        identity_provider=FileReviewerIdentityProvider(),
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        service = _service(args)
        if args.command == "grant":
            item = service.grant_current_reviewer(session_id=args.session_id, reason=args.reason)
            _write(
                {
                    "grant_id": str(item.grant_id),
                    "session_id": str(item.session_id),
                    "status": "ACTIVE",
                }
            )
        elif args.command == "revoke":
            item = service.revoke_grant(grant_id=args.grant_id, reason=args.reason)
            _write(
                {
                    "grant_id": str(item.grant_id),
                    "session_id": str(item.session_id),
                    "status": "REVOKED",
                }
            )
        elif args.command == "check":
            item = service.check_current_reviewer_grant(session_id=args.session_id)
            _write(
                {
                    "grant_id": str(item.grant_id) if item else None,
                    "session_id": str(args.session_id),
                    "status": "ACTIVE" if item else "NOT_GRANTED",
                }
            )
        elif args.command == "controlled-view":
            context = AssetContext.model_validate_json(args.asset_context.read_bytes())
            service.authorize_and_record_access(
                asset_id=args.asset_id,
                gate_revision_id=args.gate_revision_id,
                asset_context=context,
                reason=args.reason,
                action=AccessAction.CONTROLLED_REVIEW_VIEW,
                consumer=lambda _locator: None,
            )
            _write({"asset_id": str(args.asset_id), "status": "AUTHORIZED"})
        elif args.command == "audit":
            _write(
                [
                    {
                        "event_id": str(event.event_id),
                        "sequence": event.sequence,
                        "actor_id": event.actor_id,
                        "actor_role": event.actor_role.value,
                        "occurred_at": event.occurred_at.isoformat(),
                        "session_id": str(event.session_id),
                        "action": event.action.value,
                        "reason": event.reason,
                        "result": event.result.value,
                        "reason_code": event.reason_code.value,
                        "security_event": event.security_event,
                    }
                    for event in service.list_audit_events(session_id=args.session_id)
                ]
            )
        elif args.command == "provider-approve":
            item = service.approve_third_party_processing(
                provider_id=args.provider_id,
                endpoint_region=args.endpoint_region,
                purpose=args.purpose,
                cross_border=args.cross_border,
                cross_border_assessment=args.cross_border_assessment,
                max_cost_cny=args.max_cost_cny,
                approval_evidence_sha256=args.evidence_sha256,
            )
            _write(
                {
                    "approval_id": str(item.approval_id),
                    "provider_id": item.provider_id,
                    "status": "ACTIVE",
                }
            )
        else:
            item = service.revoke_third_party_approval(
                approval_id=args.approval_id, reason=args.reason
            )
            _write(
                {
                    "approval_id": str(item.approval_id),
                    "provider_id": item.provider_id,
                    "status": "REVOKED",
                }
            )
        return 0
    except (AccessError, AccessDeniedError, RetentionError, OSError, ValueError) as exc:
        print(
            json.dumps(
                {"error": type(exc).__name__, "message": "operation could not be completed"}
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
