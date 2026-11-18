"""Create durable growth evidence, snapshot, metric, outcome, and outbox storage."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "20260816_03"
down_revision = "20260810_02"
branch_labels = None
depends_on = None

UUID = postgresql.UUID(as_uuid=True)
JSONB = postgresql.JSONB(astext_type=sa.Text())
SHA = r"{column} ~ '^[0-9a-f]{{64}}$'"


def _uuid4(table: str, column: str, *, nullable: bool = False) -> None:
    expression = (
        f"substring({column}::text from 15 for 1) = '4' "
        f"AND substring({column}::text from 20 for 1) IN ('8','9','a','b')"
    )
    if nullable:
        expression = f"{column} IS NULL OR ({expression})"
    op.create_check_constraint(f"ck_{table}_{column}_uuid4", table, expression)


def _digest(table: str, *columns: str) -> None:
    for column in columns:
        op.create_check_constraint(f"ck_{table}_{column}", table, SHA.format(column=column))


def upgrade() -> None:
    op.create_table(
        "growth_evidence_batches",
        sa.Column("schema_version", sa.Text(), nullable=False),
        sa.Column("batch_id", UUID, nullable=False),
        sa.Column("user_id", UUID, nullable=False),
        sa.Column("session_id", UUID, nullable=False),
        sa.Column("report_id", UUID, nullable=False),
        sa.Column("report_revision", sa.Integer(), nullable=False),
        sa.Column("review_revision", sa.Integer(), nullable=False),
        sa.Column("approved_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("content_sha256", sa.Text(), nullable=False),
        sa.Column("payload", JSONB, nullable=False),
        sa.Column("payload_sha256", sa.Text(), nullable=False),
        sa.UniqueConstraint(
            "schema_version",
            "session_id",
            "review_revision",
            name="uq_growth_batch_business_identity",
        ),
        sa.PrimaryKeyConstraint("schema_version", "batch_id", name="pk_growth_evidence_batches"),
        sa.CheckConstraint("schema_version IN ('1.0','2.0')", name="ck_growth_batch_schema"),
        sa.CheckConstraint(
            "report_revision > 0 AND review_revision > 0",
            name="ck_growth_batch_revisions",
        ),
    )
    for column in ("batch_id", "user_id", "session_id", "report_id"):
        _uuid4("growth_evidence_batches", column)
    _digest("growth_evidence_batches", "content_sha256", "payload_sha256")
    op.create_index(
        "ix_growth_evidence_batches_user_approved",
        "growth_evidence_batches",
        ["user_id", "schema_version", "approved_at", "batch_id"],
    )

    op.create_table(
        "growth_evidence_inbox",
        sa.Column("receipt_id", UUID, primary_key=True),
        sa.Column("schema_version", sa.Text(), nullable=False),
        sa.Column("batch_id", UUID, nullable=False),
        sa.Column("user_id", UUID, nullable=False),
        sa.Column("session_id", UUID, nullable=False),
        sa.Column("report_id", UUID, nullable=False),
        sa.Column("review_revision", sa.Integer(), nullable=False),
        sa.Column("content_sha256", sa.Text(), nullable=False),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("payload", JSONB, nullable=False),
        sa.Column("payload_sha256", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["schema_version", "batch_id"],
            ["growth_evidence_batches.schema_version", "growth_evidence_batches.batch_id"],
            name="fk_growth_inbox_batch_namespace",
            ondelete="RESTRICT",
        ),
        sa.UniqueConstraint("schema_version", "batch_id", name="uq_growth_inbox_batch_namespace"),
        sa.CheckConstraint("schema_version IN ('1.0','2.0')", name="ck_growth_inbox_schema"),
        sa.CheckConstraint("review_revision > 0", name="ck_growth_inbox_revision"),
    )
    for column in ("receipt_id", "batch_id", "user_id", "session_id", "report_id"):
        _uuid4("growth_evidence_inbox", column)
    _digest("growth_evidence_inbox", "content_sha256", "payload_sha256")
    op.create_index(
        "ix_growth_evidence_inbox_batch",
        "growth_evidence_inbox",
        ["schema_version", "batch_id"],
    )

    op.create_table(
        "growth_evidence_sources",
        sa.Column("schema_version", sa.Text(), nullable=False),
        sa.Column("batch_id", UUID, nullable=False),
        sa.Column("payload", JSONB, nullable=False),
        sa.Column("payload_sha256", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["schema_version", "batch_id"],
            ["growth_evidence_batches.schema_version", "growth_evidence_batches.batch_id"],
            name="fk_growth_source_batch_namespace",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("schema_version", "batch_id", name="pk_growth_evidence_sources"),
        sa.CheckConstraint("schema_version IN ('1.0','2.0')", name="ck_growth_source_schema"),
    )
    _uuid4("growth_evidence_sources", "batch_id")
    _digest("growth_evidence_sources", "payload_sha256")

    op.create_table(
        "growth_metric_authorities",
        sa.Column("schema_version", sa.Text(), nullable=False),
        sa.Column("batch_id", UUID, nullable=False),
        sa.Column("authority_schema_version", sa.Text(), nullable=False),
        sa.Column("approval_kind", sa.Text(), nullable=False),
        sa.Column("approval_policy_version", sa.Text(), nullable=False),
        sa.Column("authority_status", sa.Text(), nullable=False),
        sa.Column("review_revision", sa.Integer(), nullable=False),
        sa.Column("payload", JSONB, nullable=False),
        sa.Column("payload_sha256", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["schema_version", "batch_id"],
            ["growth_evidence_batches.schema_version", "growth_evidence_batches.batch_id"],
            name="fk_growth_authority_batch_namespace",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("schema_version", "batch_id", name="pk_growth_metric_authorities"),
        sa.CheckConstraint("schema_version = '2.0'", name="ck_growth_authority_batch_schema"),
        sa.CheckConstraint(
            "authority_schema_version = 'growth-metric-authority.v1'",
            name="ck_growth_authority_schema",
        ),
        sa.CheckConstraint("authority_status = 'APPROVED'", name="ck_growth_authority_status"),
        sa.CheckConstraint("review_revision > 0", name="ck_growth_authority_revision"),
        sa.CheckConstraint(
            "length(approval_policy_version) > 0", name="ck_growth_authority_policy"
        ),
    )
    _uuid4("growth_metric_authorities", "batch_id")
    _digest("growth_metric_authorities", "payload_sha256")
    op.create_index(
        "ix_growth_metric_authorities_query",
        "growth_metric_authorities",
        ["approval_kind", "approval_policy_version", "authority_status", "review_revision"],
    )

    op.create_table(
        "growth_snapshots",
        sa.Column("snapshot_id", UUID, primary_key=True),
        sa.Column("user_id", UUID, nullable=False),
        sa.Column("schema_version", sa.Text(), nullable=False),
        sa.Column("snapshot_policy_version", sa.Text(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("previous_version", sa.Integer()),
        sa.Column("previous_snapshot_id", UUID),
        sa.Column("content_sha256", sa.Text(), nullable=False),
        sa.Column("evidence_sha256", sa.Text(), nullable=False),
        sa.Column("source_bundle_sha256", sa.Text(), nullable=False),
        sa.Column("idempotency_key", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("payload", JSONB, nullable=False),
        sa.Column("payload_sha256", sa.Text(), nullable=False),
        sa.UniqueConstraint(
            "user_id",
            "snapshot_policy_version",
            "version",
            name="uq_growth_snapshot_version",
        ),
        sa.UniqueConstraint(
            "user_id",
            "snapshot_policy_version",
            "content_sha256",
            name="uq_growth_snapshot_content",
        ),
        sa.UniqueConstraint(
            "user_id",
            "snapshot_policy_version",
            "version",
            "snapshot_id",
            name="uq_growth_snapshot_head_target",
        ),
        sa.UniqueConstraint(
            "user_id",
            "snapshot_policy_version",
            "snapshot_id",
            "content_sha256",
            name="uq_growth_snapshot_receipt_target",
        ),
        sa.ForeignKeyConstraint(
            [
                "user_id",
                "snapshot_policy_version",
                "previous_version",
                "previous_snapshot_id",
            ],
            [
                "growth_snapshots.user_id",
                "growth_snapshots.snapshot_policy_version",
                "growth_snapshots.version",
                "growth_snapshots.snapshot_id",
            ],
            name="fk_growth_snapshot_adjacent_predecessor",
            ondelete="RESTRICT",
        ),
        sa.CheckConstraint("schema_version IN ('2.0','3.0')", name="ck_growth_snapshot_schema"),
        sa.CheckConstraint("version > 0", name="ck_growth_snapshot_version"),
        sa.CheckConstraint(
            "(version = 1 AND previous_version IS NULL AND previous_snapshot_id IS NULL) OR "
            "(version > 1 AND previous_version = version - 1 "
            "AND previous_snapshot_id IS NOT NULL)",
            name="ck_growth_snapshot_predecessor_shape",
        ),
        sa.CheckConstraint("length(idempotency_key) > 0", name="ck_growth_snapshot_key"),
    )
    for column in ("snapshot_id", "user_id"):
        _uuid4("growth_snapshots", column)
    _uuid4("growth_snapshots", "previous_snapshot_id", nullable=True)
    _digest(
        "growth_snapshots",
        "content_sha256",
        "evidence_sha256",
        "source_bundle_sha256",
        "payload_sha256",
    )
    op.create_index(
        "ix_growth_snapshots_user_policy",
        "growth_snapshots",
        ["user_id", "snapshot_policy_version", "version"],
    )
    op.create_index("ix_growth_snapshots_previous", "growth_snapshots", ["previous_snapshot_id"])

    op.create_table(
        "growth_snapshot_idempotency",
        sa.Column("user_id", UUID, primary_key=True),
        sa.Column("snapshot_policy_version", sa.Text(), primary_key=True),
        sa.Column("idempotency_key", sa.Text(), primary_key=True),
        sa.Column("snapshot_id", UUID, nullable=False),
        sa.Column("content_sha256", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["user_id", "snapshot_policy_version", "snapshot_id", "content_sha256"],
            [
                "growth_snapshots.user_id",
                "growth_snapshots.snapshot_policy_version",
                "growth_snapshots.snapshot_id",
                "growth_snapshots.content_sha256",
            ],
            name="fk_growth_snapshot_receipt_exact_target",
            ondelete="RESTRICT",
        ),
        sa.CheckConstraint("length(idempotency_key) > 0", name="ck_growth_snapshot_receipt_key"),
    )
    _uuid4("growth_snapshot_idempotency", "user_id")
    _uuid4("growth_snapshot_idempotency", "snapshot_id")
    _digest("growth_snapshot_idempotency", "content_sha256")
    op.create_index(
        "ix_growth_snapshot_idempotency_snapshot",
        "growth_snapshot_idempotency",
        ["snapshot_id"],
    )

    op.create_table(
        "growth_snapshot_heads",
        sa.Column("user_id", UUID, primary_key=True),
        sa.Column("snapshot_policy_version", sa.Text(), primary_key=True),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("snapshot_id", UUID, nullable=False),
        sa.ForeignKeyConstraint(
            ["user_id", "snapshot_policy_version", "version", "snapshot_id"],
            [
                "growth_snapshots.user_id",
                "growth_snapshots.snapshot_policy_version",
                "growth_snapshots.version",
                "growth_snapshots.snapshot_id",
            ],
            name="fk_growth_head_exact_snapshot",
            ondelete="RESTRICT",
        ),
        sa.CheckConstraint("version > 0", name="ck_growth_head_version"),
    )
    _uuid4("growth_snapshot_heads", "user_id")
    _uuid4("growth_snapshot_heads", "snapshot_id")
    op.create_index("ix_growth_snapshot_heads_target", "growth_snapshot_heads", ["snapshot_id"])

    op.create_table(
        "growth_metrics",
        sa.Column("metric_id", UUID, primary_key=True),
        sa.Column(
            "snapshot_id",
            UUID,
            sa.ForeignKey("growth_snapshots.snapshot_id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("metric_code", sa.Text(), nullable=False),
        sa.Column("support_status", sa.Text(), nullable=False),
        sa.Column("reliability", sa.Text(), nullable=False),
        sa.Column("payload", JSONB, nullable=False),
        sa.Column("payload_sha256", sa.Text(), nullable=False),
        sa.UniqueConstraint("snapshot_id", "ordinal", name="uq_growth_metric_ordinal"),
        sa.CheckConstraint("ordinal >= 0", name="ck_growth_metric_ordinal"),
    )
    _uuid4("growth_metrics", "metric_id")
    _uuid4("growth_metrics", "snapshot_id")
    _digest("growth_metrics", "payload_sha256")
    op.create_index("ix_growth_metrics_snapshot", "growth_metrics", ["snapshot_id"])
    op.create_index("ix_growth_metrics_query", "growth_metrics", ["metric_code", "support_status"])

    op.create_table(
        "growth_plan_outcomes",
        sa.Column("history_id", UUID, primary_key=True),
        sa.Column("receipt_id", UUID, nullable=False, unique=True),
        sa.Column("outcome_id", UUID, nullable=False, unique=True),
        sa.Column("event_id", UUID, nullable=False, unique=True),
        sa.Column("user_id", UUID, nullable=False),
        sa.Column("plan_id", UUID, nullable=False),
        sa.Column("session_id", UUID, nullable=False),
        sa.Column("idempotency_key", sa.Text(), nullable=False, unique=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("outcome_payload", JSONB, nullable=False),
        sa.Column("outcome_sha256", sa.Text(), nullable=False),
        sa.Column("event_payload", JSONB, nullable=False),
        sa.Column("event_sha256", sa.Text(), nullable=False),
        sa.UniqueConstraint("plan_id", "session_id", name="uq_growth_outcome_business"),
        sa.CheckConstraint("length(idempotency_key) > 0", name="ck_growth_outcome_key"),
        sa.CheckConstraint("started_at <= completed_at", name="ck_growth_outcome_time_order"),
    )
    for column in (
        "history_id",
        "receipt_id",
        "outcome_id",
        "event_id",
        "user_id",
        "plan_id",
        "session_id",
    ):
        _uuid4("growth_plan_outcomes", column)
    _digest("growth_plan_outcomes", "outcome_sha256", "event_sha256")
    op.create_index(
        "ix_growth_plan_outcomes_user_time",
        "growth_plan_outcomes",
        ["user_id", "started_at", "completed_at"],
    )

    op.create_table(
        "growth_outcome_inbox",
        sa.Column("receipt_id", UUID, primary_key=True),
        sa.Column(
            "history_id",
            UUID,
            sa.ForeignKey("growth_plan_outcomes.history_id", ondelete="RESTRICT"),
            nullable=False,
            unique=True,
        ),
        sa.Column("event_id", UUID, nullable=False, unique=True),
        sa.Column("outcome_id", UUID, nullable=False, unique=True),
        sa.Column("user_id", UUID, nullable=False),
        sa.Column("idempotency_key", sa.Text(), nullable=False, unique=True),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("payload", JSONB, nullable=False),
        sa.Column("payload_sha256", sa.Text(), nullable=False),
    )
    for column in (
        "receipt_id",
        "history_id",
        "event_id",
        "outcome_id",
        "user_id",
    ):
        _uuid4("growth_outcome_inbox", column)
    _digest("growth_outcome_inbox", "payload_sha256")
    op.create_index("ix_growth_outcome_inbox_history", "growth_outcome_inbox", ["history_id"])

    op.create_table(
        "core_intelligence_outbox",
        sa.Column("event_id", UUID, primary_key=True),
        sa.Column("owner_user_id", UUID, nullable=False),
        sa.Column("aggregate_namespace", sa.Text(), nullable=False),
        sa.Column("schema_version", sa.Text(), nullable=False),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column("aggregate_type", sa.Text(), nullable=False),
        sa.Column("aggregate_id", UUID, nullable=False),
        sa.Column("aggregate_version", sa.Integer(), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("causation_id", UUID),
        sa.Column("correlation_id", UUID, nullable=False),
        sa.Column("idempotency_key", sa.Text(), nullable=False),
        sa.Column("payload", JSONB, nullable=False),
        sa.Column("payload_sha256", sa.Text(), nullable=False),
        sa.Column("event_sha256", sa.Text(), nullable=False),
        sa.Column("available_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True)),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("claimed_by", sa.Text()),
        sa.Column("claimed_at", sa.DateTime(timezone=True)),
        sa.Column("claim_until", sa.DateTime(timezone=True)),
        sa.Column("claim_token", UUID),
        sa.Column("last_error", sa.Text()),
        sa.UniqueConstraint(
            "owner_user_id",
            "aggregate_namespace",
            "event_type",
            "idempotency_key",
            name="uq_core_outbox_idempotency",
        ),
        sa.UniqueConstraint(
            "owner_user_id",
            "aggregate_namespace",
            "aggregate_type",
            "aggregate_id",
            "aggregate_version",
            "event_type",
            name="uq_core_outbox_aggregate_version",
        ),
        sa.CheckConstraint("aggregate_version > 0", name="ck_core_outbox_version"),
        sa.CheckConstraint("length(aggregate_namespace) > 0", name="ck_core_outbox_namespace"),
        sa.CheckConstraint("attempt_count >= 0", name="ck_core_outbox_attempt_count"),
        sa.CheckConstraint(
            "(claimed_by IS NULL AND claimed_at IS NULL AND claim_until IS NULL "
            "AND claim_token IS NULL) OR (claimed_by IS NOT NULL AND claimed_at IS NOT NULL "
            "AND claim_until IS NOT NULL AND claim_token IS NOT NULL)",
            name="ck_core_outbox_claim_shape",
        ),
    )
    for column in ("event_id", "owner_user_id", "aggregate_id", "correlation_id"):
        _uuid4("core_intelligence_outbox", column)
    _uuid4("core_intelligence_outbox", "causation_id", nullable=True)
    _uuid4("core_intelligence_outbox", "claim_token", nullable=True)
    _digest("core_intelligence_outbox", "payload_sha256", "event_sha256")
    op.create_index(
        "ix_core_intelligence_outbox_order",
        "core_intelligence_outbox",
        ["occurred_at", "event_id"],
    )
    op.create_index(
        "ix_core_intelligence_outbox_pending",
        "core_intelligence_outbox",
        ["available_at", "occurred_at", "event_id"],
        postgresql_where=sa.text("published_at IS NULL"),
    )


def downgrade() -> None:
    for table in (
        "core_intelligence_outbox",
        "growth_outcome_inbox",
        "growth_plan_outcomes",
        "growth_metrics",
        "growth_snapshot_heads",
        "growth_snapshot_idempotency",
        "growth_snapshots",
        "growth_metric_authorities",
        "growth_evidence_sources",
        "growth_evidence_inbox",
        "growth_evidence_batches",
    ):
        op.drop_table(table)
