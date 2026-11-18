"""Create quality-gate authority, idempotency, and outbox tables."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "20260810_01"
down_revision = None
branch_labels = None
depends_on = None

UUID = postgresql.UUID(as_uuid=True)
JSONB = postgresql.JSONB(astext_type=sa.Text())
SHA_CHECK = "{column} ~ '^[0-9a-f]{{64}}$'"


def _payload_columns() -> list[sa.Column]:
    return [
        sa.Column("payload", JSONB, nullable=False),
        sa.Column("payload_sha256", sa.Text(), nullable=False),
    ]


def _digest_check(table: str) -> None:
    op.create_check_constraint(
        f"ck_{table}_payload_sha256", table, SHA_CHECK.format(column="payload_sha256")
    )


def _uuid4_check(table: str, column: str, *, nullable: bool = False) -> None:
    expression = (
        f"substring({column}::text from 15 for 1) = '4' "
        f"AND substring({column}::text from 20 for 1) IN ('8','9','a','b')"
    )
    if nullable:
        expression = f"{column} IS NULL OR ({expression})"
    op.create_check_constraint(f"ck_{table}_{column}_uuid4", table, expression)


def _install_report_parent_guard() -> None:
    op.execute(
        """
        CREATE FUNCTION enforce_review_report_parent_lineage()
        RETURNS trigger AS $$
        DECLARE
            parent_session_id uuid;
            parent_revision integer;
        BEGIN
            IF NEW.parent_report_id IS NULL THEN
                RETURN NEW;
            END IF;

            SELECT session_id, revision
              INTO STRICT parent_session_id, parent_revision
              FROM review_reports
             WHERE report_id = NEW.parent_report_id;

            IF NEW.session_id <> parent_session_id OR NEW.revision <> parent_revision + 1 THEN
                RAISE EXCEPTION 'invalid review report parent lineage'
                    USING ERRCODE = '23514';
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_review_reports_parent_lineage
        BEFORE INSERT OR UPDATE OF parent_report_id, session_id, revision
        ON review_reports
        FOR EACH ROW
        EXECUTE FUNCTION enforce_review_report_parent_lineage();
        """
    )


def _install_terminal_authority_guard() -> None:
    op.execute(
        """
        CREATE FUNCTION enforce_review_task_terminal_authority()
        RETURNS trigger AS $$
        DECLARE
            checked_task_id uuid;
            task_status text;
            result_report_status text;
            source_count integer;
            decision_count integer;
            result_count integer;
            rejection_count integer;
            invalid_outcome_count integer;
            invalid_correction_count integer;
        BEGIN
            checked_task_id := COALESCE(NEW.task_id, OLD.task_id);
            SELECT status INTO task_status FROM review_tasks WHERE task_id = checked_task_id;
            IF task_status IS NULL OR task_status IN ('pending_review', 'in_review') THEN
                RETURN NULL;
            END IF;

            SELECT count(*) INTO source_count
              FROM review_shot_sources WHERE task_id = checked_task_id;
            SELECT count(*) INTO decision_count
              FROM review_shot_decisions WHERE task_id = checked_task_id;
            SELECT count(*) INTO result_count
              FROM review_shot_results WHERE task_id = checked_task_id;
            SELECT count(*) INTO rejection_count
              FROM review_rejections WHERE task_id = checked_task_id;

            IF task_status = 'rejected' THEN
                IF rejection_count <> 1 OR decision_count <> 0 OR result_count <> 0 THEN
                    RAISE EXCEPTION 'invalid rejected task authority coverage'
                        USING ERRCODE = '23514';
                END IF;
                RETURN NULL;
            END IF;

            SELECT rr.status INTO result_report_status
              FROM review_tasks t
              JOIN review_result_reports rr
                ON rr.report_id = t.result_report_id
               AND rr.revision = t.result_report_revision
             WHERE t.task_id = checked_task_id;

            IF source_count = 0 OR decision_count <> source_count OR result_count <> source_count
               OR rejection_count <> 0
               OR result_report_status <> upper(task_status) THEN
                RAISE EXCEPTION 'invalid terminal task authority coverage'
                    USING ERRCODE = '23514';
            END IF;

            SELECT count(*) INTO invalid_outcome_count
              FROM review_shot_decisions d
              JOIN review_shot_results r USING (task_id, shot_id)
             WHERE d.task_id = checked_task_id
               AND (d.outcome <> r.review_status OR d.outcome = 'rejected');

            SELECT count(*) INTO invalid_correction_count
              FROM review_shot_decisions d
             WHERE d.task_id = checked_task_id
               AND ((d.outcome = 'corrected' AND NOT EXISTS (
                        SELECT 1 FROM review_corrections c
                         WHERE c.task_id = d.task_id AND c.shot_id = d.shot_id
                    ))
                    OR (d.outcome <> 'corrected' AND EXISTS (
                        SELECT 1 FROM review_corrections c
                         WHERE c.task_id = d.task_id AND c.shot_id = d.shot_id
                    )));

            IF invalid_outcome_count <> 0 OR invalid_correction_count <> 0
               OR (task_status = 'approved' AND EXISTS (
                    SELECT 1 FROM review_shot_decisions
                     WHERE task_id = checked_task_id AND outcome <> 'approved'
               ))
               OR (task_status = 'corrected' AND NOT EXISTS (
                    SELECT 1 FROM review_shot_decisions
                     WHERE task_id = checked_task_id AND outcome = 'corrected'
               )) THEN
                RAISE EXCEPTION 'invalid terminal task outcome or correction authority'
                    USING ERRCODE = '23514';
            END IF;
            RETURN NULL;
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    for trigger_name, table, operations in (
        ("trg_review_tasks_terminal_authority", "review_tasks", "INSERT OR UPDATE"),
        (
            "trg_review_decisions_terminal_authority",
            "review_shot_decisions",
            "INSERT OR UPDATE OR DELETE",
        ),
        (
            "trg_review_corrections_terminal_authority",
            "review_corrections",
            "INSERT OR UPDATE OR DELETE",
        ),
        (
            "trg_review_results_terminal_authority",
            "review_shot_results",
            "INSERT OR UPDATE OR DELETE",
        ),
        (
            "trg_review_rejections_terminal_authority",
            "review_rejections",
            "INSERT OR UPDATE OR DELETE",
        ),
    ):
        op.execute(
            f"CREATE CONSTRAINT TRIGGER {trigger_name} "
            f"AFTER {operations} ON {table} "
            "DEFERRABLE INITIALLY DEFERRED FOR EACH ROW "
            "EXECUTE FUNCTION enforce_review_task_terminal_authority()"
        )


def upgrade() -> None:
    op.create_table(
        "review_reports",
        sa.Column("report_id", UUID, primary_key=True),
        sa.Column("session_id", UUID, nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column(
            "parent_report_id", UUID, sa.ForeignKey("review_reports.report_id", ondelete="RESTRICT")
        ),
        sa.Column("schema_version", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("generated_at", sa.DateTime(timezone=True), nullable=False),
        *_payload_columns(),
        sa.CheckConstraint("revision > 0", name="ck_review_reports_revision"),
        sa.CheckConstraint(
            "(parent_report_id IS NULL AND revision = 1) OR "
            "(parent_report_id IS NOT NULL AND revision > 1)",
            name="ck_review_reports_parent_shape",
        ),
        sa.UniqueConstraint("session_id", "revision", name="uq_review_reports_session_revision"),
        sa.UniqueConstraint("report_id", "revision", name="uq_review_reports_id_revision"),
        sa.UniqueConstraint("parent_report_id", name="uq_review_reports_parent"),
    )
    _digest_check("review_reports")
    op.create_index("ix_review_reports_session", "review_reports", ["session_id", "revision"])
    op.create_index("ix_review_reports_parent", "review_reports", ["parent_report_id"])
    _uuid4_check("review_reports", "report_id")
    _uuid4_check("review_reports", "session_id")
    _uuid4_check("review_reports", "parent_report_id", nullable=True)
    _install_report_parent_guard()

    op.create_table(
        "review_result_reports",
        sa.Column("report_id", UUID, primary_key=True),
        sa.Column("revision", sa.Integer(), primary_key=True),
        sa.Column("source_report_id", UUID, nullable=False),
        sa.Column("source_report_revision", sa.Integer(), nullable=False),
        sa.Column("parent_report_id", UUID),
        sa.Column("session_id", UUID, nullable=False),
        sa.Column("schema_version", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("generated_at", sa.DateTime(timezone=True), nullable=False),
        *_payload_columns(),
        sa.ForeignKeyConstraint(
            ["source_report_id", "source_report_revision"],
            ["review_reports.report_id", "review_reports.revision"],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["parent_report_id"], ["review_reports.report_id"], ondelete="RESTRICT"
        ),
        sa.UniqueConstraint(
            "source_report_id",
            "source_report_revision",
            name="uq_review_result_reports_source",
        ),
        sa.CheckConstraint(
            "status IN ('APPROVED','CORRECTED')", name="ck_review_result_reports_status"
        ),
        sa.CheckConstraint(
            "(status = 'APPROVED' AND report_id = source_report_id "
            "AND revision = source_report_revision AND parent_report_id IS NULL) OR "
            "(status = 'CORRECTED' AND report_id <> source_report_id "
            "AND revision = source_report_revision + 1 "
            "AND parent_report_id = source_report_id)",
            name="ck_review_result_reports_lineage",
        ),
    )
    _digest_check("review_result_reports")
    op.create_index(
        "ix_review_result_reports_source",
        "review_result_reports",
        ["source_report_id", "source_report_revision"],
    )
    op.create_index(
        "ix_review_result_reports_parent", "review_result_reports", ["parent_report_id"]
    )
    _uuid4_check("review_result_reports", "report_id")
    _uuid4_check("review_result_reports", "source_report_id")
    _uuid4_check("review_result_reports", "parent_report_id", nullable=True)
    _uuid4_check("review_result_reports", "session_id")

    op.create_table(
        "review_tasks",
        sa.Column("task_id", UUID, primary_key=True),
        sa.Column("session_id", UUID, nullable=False),
        sa.Column("source_report_id", UUID, nullable=False),
        sa.Column("source_report_revision", sa.Integer(), nullable=False),
        sa.Column("result_report_id", UUID),
        sa.Column("result_report_revision", sa.Integer()),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("claimed_principal_id", sa.Text()),
        sa.Column("claimed_credential_id", sa.Text()),
        sa.Column("claimed_at", sa.DateTime(timezone=True)),
        sa.Column("lock_version", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.ForeignKeyConstraint(
            ["source_report_id", "source_report_revision"],
            ["review_reports.report_id", "review_reports.revision"],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["result_report_id", "result_report_revision"],
            ["review_result_reports.report_id", "review_result_reports.revision"],
            ondelete="RESTRICT",
        ),
        sa.UniqueConstraint(
            "source_report_id", "source_report_revision", name="uq_review_tasks_source"
        ),
        sa.UniqueConstraint("task_id", "session_id", name="uq_review_tasks_task_session"),
        sa.CheckConstraint(
            "status IN ('pending_review','in_review','approved','corrected','rejected')",
            name="ck_review_tasks_status",
        ),
        sa.CheckConstraint(
            "(claimed_principal_id IS NULL AND claimed_credential_id IS NULL "
            "AND claimed_at IS NULL) "
            "OR (claimed_principal_id IS NOT NULL AND claimed_credential_id IS NOT NULL "
            "AND claimed_at IS NOT NULL)",
            name="ck_review_tasks_claim_all_or_none",
        ),
        sa.CheckConstraint(
            "(status = 'pending_review' AND claimed_principal_id IS NULL) OR "
            "(status IN ('in_review','approved','corrected','rejected') "
            "AND claimed_principal_id IS NOT NULL)",
            name="ck_review_tasks_status_claim",
        ),
        sa.CheckConstraint(
            "claimed_principal_id IS NULL OR "
            "(btrim(claimed_principal_id) <> '' AND btrim(claimed_credential_id) <> '')",
            name="ck_review_tasks_claim_nonblank",
        ),
        sa.CheckConstraint(
            "(status IN ('pending_review','in_review') AND completed_at IS NULL "
            "AND result_report_id IS NULL AND result_report_revision IS NULL) OR "
            "(status IN ('approved','corrected') AND completed_at IS NOT NULL "
            "AND result_report_id IS NOT NULL AND result_report_revision IS NOT NULL) OR "
            "(status = 'rejected' AND completed_at IS NOT NULL "
            "AND result_report_id IS NULL AND result_report_revision IS NULL)",
            name="ck_review_tasks_terminal_shape",
        ),
        sa.CheckConstraint("lock_version >= 0", name="ck_review_tasks_lock_version"),
    )
    op.create_index(
        "ix_review_tasks_source_report",
        "review_tasks",
        ["source_report_id", "source_report_revision"],
    )
    op.create_index(
        "ix_review_tasks_result_report",
        "review_tasks",
        ["result_report_id", "result_report_revision"],
    )
    op.create_index("ix_review_tasks_session_status", "review_tasks", ["session_id", "status"])
    op.create_index(
        "uq_review_tasks_active_session",
        "review_tasks",
        ["session_id"],
        unique=True,
        postgresql_where=sa.text("status IN ('pending_review','in_review')"),
    )
    _uuid4_check("review_tasks", "task_id")
    _uuid4_check("review_tasks", "session_id")
    _uuid4_check("review_tasks", "source_report_id")
    _uuid4_check("review_tasks", "result_report_id", nullable=True)

    op.create_table(
        "review_shot_sources",
        sa.Column(
            "task_id",
            UUID,
            primary_key=True,
        ),
        sa.Column("shot_id", UUID, primary_key=True),
        sa.Column("session_id", UUID, nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("source_envelope_id", UUID, nullable=False),
        *_payload_columns(),
        sa.CheckConstraint("sequence > 0", name="ck_review_shot_sources_sequence"),
        sa.UniqueConstraint("task_id", "sequence", name="uq_review_shot_sources_task_sequence"),
        sa.UniqueConstraint("source_envelope_id", name="uq_review_shot_sources_envelope"),
        sa.UniqueConstraint(
            "task_id", "shot_id", "session_id", name="uq_review_shot_sources_identity"
        ),
        sa.ForeignKeyConstraint(
            ["task_id", "session_id"],
            ["review_tasks.task_id", "review_tasks.session_id"],
            ondelete="CASCADE",
        ),
    )
    _digest_check("review_shot_sources")
    op.create_index("ix_review_shot_sources_task", "review_shot_sources", ["task_id"])
    op.create_index(
        "ix_review_shot_sources_task_session",
        "review_shot_sources",
        ["task_id", "session_id"],
    )
    _uuid4_check("review_shot_sources", "task_id")
    _uuid4_check("review_shot_sources", "shot_id")
    _uuid4_check("review_shot_sources", "session_id")
    _uuid4_check("review_shot_sources", "source_envelope_id")

    op.create_table(
        "review_shot_decisions",
        sa.Column("task_id", UUID, primary_key=True),
        sa.Column("shot_id", UUID, primary_key=True),
        sa.Column("outcome", sa.Text(), nullable=False),
        sa.Column("reviewer_principal_id", sa.Text(), nullable=False),
        sa.Column("reviewer_credential_id", sa.Text(), nullable=False),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("trajectory_reviewed", sa.Boolean(), nullable=False),
        *_payload_columns(),
        sa.ForeignKeyConstraint(
            ["task_id", "shot_id"],
            ["review_shot_sources.task_id", "review_shot_sources.shot_id"],
            ondelete="RESTRICT",
        ),
        sa.CheckConstraint(
            "outcome IN ('approved','corrected','rejected')",
            name="ck_review_shot_decisions_outcome",
        ),
    )
    _digest_check("review_shot_decisions")
    op.create_index("ix_review_shot_decisions_task", "review_shot_decisions", ["task_id"])
    _uuid4_check("review_shot_decisions", "task_id")
    _uuid4_check("review_shot_decisions", "shot_id")

    op.create_table(
        "review_corrections",
        sa.Column("task_id", UUID, primary_key=True),
        sa.Column("shot_id", UUID, primary_key=True),
        sa.Column("field", sa.Text(), primary_key=True),
        sa.Column("ai_value", JSONB, nullable=False),
        sa.Column("correct_value", JSONB, nullable=False),
        sa.Column("note", sa.Text(), nullable=False),
        sa.Column("reviewer_principal_id", sa.Text(), nullable=False),
        sa.Column("reviewer_credential_id", sa.Text(), nullable=False),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=False),
        *_payload_columns(),
        sa.ForeignKeyConstraint(
            ["task_id", "shot_id"],
            ["review_shot_decisions.task_id", "review_shot_decisions.shot_id"],
            ondelete="RESTRICT",
        ),
    )
    _digest_check("review_corrections")
    op.create_index("ix_review_corrections_task", "review_corrections", ["task_id"])
    _uuid4_check("review_corrections", "task_id")
    _uuid4_check("review_corrections", "shot_id")

    op.create_table(
        "review_shot_results",
        sa.Column("task_id", UUID, primary_key=True),
        sa.Column("shot_id", UUID, primary_key=True),
        sa.Column("result_report_id", UUID, nullable=False),
        sa.Column("result_report_revision", sa.Integer(), nullable=False),
        sa.Column("session_id", UUID, nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("review_status", sa.Text(), nullable=False),
        *_payload_columns(),
        sa.ForeignKeyConstraint(
            ["task_id", "shot_id", "session_id"],
            [
                "review_shot_sources.task_id",
                "review_shot_sources.shot_id",
                "review_shot_sources.session_id",
            ],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["result_report_id", "result_report_revision"],
            ["review_result_reports.report_id", "review_result_reports.revision"],
            ondelete="RESTRICT",
        ),
        sa.CheckConstraint("sequence > 0", name="ck_review_shot_results_sequence"),
        sa.CheckConstraint(
            "review_status IN ('approved','corrected','rejected')",
            name="ck_review_shot_results_status",
        ),
        sa.UniqueConstraint("task_id", "sequence", name="uq_review_shot_results_task_sequence"),
        sa.UniqueConstraint(
            "task_id",
            "shot_id",
            "result_report_id",
            "result_report_revision",
            "review_status",
            name="uq_review_shot_results_batch_binding",
        ),
    )
    _digest_check("review_shot_results")
    op.create_index(
        "ix_review_shot_results_report",
        "review_shot_results",
        ["result_report_id", "result_report_revision"],
    )
    op.create_index(
        "ix_review_shot_results_source_identity",
        "review_shot_results",
        ["task_id", "shot_id", "session_id"],
    )
    _uuid4_check("review_shot_results", "task_id")
    _uuid4_check("review_shot_results", "shot_id")
    _uuid4_check("review_shot_results", "result_report_id")
    _uuid4_check("review_shot_results", "session_id")

    op.create_table(
        "review_rejections",
        sa.Column("rejection_id", UUID, primary_key=True),
        sa.Column(
            "task_id",
            UUID,
            sa.ForeignKey("review_tasks.task_id", ondelete="RESTRICT"),
            nullable=False,
            unique=True,
        ),
        sa.Column("session_id", UUID, nullable=False),
        sa.Column("source_report_id", UUID, nullable=False),
        sa.Column("source_report_revision", sa.Integer(), nullable=False),
        sa.Column("reason_code", sa.Text(), nullable=False),
        sa.Column("reviewer_principal_id", sa.Text(), nullable=False),
        sa.Column("reviewer_credential_id", sa.Text(), nullable=False),
        sa.Column("rejected_at", sa.DateTime(timezone=True), nullable=False),
        *_payload_columns(),
        sa.ForeignKeyConstraint(
            ["source_report_id", "source_report_revision"],
            ["review_reports.report_id", "review_reports.revision"],
            ondelete="RESTRICT",
        ),
    )
    _digest_check("review_rejections")
    op.create_index(
        "ix_review_rejections_source",
        "review_rejections",
        ["source_report_id", "source_report_revision"],
    )
    _uuid4_check("review_rejections", "rejection_id")
    _uuid4_check("review_rejections", "task_id")
    _uuid4_check("review_rejections", "session_id")
    _uuid4_check("review_rejections", "source_report_id")

    op.create_table(
        "review_publication_lineage",
        sa.Column(
            "task_id",
            UUID,
            sa.ForeignKey("review_tasks.task_id", ondelete="RESTRICT"),
            primary_key=True,
        ),
        sa.Column("user_id", UUID, nullable=False),
        sa.Column("session_id", UUID, nullable=False),
        sa.Column("report_id", UUID, nullable=False),
        sa.Column("review_revision", sa.Integer(), nullable=False),
        sa.Column("approved_at", sa.DateTime(timezone=True), nullable=False),
        *_payload_columns(),
        sa.ForeignKeyConstraint(
            ["report_id", "review_revision"],
            ["review_result_reports.report_id", "review_result_reports.revision"],
            ondelete="RESTRICT",
        ),
    )
    _digest_check("review_publication_lineage")
    op.create_index(
        "ix_review_publication_lineage_report",
        "review_publication_lineage",
        ["report_id", "review_revision"],
    )
    _uuid4_check("review_publication_lineage", "task_id")
    _uuid4_check("review_publication_lineage", "user_id")
    _uuid4_check("review_publication_lineage", "session_id")
    _uuid4_check("review_publication_lineage", "report_id")

    op.create_table(
        "review_metric_authority",
        sa.Column("task_id", UUID, primary_key=True),
        sa.Column("shot_id", UUID, primary_key=True),
        sa.Column("review_revision", sa.Integer(), nullable=False),
        *_payload_columns(),
        sa.ForeignKeyConstraint(
            ["task_id", "shot_id"],
            ["review_shot_sources.task_id", "review_shot_sources.shot_id"],
            ondelete="RESTRICT",
        ),
    )
    _digest_check("review_metric_authority")
    op.create_index("ix_review_metric_authority_task", "review_metric_authority", ["task_id"])
    _uuid4_check("review_metric_authority", "task_id")
    _uuid4_check("review_metric_authority", "shot_id")

    op.create_table(
        "approved_evidence_batches",
        sa.Column("batch_id", UUID, primary_key=True),
        sa.Column("schema_version", sa.Text(), nullable=False),
        sa.Column("user_id", UUID, nullable=False),
        sa.Column("session_id", UUID, nullable=False),
        sa.Column("report_id", UUID, nullable=False),
        sa.Column("review_revision", sa.Integer(), nullable=False),
        sa.Column("approved_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("content_sha256", sa.Text(), nullable=False),
        *_payload_columns(),
        sa.ForeignKeyConstraint(
            ["report_id", "review_revision"],
            ["review_result_reports.report_id", "review_result_reports.revision"],
            ondelete="RESTRICT",
        ),
        sa.UniqueConstraint(
            "schema_version", "session_id", "review_revision", name="uq_approved_batches_identity"
        ),
        sa.UniqueConstraint(
            "batch_id", "report_id", "review_revision", name="uq_approved_batches_lineage"
        ),
        sa.CheckConstraint("schema_version = '2.0'", name="ck_approved_batches_v2_only"),
    )
    _digest_check("approved_evidence_batches")
    op.create_check_constraint(
        "ck_approved_batches_content_sha256",
        "approved_evidence_batches",
        SHA_CHECK.format(column="content_sha256"),
    )
    op.create_index(
        "ix_approved_batches_report", "approved_evidence_batches", ["report_id", "review_revision"]
    )
    _uuid4_check("approved_evidence_batches", "batch_id")
    _uuid4_check("approved_evidence_batches", "user_id")
    _uuid4_check("approved_evidence_batches", "session_id")
    _uuid4_check("approved_evidence_batches", "report_id")

    op.create_table(
        "approved_batch_sources",
        sa.Column(
            "batch_id",
            UUID,
            sa.ForeignKey("approved_evidence_batches.batch_id", ondelete="RESTRICT"),
            primary_key=True,
        ),
        sa.Column("task_id", UUID, primary_key=True),
        sa.Column("shot_id", UUID, primary_key=True),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("source_envelope_id", UUID, nullable=False),
        sa.Column("report_id", UUID, nullable=False),
        sa.Column("review_revision", sa.Integer(), nullable=False),
        sa.Column("review_status", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["task_id", "shot_id", "report_id", "review_revision", "review_status"],
            [
                "review_shot_results.task_id",
                "review_shot_results.shot_id",
                "review_shot_results.result_report_id",
                "review_shot_results.result_report_revision",
                "review_shot_results.review_status",
            ],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["batch_id", "report_id", "review_revision"],
            [
                "approved_evidence_batches.batch_id",
                "approved_evidence_batches.report_id",
                "approved_evidence_batches.review_revision",
            ],
            ondelete="RESTRICT",
        ),
        sa.CheckConstraint(
            "review_status IN ('approved','corrected')",
            name="ck_approved_batch_sources_approved_only",
        ),
        sa.CheckConstraint("sequence > 0", name="ck_approved_batch_sources_sequence"),
        sa.UniqueConstraint("batch_id", "sequence", name="uq_approved_batch_sources_sequence"),
        sa.UniqueConstraint(
            "batch_id", "source_envelope_id", name="uq_approved_batch_sources_envelope"
        ),
    )
    op.create_index(
        "ix_approved_batch_sources_task", "approved_batch_sources", ["task_id", "shot_id"]
    )
    op.create_index(
        "ix_approved_batch_sources_result",
        "approved_batch_sources",
        ["task_id", "shot_id", "report_id", "review_revision", "review_status"],
    )
    op.create_index(
        "ix_approved_batch_sources_batch_lineage",
        "approved_batch_sources",
        ["batch_id", "report_id", "review_revision"],
    )
    _uuid4_check("approved_batch_sources", "batch_id")
    _uuid4_check("approved_batch_sources", "task_id")
    _uuid4_check("approved_batch_sources", "shot_id")
    _uuid4_check("approved_batch_sources", "source_envelope_id")

    op.create_table(
        "quality_gate_idempotency_receipts",
        sa.Column("actor_principal_id", sa.Text(), primary_key=True),
        sa.Column("operation", sa.Text(), primary_key=True),
        sa.Column("task_id", UUID, primary_key=True),
        sa.Column("shot_id_or_zero_uuid", UUID, primary_key=True),
        sa.Column("idempotency_key", sa.Text(), primary_key=True),
        sa.Column("request_digest", sa.Text(), nullable=False),
        sa.Column("result_kind", sa.Text(), nullable=False),
        sa.Column("result_identity", sa.Text(), nullable=False),
        sa.Column("result_digest", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            SHA_CHECK.format(column="request_digest"), name="ck_qg_receipts_request_digest"
        ),
        sa.CheckConstraint(
            SHA_CHECK.format(column="result_digest"), name="ck_qg_receipts_result_digest"
        ),
    )
    _uuid4_check("quality_gate_idempotency_receipts", "task_id")
    op.create_check_constraint(
        "ck_qg_receipts_shot_uuid4_or_zero",
        "quality_gate_idempotency_receipts",
        "shot_id_or_zero_uuid = '00000000-0000-0000-0000-000000000000'::uuid OR "
        "(substring(shot_id_or_zero_uuid::text from 15 for 1) = '4' AND "
        "substring(shot_id_or_zero_uuid::text from 20 for 1) IN ('8','9','a','b'))",
    )

    op.create_table(
        "quality_gate_outbox",
        sa.Column("event_id", UUID, primary_key=True),
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
        sa.Column("claimed_by", sa.Text()),
        sa.Column("claimed_at", sa.DateTime(timezone=True)),
        sa.Column("claim_until", sa.DateTime(timezone=True)),
        sa.Column("claim_token", UUID),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_error", sa.Text()),
        sa.Column("published_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint("event_type", "idempotency_key", name="uq_qg_outbox_event_idempotency"),
        sa.UniqueConstraint(
            "aggregate_type",
            "aggregate_id",
            "aggregate_version",
            "event_type",
            name="uq_qg_outbox_aggregate_version",
        ),
        sa.CheckConstraint("aggregate_version > 0", name="ck_qg_outbox_aggregate_version"),
        sa.CheckConstraint("attempt_count >= 0", name="ck_qg_outbox_attempt_count"),
        sa.CheckConstraint(
            SHA_CHECK.format(column="payload_sha256"), name="ck_qg_outbox_payload_sha256"
        ),
        sa.CheckConstraint(
            SHA_CHECK.format(column="event_sha256"), name="ck_qg_outbox_event_sha256"
        ),
        sa.CheckConstraint(
            "(claimed_by IS NULL AND claimed_at IS NULL AND claim_until IS NULL "
            "AND claim_token IS NULL) OR "
            "(claimed_by IS NOT NULL AND claimed_at IS NOT NULL AND claim_until IS NOT NULL "
            "AND claim_token IS NOT NULL AND claim_until > claimed_at)",
            name="ck_qg_outbox_claim_shape",
        ),
    )
    op.create_index(
        "ix_quality_gate_outbox_pending",
        "quality_gate_outbox",
        ["available_at", "occurred_at", "event_id"],
        postgresql_where=sa.text("published_at IS NULL"),
    )
    _uuid4_check("quality_gate_outbox", "event_id")
    _uuid4_check("quality_gate_outbox", "aggregate_id")
    _uuid4_check("quality_gate_outbox", "causation_id", nullable=True)
    _uuid4_check("quality_gate_outbox", "correlation_id")
    _uuid4_check("quality_gate_outbox", "claim_token", nullable=True)

    _install_terminal_authority_guard()


def downgrade() -> None:
    op.execute("DROP FUNCTION IF EXISTS enforce_review_task_terminal_authority() CASCADE")
    for table in (
        "quality_gate_outbox",
        "quality_gate_idempotency_receipts",
        "approved_batch_sources",
        "approved_evidence_batches",
        "review_metric_authority",
        "review_publication_lineage",
        "review_rejections",
        "review_shot_results",
        "review_corrections",
        "review_shot_decisions",
        "review_shot_sources",
        "review_tasks",
        "review_result_reports",
        "review_reports",
    ):
        op.drop_table(table)
    op.execute("DROP FUNCTION IF EXISTS enforce_review_report_parent_lineage()")
