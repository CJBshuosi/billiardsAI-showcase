"""Bind quality-gate publication authorities to exact reviewed evidence."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "20260810_02"
down_revision = "20260810_01"
branch_labels = None
depends_on = None

UUID = postgresql.UUID(as_uuid=True)
JSONB = postgresql.JSONB(astext_type=sa.Text())
SHA_CHECK = "{column} ~ '^[0-9a-f]{{64}}$'"


def _uuid4_check(table: str, column: str) -> None:
    op.create_check_constraint(
        f"ck_{table}_{column}_uuid4",
        table,
        f"substring({column}::text from 15 for 1) = '4' "
        f"AND substring({column}::text from 20 for 1) IN ('8','9','a','b')",
    )


def _reject_legacy_publication_rows() -> None:
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM review_publication_lineage)
               OR EXISTS (SELECT 1 FROM review_metric_authority)
               OR EXISTS (SELECT 1 FROM approved_evidence_batches)
               OR EXISTS (SELECT 1 FROM approved_batch_sources) THEN
                RAISE EXCEPTION USING
                    MESSAGE = 'cannot upgrade 20260810_02: legacy quality-gate publication'
                        || ' rows require explicit migration',
                    ERRCODE = '55000';
            END IF;
        END;
        $$;
        """
    )


def _replace_terminal_authority_guard() -> None:
    op.execute(
        """
        CREATE OR REPLACE FUNCTION enforce_review_task_terminal_authority()
        RETURNS trigger AS $$
        DECLARE
            checked_task_id uuid;
            terminal_status text;
            result_report_status text;
            source_count integer;
            decision_count integer;
            result_count integer;
            rejection_count integer;
            promotable_count integer;
            corrected_count integer;
            invalid_outcome_count integer;
            invalid_correction_count integer;
            invalid_result_report_count integer;
            invalid_result_source_count integer;
            publication_count integer;
            batch_count integer;
            outbox_count integer;
            metric_count integer;
            batch_source_count integer;
            complete_graph_count integer;
            incomplete_projection_count integer;
        BEGIN
            IF TG_TABLE_NAME = 'review_result_reports' THEN
                SELECT task_id INTO checked_task_id
                  FROM review_tasks
                 WHERE result_report_id = COALESCE(NEW.report_id, OLD.report_id)
                   AND result_report_revision = COALESCE(NEW.revision, OLD.revision);
            ELSE
                checked_task_id := COALESCE(
                    NULLIF(to_jsonb(NEW) ->> 'task_id', '')::uuid,
                    NULLIF(to_jsonb(OLD) ->> 'task_id', '')::uuid,
                    NULLIF(to_jsonb(NEW) ->> 'causation_id', '')::uuid,
                    NULLIF(to_jsonb(OLD) ->> 'causation_id', '')::uuid
                );
            END IF;

            IF checked_task_id IS NULL THEN
                RETURN NULL;
            END IF;

            SELECT status INTO terminal_status FROM review_tasks WHERE task_id = checked_task_id;
            IF terminal_status IS NULL OR terminal_status IN ('pending_review', 'in_review') THEN
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

            IF terminal_status = 'rejected' THEN
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

            SELECT count(*) INTO invalid_result_report_count
              FROM review_tasks t
              JOIN review_result_reports rr
                ON rr.report_id = t.result_report_id
               AND rr.revision = t.result_report_revision
             WHERE t.task_id = checked_task_id
               AND (rr.session_id IS DISTINCT FROM t.session_id
                    OR rr.source_report_id IS DISTINCT FROM t.source_report_id
                    OR rr.source_report_revision IS DISTINCT FROM t.source_report_revision);

            IF invalid_result_report_count <> 0 THEN
                RAISE EXCEPTION 'invalid terminal task result report identity'
                    USING ERRCODE = '23514';
            END IF;

            IF source_count = 0 OR decision_count <> source_count OR result_count <> source_count
               OR rejection_count <> 0
               OR result_report_status IS DISTINCT FROM upper(terminal_status) THEN
                RAISE EXCEPTION 'invalid terminal task authority coverage'
                    USING ERRCODE = '23514';
            END IF;

            SELECT count(*) INTO invalid_result_source_count
              FROM review_shot_results r
              LEFT JOIN review_shot_sources s
                ON s.task_id = r.task_id
               AND s.shot_id = r.shot_id
               AND s.session_id = r.session_id
               AND s.sequence = r.sequence
             WHERE r.task_id = checked_task_id
               AND s.task_id IS NULL;

            IF invalid_result_source_count <> 0 THEN
                RAISE EXCEPTION 'invalid terminal task result source identity'
                    USING ERRCODE = '23514';
            END IF;

            SELECT count(*) FILTER (WHERE outcome IN ('approved', 'corrected')),
                   count(*) FILTER (WHERE outcome = 'corrected')
              INTO promotable_count, corrected_count
              FROM review_shot_decisions
             WHERE task_id = checked_task_id;

            SELECT count(*) INTO invalid_outcome_count
              FROM review_shot_decisions d
              JOIN review_shot_results r USING (task_id, shot_id)
             WHERE d.task_id = checked_task_id
               AND d.outcome <> r.review_status;

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
               OR promotable_count = 0
               OR (terminal_status = 'approved' AND corrected_count <> 0)
               OR (terminal_status = 'corrected' AND corrected_count = 0) THEN
                RAISE EXCEPTION 'invalid terminal task outcome or correction authority'
                    USING ERRCODE = '23514';
            END IF;

            SELECT count(*) INTO publication_count
              FROM review_publication_lineage
             WHERE task_id = checked_task_id;
            SELECT count(*) INTO batch_count
              FROM approved_evidence_batches
             WHERE task_id = checked_task_id;
            SELECT count(*) INTO outbox_count
              FROM quality_gate_outbox
             WHERE causation_id = checked_task_id;
            SELECT count(*) INTO metric_count
              FROM review_metric_authority
             WHERE task_id = checked_task_id;
            SELECT count(*) INTO batch_source_count
              FROM approved_batch_sources
             WHERE task_id = checked_task_id;

            SELECT count(*) INTO complete_graph_count
              FROM review_publication_lineage p
              JOIN approved_evidence_batches b
                ON b.batch_id = p.batch_id
               AND b.task_id = p.task_id
               AND b.user_id = p.user_id
               AND b.session_id = p.session_id
               AND b.report_id = p.report_id
               AND b.review_revision = p.review_revision
               AND b.event_id = p.event_id
               AND b.content_sha256 = p.content_sha256
               AND b.approval_kind = p.approval_kind
               AND b.approval_policy_version = p.approval_policy_version
               AND b.calibration_id = p.calibration_id
               AND b.calibration_revision_id = p.calibration_revision_id
               AND b.calibration_revision_number = p.calibration_revision_number
              JOIN quality_gate_outbox o ON o.event_id = p.event_id
             WHERE p.task_id = checked_task_id
               AND p.task_status = terminal_status
               AND o.event_type = 'evidence.session.approved.v1'
               AND o.aggregate_type = 'review_session'
               AND o.aggregate_id = p.session_id
               AND o.aggregate_version = p.review_revision
               AND o.causation_id = p.task_id
               AND o.correlation_id = p.session_id;

            SELECT count(*) INTO incomplete_projection_count
              FROM review_shot_results r
              LEFT JOIN review_metric_authority m
                ON m.task_id = r.task_id AND m.shot_id = r.shot_id
              LEFT JOIN approved_batch_sources s
                ON s.task_id = r.task_id AND s.shot_id = r.shot_id
             WHERE r.task_id = checked_task_id
               AND r.review_status IN ('approved', 'corrected')
               AND (m.task_id IS NULL OR s.task_id IS NULL);

            IF publication_count <> 1 OR batch_count <> 1 OR outbox_count <> 1
               OR complete_graph_count <> 1
               OR metric_count <> promotable_count
               OR batch_source_count <> promotable_count
               OR incomplete_projection_count <> 0 THEN
                RAISE EXCEPTION 'invalid terminal task publication graph completeness'
                    USING ERRCODE = '23514';
            END IF;
            RETURN NULL;
        END;
        $$ LANGUAGE plpgsql;
        """
    )


def _install_ownership_guard() -> None:
    op.execute(
        """
        CREATE FUNCTION enforce_review_session_ownership_binding()
        RETURNS trigger AS $$
        DECLARE
            current_status text;
        BEGIN
            IF TG_OP IN ('UPDATE', 'DELETE') THEN
                RAISE EXCEPTION 'review session ownership is immutable'
                    USING ERRCODE = '23514';
            END IF;
            SELECT status INTO STRICT current_status
              FROM review_tasks
             WHERE task_id = NEW.task_id AND session_id = NEW.session_id;
            IF current_status NOT IN ('pending_review', 'in_review') THEN
                RAISE EXCEPTION 'review session ownership must be bound before finalize'
                    USING ERRCODE = '23514';
            END IF;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_review_session_ownership_binding
        BEFORE INSERT OR UPDATE OR DELETE ON review_session_ownership
        FOR EACH ROW EXECUTE FUNCTION enforce_review_session_ownership_binding();
        """
    )


def _install_publication_graph_guards() -> None:
    for trigger_name, table in (
        ("trg_review_result_reports_terminal_authority", "review_result_reports"),
        ("trg_review_sources_terminal_authority", "review_shot_sources"),
        ("trg_review_publication_terminal_authority", "review_publication_lineage"),
        ("trg_review_metric_terminal_authority", "review_metric_authority"),
        ("trg_approved_batches_terminal_authority", "approved_evidence_batches"),
        ("trg_approved_batch_sources_terminal_authority", "approved_batch_sources"),
        ("trg_quality_gate_outbox_terminal_authority", "quality_gate_outbox"),
    ):
        op.execute(
            f"CREATE CONSTRAINT TRIGGER {trigger_name} "
            f"AFTER INSERT OR UPDATE OR DELETE ON {table} "
            "DEFERRABLE INITIALLY DEFERRED FOR EACH ROW "
            "EXECUTE FUNCTION enforce_review_task_terminal_authority()"
        )


def upgrade() -> None:
    _reject_legacy_publication_rows()
    _replace_terminal_authority_guard()

    op.create_unique_constraint(
        "uq_review_tasks_terminal_authority",
        "review_tasks",
        ["task_id", "session_id", "result_report_id", "result_report_revision", "status"],
    )
    op.create_unique_constraint(
        "uq_review_shot_sources_exact_authority",
        "review_shot_sources",
        ["task_id", "shot_id", "session_id", "sequence", "source_envelope_id"],
    )
    op.create_unique_constraint(
        "uq_review_shot_results_exact_authority",
        "review_shot_results",
        [
            "task_id",
            "shot_id",
            "result_report_id",
            "result_report_revision",
            "review_status",
            "sequence",
            "session_id",
        ],
    )

    op.create_table(
        "review_session_ownership",
        sa.Column("task_id", UUID, primary_key=True),
        sa.Column("session_id", UUID, nullable=False),
        sa.Column("user_id", UUID, nullable=False),
        sa.Column("bound_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("payload", JSONB, nullable=False),
        sa.Column("payload_sha256", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(
            ["task_id", "session_id"],
            ["review_tasks.task_id", "review_tasks.session_id"],
            ondelete="RESTRICT",
        ),
        sa.UniqueConstraint(
            "task_id",
            "session_id",
            "user_id",
            name="uq_review_session_ownership_authority",
        ),
        sa.CheckConstraint(
            SHA_CHECK.format(column="payload_sha256"),
            name="ck_review_session_ownership_payload_sha256",
        ),
    )
    _uuid4_check("review_session_ownership", "task_id")
    _uuid4_check("review_session_ownership", "session_id")
    _uuid4_check("review_session_ownership", "user_id")
    _install_ownership_guard()

    publication_columns = (
        sa.Column("task_status", sa.Text(), nullable=False),
        sa.Column("batch_id", UUID, nullable=False),
        sa.Column("event_id", UUID, nullable=False),
        sa.Column("content_sha256", sa.Text(), nullable=False),
        sa.Column("approval_kind", sa.Text(), nullable=False),
        sa.Column("approval_policy_version", sa.Text(), nullable=False),
        sa.Column("calibration_id", UUID, nullable=False),
        sa.Column("calibration_revision_id", UUID, nullable=False),
        sa.Column("calibration_revision_number", sa.Integer(), nullable=False),
    )
    for column in publication_columns:
        op.add_column("review_publication_lineage", column)
    op.create_check_constraint(
        "ck_review_publication_lineage_task_status",
        "review_publication_lineage",
        "task_status IN ('approved','corrected')",
    )
    op.create_check_constraint(
        "ck_review_publication_lineage_human_policy",
        "review_publication_lineage",
        "approval_kind = 'HUMAN_REVIEW' AND approval_policy_version = 'human-review-v1'",
    )
    op.create_check_constraint(
        "ck_review_publication_lineage_content_sha256",
        "review_publication_lineage",
        SHA_CHECK.format(column="content_sha256"),
    )
    op.create_check_constraint(
        "ck_review_publication_lineage_calibration_revision",
        "review_publication_lineage",
        "calibration_revision_number > 0",
    )
    for column in ("batch_id", "event_id", "calibration_id", "calibration_revision_id"):
        _uuid4_check("review_publication_lineage", column)
    op.create_foreign_key(
        "fk_review_publication_ownership",
        "review_publication_lineage",
        "review_session_ownership",
        ["task_id", "session_id", "user_id"],
        ["task_id", "session_id", "user_id"],
        ondelete="RESTRICT",
    )
    op.create_foreign_key(
        "fk_review_publication_terminal_task",
        "review_publication_lineage",
        "review_tasks",
        ["task_id", "session_id", "report_id", "review_revision", "task_status"],
        ["task_id", "session_id", "result_report_id", "result_report_revision", "status"],
        ondelete="RESTRICT",
        deferrable=True,
        initially="DEFERRED",
    )
    op.create_foreign_key(
        "fk_review_publication_batch",
        "review_publication_lineage",
        "approved_evidence_batches",
        ["batch_id"],
        ["batch_id"],
        ondelete="RESTRICT",
        deferrable=True,
        initially="DEFERRED",
    )
    op.create_foreign_key(
        "fk_review_publication_event",
        "review_publication_lineage",
        "quality_gate_outbox",
        ["event_id"],
        ["event_id"],
        ondelete="RESTRICT",
        deferrable=True,
        initially="DEFERRED",
    )
    op.create_unique_constraint(
        "uq_review_publication_metric_authority",
        "review_publication_lineage",
        [
            "task_id",
            "session_id",
            "report_id",
            "review_revision",
            "calibration_id",
            "calibration_revision_id",
            "calibration_revision_number",
        ],
    )
    op.create_unique_constraint(
        "uq_review_publication_batch_authority",
        "review_publication_lineage",
        [
            "task_id",
            "batch_id",
            "user_id",
            "session_id",
            "report_id",
            "review_revision",
            "event_id",
            "content_sha256",
            "approval_kind",
            "approval_policy_version",
            "calibration_id",
            "calibration_revision_id",
            "calibration_revision_number",
        ],
    )
    op.create_index(
        "ix_review_publication_ownership",
        "review_publication_lineage",
        ["task_id", "session_id", "user_id"],
    )
    op.create_index(
        "ix_review_publication_batch",
        "review_publication_lineage",
        ["batch_id"],
    )
    op.create_index(
        "ix_review_publication_event",
        "review_publication_lineage",
        ["event_id"],
    )
    op.create_index(
        "ix_review_publication_terminal_task",
        "review_publication_lineage",
        ["task_id", "session_id", "report_id", "review_revision", "task_status"],
    )

    metric_columns = (
        sa.Column("result_report_id", UUID, nullable=False),
        sa.Column("result_report_revision", sa.Integer(), nullable=False),
        sa.Column("session_id", UUID, nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("review_status", sa.Text(), nullable=False),
        sa.Column("calibration_id", UUID, nullable=False),
        sa.Column("calibration_revision_id", UUID, nullable=False),
        sa.Column("calibration_revision_number", sa.Integer(), nullable=False),
    )
    for column in metric_columns:
        op.add_column("review_metric_authority", column)
    op.create_check_constraint(
        "ck_review_metric_authority_status",
        "review_metric_authority",
        "review_status IN ('approved','corrected')",
    )
    op.create_check_constraint(
        "ck_review_metric_authority_sequence",
        "review_metric_authority",
        "sequence > 0 AND result_report_revision > 0 AND calibration_revision_number > 0",
    )
    for column in (
        "result_report_id",
        "session_id",
        "calibration_id",
        "calibration_revision_id",
    ):
        _uuid4_check("review_metric_authority", column)
    op.create_foreign_key(
        "fk_review_metric_exact_result",
        "review_metric_authority",
        "review_shot_results",
        [
            "task_id",
            "shot_id",
            "result_report_id",
            "result_report_revision",
            "review_status",
            "sequence",
            "session_id",
        ],
        [
            "task_id",
            "shot_id",
            "result_report_id",
            "result_report_revision",
            "review_status",
            "sequence",
            "session_id",
        ],
        ondelete="RESTRICT",
        deferrable=True,
        initially="DEFERRED",
    )
    op.create_foreign_key(
        "fk_review_metric_publication",
        "review_metric_authority",
        "review_publication_lineage",
        [
            "task_id",
            "session_id",
            "result_report_id",
            "result_report_revision",
            "calibration_id",
            "calibration_revision_id",
            "calibration_revision_number",
        ],
        [
            "task_id",
            "session_id",
            "report_id",
            "review_revision",
            "calibration_id",
            "calibration_revision_id",
            "calibration_revision_number",
        ],
        ondelete="RESTRICT",
        deferrable=True,
        initially="DEFERRED",
    )
    op.create_index(
        "ix_review_metric_exact_result",
        "review_metric_authority",
        [
            "task_id",
            "shot_id",
            "result_report_id",
            "result_report_revision",
            "review_status",
            "sequence",
            "session_id",
        ],
    )
    op.create_index(
        "ix_review_metric_publication",
        "review_metric_authority",
        [
            "task_id",
            "session_id",
            "result_report_id",
            "result_report_revision",
            "calibration_id",
            "calibration_revision_id",
            "calibration_revision_number",
        ],
    )

    batch_columns = (
        sa.Column("task_id", UUID, nullable=False),
        sa.Column("event_id", UUID, nullable=False),
        sa.Column("approval_kind", sa.Text(), nullable=False),
        sa.Column("approval_policy_version", sa.Text(), nullable=False),
        sa.Column("calibration_id", UUID, nullable=False),
        sa.Column("calibration_revision_id", UUID, nullable=False),
        sa.Column("calibration_revision_number", sa.Integer(), nullable=False),
    )
    for column in batch_columns:
        op.add_column("approved_evidence_batches", column)
    op.create_check_constraint(
        "ck_approved_batches_human_policy",
        "approved_evidence_batches",
        "approval_kind = 'HUMAN_REVIEW' AND approval_policy_version = 'human-review-v1'",
    )
    op.create_check_constraint(
        "ck_approved_batches_calibration_revision",
        "approved_evidence_batches",
        "calibration_revision_number > 0",
    )
    for column in ("task_id", "event_id", "calibration_id", "calibration_revision_id"):
        _uuid4_check("approved_evidence_batches", column)
    op.create_foreign_key(
        "fk_approved_batches_publication",
        "approved_evidence_batches",
        "review_publication_lineage",
        [
            "task_id",
            "batch_id",
            "user_id",
            "session_id",
            "report_id",
            "review_revision",
            "event_id",
            "content_sha256",
            "approval_kind",
            "approval_policy_version",
            "calibration_id",
            "calibration_revision_id",
            "calibration_revision_number",
        ],
        [
            "task_id",
            "batch_id",
            "user_id",
            "session_id",
            "report_id",
            "review_revision",
            "event_id",
            "content_sha256",
            "approval_kind",
            "approval_policy_version",
            "calibration_id",
            "calibration_revision_id",
            "calibration_revision_number",
        ],
        ondelete="RESTRICT",
    )
    op.create_index(
        "ix_approved_batches_publication",
        "approved_evidence_batches",
        [
            "task_id",
            "batch_id",
            "user_id",
            "session_id",
            "report_id",
            "review_revision",
            "event_id",
            "content_sha256",
            "approval_kind",
            "approval_policy_version",
            "calibration_id",
            "calibration_revision_id",
            "calibration_revision_number",
        ],
    )

    op.add_column("approved_batch_sources", sa.Column("session_id", UUID, nullable=False))
    _uuid4_check("approved_batch_sources", "session_id")
    op.create_foreign_key(
        "fk_approved_batch_sources_exact_result",
        "approved_batch_sources",
        "review_shot_results",
        [
            "task_id",
            "shot_id",
            "report_id",
            "review_revision",
            "review_status",
            "sequence",
            "session_id",
        ],
        [
            "task_id",
            "shot_id",
            "result_report_id",
            "result_report_revision",
            "review_status",
            "sequence",
            "session_id",
        ],
        ondelete="RESTRICT",
    )
    op.create_foreign_key(
        "fk_approved_batch_sources_exact_source",
        "approved_batch_sources",
        "review_shot_sources",
        ["task_id", "shot_id", "session_id", "sequence", "source_envelope_id"],
        ["task_id", "shot_id", "session_id", "sequence", "source_envelope_id"],
        ondelete="RESTRICT",
    )
    op.create_index(
        "ix_approved_batch_sources_exact_result",
        "approved_batch_sources",
        [
            "task_id",
            "shot_id",
            "report_id",
            "review_revision",
            "review_status",
            "sequence",
            "session_id",
        ],
    )
    op.create_index(
        "ix_approved_batch_sources_exact_source",
        "approved_batch_sources",
        ["task_id", "shot_id", "session_id", "sequence", "source_envelope_id"],
    )
    _install_publication_graph_guards()


def downgrade() -> None:
    for trigger_name, table in (
        ("trg_quality_gate_outbox_terminal_authority", "quality_gate_outbox"),
        ("trg_approved_batch_sources_terminal_authority", "approved_batch_sources"),
        ("trg_approved_batches_terminal_authority", "approved_evidence_batches"),
        ("trg_review_metric_terminal_authority", "review_metric_authority"),
        ("trg_review_publication_terminal_authority", "review_publication_lineage"),
        ("trg_review_sources_terminal_authority", "review_shot_sources"),
        ("trg_review_result_reports_terminal_authority", "review_result_reports"),
    ):
        op.execute(f"DROP TRIGGER IF EXISTS {trigger_name} ON {table}")

    op.drop_index("ix_approved_batch_sources_exact_source", table_name="approved_batch_sources")
    op.drop_index("ix_approved_batch_sources_exact_result", table_name="approved_batch_sources")
    op.drop_constraint(
        "fk_approved_batch_sources_exact_source", "approved_batch_sources", type_="foreignkey"
    )
    op.drop_constraint(
        "fk_approved_batch_sources_exact_result", "approved_batch_sources", type_="foreignkey"
    )
    op.drop_constraint(
        "ck_approved_batch_sources_session_id_uuid4", "approved_batch_sources", type_="check"
    )
    op.drop_column("approved_batch_sources", "session_id")

    op.drop_index("ix_approved_batches_publication", table_name="approved_evidence_batches")
    op.drop_constraint(
        "fk_approved_batches_publication", "approved_evidence_batches", type_="foreignkey"
    )
    op.drop_constraint(
        "ck_approved_batches_calibration_revision", "approved_evidence_batches", type_="check"
    )
    op.drop_constraint(
        "ck_approved_batches_human_policy", "approved_evidence_batches", type_="check"
    )
    for column in ("calibration_revision_id", "calibration_id", "event_id", "task_id"):
        op.drop_constraint(
            f"ck_approved_evidence_batches_{column}_uuid4",
            "approved_evidence_batches",
            type_="check",
        )
    for column in (
        "calibration_revision_number",
        "calibration_revision_id",
        "calibration_id",
        "approval_policy_version",
        "approval_kind",
        "event_id",
        "task_id",
    ):
        op.drop_column("approved_evidence_batches", column)

    op.drop_index("ix_review_metric_publication", table_name="review_metric_authority")
    op.drop_index("ix_review_metric_exact_result", table_name="review_metric_authority")
    op.drop_constraint(
        "fk_review_metric_publication", "review_metric_authority", type_="foreignkey"
    )
    op.drop_constraint(
        "fk_review_metric_exact_result", "review_metric_authority", type_="foreignkey"
    )
    op.drop_constraint(
        "ck_review_metric_authority_sequence", "review_metric_authority", type_="check"
    )
    op.drop_constraint(
        "ck_review_metric_authority_status", "review_metric_authority", type_="check"
    )
    for column in (
        "calibration_revision_id",
        "calibration_id",
        "session_id",
        "result_report_id",
    ):
        op.drop_constraint(
            f"ck_review_metric_authority_{column}_uuid4",
            "review_metric_authority",
            type_="check",
        )
    for column in (
        "calibration_revision_number",
        "calibration_revision_id",
        "calibration_id",
        "review_status",
        "sequence",
        "session_id",
        "result_report_revision",
        "result_report_id",
    ):
        op.drop_column("review_metric_authority", column)

    op.drop_index("ix_review_publication_terminal_task", table_name="review_publication_lineage")
    op.drop_index("ix_review_publication_event", table_name="review_publication_lineage")
    op.drop_index("ix_review_publication_batch", table_name="review_publication_lineage")
    op.drop_index("ix_review_publication_ownership", table_name="review_publication_lineage")
    op.drop_constraint(
        "uq_review_publication_batch_authority", "review_publication_lineage", type_="unique"
    )
    op.drop_constraint(
        "uq_review_publication_metric_authority", "review_publication_lineage", type_="unique"
    )
    op.drop_constraint(
        "fk_review_publication_terminal_task", "review_publication_lineage", type_="foreignkey"
    )
    op.drop_constraint(
        "fk_review_publication_event", "review_publication_lineage", type_="foreignkey"
    )
    op.drop_constraint(
        "fk_review_publication_batch", "review_publication_lineage", type_="foreignkey"
    )
    op.drop_constraint(
        "fk_review_publication_ownership", "review_publication_lineage", type_="foreignkey"
    )
    op.drop_constraint(
        "ck_review_publication_lineage_calibration_revision",
        "review_publication_lineage",
        type_="check",
    )
    op.drop_constraint(
        "ck_review_publication_lineage_content_sha256",
        "review_publication_lineage",
        type_="check",
    )
    op.drop_constraint(
        "ck_review_publication_lineage_human_policy",
        "review_publication_lineage",
        type_="check",
    )
    op.drop_constraint(
        "ck_review_publication_lineage_task_status",
        "review_publication_lineage",
        type_="check",
    )
    for column in ("calibration_revision_id", "calibration_id", "event_id", "batch_id"):
        op.drop_constraint(
            f"ck_review_publication_lineage_{column}_uuid4",
            "review_publication_lineage",
            type_="check",
        )
    for column in (
        "calibration_revision_number",
        "calibration_revision_id",
        "calibration_id",
        "approval_policy_version",
        "approval_kind",
        "content_sha256",
        "event_id",
        "batch_id",
        "task_status",
    ):
        op.drop_column("review_publication_lineage", column)

    op.execute("DROP FUNCTION IF EXISTS enforce_review_session_ownership_binding() CASCADE")
    op.drop_table("review_session_ownership")
    op.drop_constraint(
        "uq_review_shot_results_exact_authority", "review_shot_results", type_="unique"
    )
    op.drop_constraint(
        "uq_review_shot_sources_exact_authority", "review_shot_sources", type_="unique"
    )
    op.drop_constraint("uq_review_tasks_terminal_authority", "review_tasks", type_="unique")

    # Restore the Story 11.1 guard verbatim for a reversible one-step downgrade.
    op.execute(
        """
        CREATE OR REPLACE FUNCTION enforce_review_task_terminal_authority()
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
              FROM review_tasks t JOIN review_result_reports rr
                ON rr.report_id = t.result_report_id AND rr.revision = t.result_report_revision
             WHERE t.task_id = checked_task_id;
            IF source_count = 0 OR decision_count <> source_count OR result_count <> source_count
               OR rejection_count <> 0 OR result_report_status <> upper(task_status) THEN
                RAISE EXCEPTION 'invalid terminal task authority coverage' USING ERRCODE = '23514';
            END IF;
            SELECT count(*) INTO invalid_outcome_count
              FROM review_shot_decisions d JOIN review_shot_results r USING (task_id, shot_id)
             WHERE d.task_id = checked_task_id
               AND (d.outcome <> r.review_status OR d.outcome = 'rejected');
            SELECT count(*) INTO invalid_correction_count
              FROM review_shot_decisions d
             WHERE d.task_id = checked_task_id
               AND ((d.outcome = 'corrected' AND NOT EXISTS (
                        SELECT 1 FROM review_corrections c
                         WHERE c.task_id = d.task_id AND c.shot_id = d.shot_id))
                    OR (d.outcome <> 'corrected' AND EXISTS (
                        SELECT 1 FROM review_corrections c
                         WHERE c.task_id = d.task_id AND c.shot_id = d.shot_id)));
            IF invalid_outcome_count <> 0 OR invalid_correction_count <> 0
               OR (task_status = 'approved' AND EXISTS (
                    SELECT 1 FROM review_shot_decisions
                     WHERE task_id = checked_task_id AND outcome <> 'approved'))
               OR (task_status = 'corrected' AND NOT EXISTS (
                    SELECT 1 FROM review_shot_decisions
                     WHERE task_id = checked_task_id AND outcome = 'corrected')) THEN
                RAISE EXCEPTION 'invalid terminal task outcome or correction authority'
                    USING ERRCODE = '23514';
            END IF;
            RETURN NULL;
        END;
        $$ LANGUAGE plpgsql;
        """
    )
