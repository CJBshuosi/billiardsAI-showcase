"""Bounded managed startup before any capture or product state exists."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import asdict

from billiards_ai.edge.route_assistance.managed_sessions import (
    ManagedSessionBinding,
    ManagedSessionScope,
    validate_managed_artifact_id,
    validate_managed_rule_profile_id,
)
from billiards_ai.edge.route_assistance.records import FieldSessionRecord, FieldSessionStatus
from billiards_ai.edge.route_assistance.runtime_resources import ManagedRuntimeResourceSpec
from billiards_ai.edge.route_assistance.sync import UrllibSyncHttpTransport
from billiards_ai.lab.field_assistance.managed_lifecycle import close_and_drain_once


def _remaining(deadline, cancelled, monotonic):
    if cancelled():
        raise OSError("managed startup cancelled")
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise TimeoutError("managed startup deadline exceeded")
    return remaining


class BoundedStartupTransport:
    """An independent transport whose every HTTP request recomputes its timeout."""

    def __init__(
        self,
        base_url,
        *,
        deadline,
        cancelled,
        monotonic=time.monotonic,
        transport_factory=UrllibSyncHttpTransport,
    ):
        if (
            isinstance(deadline, bool)
            or not isinstance(deadline, (int, float))
            or not math.isfinite(deadline)
        ):
            raise ValueError("managed startup requires a finite monotonic deadline")
        self._base_url = base_url
        self._deadline, self._cancelled, self._monotonic = deadline, cancelled, monotonic
        self._factory = transport_factory

    def _request(self, method, path, headers, body):
        remaining = _remaining(self._deadline, self._cancelled, self._monotonic)
        transport = self._factory(
            self._base_url, timeout_seconds=min(20.0, remaining), allow_redirects=False
        )
        result = getattr(transport, method)(path, headers, body)
        _remaining(self._deadline, self._cancelled, self._monotonic)
        return result

    def post_json(self, path, headers, body):
        return self._request("post_json", path, headers, body)

    def put_json(self, path, headers, body):
        return self._request("put_json", path, headers, body)


class ManagedStartupCoordinator:
    """Recover an immutable request or positively close and retire its predecessor."""

    def __init__(self, *, scope: ManagedSessionScope, gateway_factory, monotonic=time.monotonic):
        if not isinstance(scope, ManagedSessionScope):
            raise ValueError("managed startup requires configured scope")
        self.scope, self.gateway_factory, self.monotonic = scope, gateway_factory, monotonic
        self._match_lock = threading.RLock()

    def prepare(
        self,
        store,
        *,
        artifact_id,
        rule_profile_id,
        policy,
        deadline=None,
        cancelled=lambda: False,
        resource_spec=None,
        session_policy="legacy",
    ) -> FieldSessionRecord:
        if policy not in {"usb_resume", "blackbox_fresh"}:
            raise ValueError("invalid managed startup policy")
        validate_managed_artifact_id(artifact_id)
        validate_managed_rule_profile_id(rule_profile_id)
        if resource_spec is not None:
            if not isinstance(resource_spec, ManagedRuntimeResourceSpec):
                raise ValueError("invalid managed runtime resource specification")
            resource_spec = ManagedRuntimeResourceSpec.from_payload(resource_spec.as_payload())
            if (
                resource_spec.artifact_id != artifact_id
                or resource_spec.rule_profile_id != rule_profile_id
            ):
                raise ValueError("managed startup resource identity changed")
        if deadline is None:
            deadline = self.monotonic() + 60.0
        if (
            isinstance(deadline, bool)
            or not isinstance(deadline, (int, float))
            or not math.isfinite(deadline)
        ):
            raise ValueError("managed startup requires a finite monotonic deadline")

        def check():
            _remaining(deadline, cancelled, self.monotonic)

        check()
        request = store.get_active_managed_open_request(self.scope)
        if (
            request is not None
            and request.phase != "materialized"
            and (request.artifact_id != artifact_id or request.rule_profile_id != rule_profile_id)
        ):
            raise ValueError("pending managed startup artifact or profile changed")
        predecessor = (
            store.get_latest_retired_managed_open_request(self.scope) if request is None else None
        )
        retained = request if request is not None else predecessor
        if retained is not None and retained.phase == "materialized":
            with self._match_lock:
                session = store.validate_managed_local_session(self.scope, retained.open_request_id)
                shots = store.list_shots(session.session_id)
                unsettled = tuple(shot for shot in shots if shot.settled_ms is None)
                if any(
                    shot.sequence != index
                    or shot.session_id != session.session_id
                    or shot.media.artifact_id != session.artifact_id
                    for index, shot in enumerate(shots, 1)
                ) or (unsettled and (len(unsettled) != 1 or unsettled[0] != shots[-1])):
                    raise ValueError("persisted shot recovery is ambiguous")
        can_resume = (
            request is not None
            and request.phase == "materialized"
            and policy == "usb_resume"
            and session.status is FieldSessionStatus.OPEN
            and request.artifact_id == artifact_id
            and request.rule_profile_id == rule_profile_id
            and not unsettled
            and request.binding.status == "open"
        )
        if can_resume and store.get_session_policy(session.session_id) != session_policy:
            raise ValueError("session recording policy changed on resume")
        # Replay allocates directly instead of reserving again. Compare its
        # immutable intent before any RPC, but allow capture-free old closure.
        if (
            request is not None
            and resource_spec is not None
            and (request.phase != "materialized" or can_resume)
        ):
            original = store.get_managed_resource_spec(self.scope, request.open_request_id)
            if original.spec != resource_spec:
                raise ValueError("managed startup resources changed for retained scene")
        check()
        gateway = self.gateway_factory(deadline=deadline, cancelled=cancelled)
        check()

        def allocate(request):
            check()
            binding = gateway.open_managed_session(
                open_request_id=request.open_request_id, expected_table_id=self.scope.table_id
            )
            check()
            if not isinstance(binding, ManagedSessionBinding):
                raise OSError("cloud gateway returned invalid managed session")
            binding = ManagedSessionBinding.from_payload(
                asdict(binding),
                expected_table_id=self.scope.table_id,
                expected_device_id=self.scope.device_id,
            )
            request = store.record_managed_binding(self.scope, request.open_request_id, binding)
            check()
            return request

        if request is not None and request.phase != "materialized":
            request = allocate(request)
            # Ended allocation without a retained scene is recovery, never a fake
            # local ended scene and never a reason to rotate the original UUID.
            result = store.materialize_managed_session(
                self.scope,
                request.open_request_id,
                session_policy=session_policy,
            )
            check()
            return result

        if request is not None:
            if can_resume:
                request = allocate(request)
                if request.binding.status == "open":
                    result = store.materialize_managed_session(
                        self.scope,
                        request.open_request_id,
                        session_policy=session_policy,
                    )
                    check()
                    return result
            check()
            with self._match_lock:
                store.end_session(session.session_id)
            check()
        else:
            # A local audit snapshot never substitutes for fresh signed closure.
            request = predecessor

        if request is not None:
            while True:
                check()
                try:
                    close = close_and_drain_once(store, request, gateway, self._match_lock)
                except OSError:
                    check()
                    raise
                check()
                if close.coverage == "sealed":
                    store.retire_managed_open(
                        self.scope, request.open_request_id, sealed_close=close
                    )
                    break

        check()
        request = store.reserve_managed_open(
            self.scope,
            artifact_id=artifact_id,
            rule_profile_id=rule_profile_id,
            resource_spec=resource_spec,
        )
        request = allocate(request)
        result = store.materialize_managed_session(
            self.scope,
            request.open_request_id,
            session_policy=session_policy,
        )
        check()
        return result
