"""Finite S1 feasibility witnesses using the existing pinned physics adapter.

Internal candidate parameters are not user instructions. Success is conditional
on the configured model; no real-table accuracy or exhaustive reachability claim.
"""

from math import hypot


def prepare_s1_physics():
    """Load the original Python physics functions in a fresh owned worker.

    No table state is constructed. The caller runs this before model loading,
    so its temporary Numba decorator setting cannot affect the model thread.
    """
    import sys

    from numba import config
    from numba.core.compiler_lock import global_compiler_lock

    with global_compiler_lock:
        if any(
            name == "pooltool"
            or name.startswith("pooltool.")
            or name == "billiards_ai.cloud.engine.pooltool_adapter"
            for name in tuple(sys.modules)
        ):
            raise RuntimeError("S1 physics preparation requires a fresh worker")
        previous = config.DISABLE_JIT
        try:
            config.DISABLE_JIT = 1
            from billiards_ai.cloud.engine.pooltool_adapter import PooltoolIdealPathAdapter

            PooltoolIdealPathAdapter.prepare_numeric_runtime()
        finally:
            config.DISABLE_JIT = previous


def stop_witnesses(state, route, center, radius, cancelled):
    # Reuse the stricter existing table-plane mapping/event boundary rather than
    # the legacy default-table centre-preference score. Engine loading stays lazy.
    from billiards_ai.cloud.engine.pooltool_adapter import PooltoolIdealPathAdapter
    from billiards_ai.shared.models.ideal_path import IdealPathPocket, IdealPathRequest

    pockets = {
        "bottom_left": "lb",
        "bottom_middle": "lc",
        "bottom_right": "lt",
        "top_left": "rb",
        "top_middle": "rc",
        "top_right": "rt",
    }
    names = {"bottom_middle": "BOTTOM_CENTER", "top_middle": "TOP_CENTER"}
    objects = [ball for ball in state.balls if ball.track_id != state.cue_ball().track_id]
    ids = {ball.track_id: index for index, ball in enumerate(objects, 1)}
    target = ids[route.target_track_id]
    table = dict(
        configuration_version="chinese-eight-ball-9ft-v1",
        length_mm=2540,
        width_mm=1270,
        ball_diameter_mm=57,
        corner_pocket_mouth_mm=86,
        side_pocket_mouth_mm=92,
        cushion_nose_width_mm=40,
    )
    request = IdealPathRequest(
        cue_ball={"ball_id": 0, "position": state.cue_ball().center.model_dump()},
        object_balls=tuple(
            {"ball_id": ids[ball.track_id], "position": ball.center.model_dump()}
            for ball in objects
        ),
        target_ball_id=target,
        target_pocket=IdealPathPocket(names.get(route.pocket.value, route.pocket.value.upper())),
        intent="DIRECT_POT",
        target_ball_reviewed=True,
        target_pocket_reviewed=True,
        intent_reviewed=True,
        calibration_status="ACTIVE",
        table=table,
    )
    adapter = PooltoolIdealPathAdapter()
    adapter._require_engine()
    from billiards_ai.cloud.engine.pooltool_adapter import pt

    passed = 0
    for offset in (-0.5, -0.4, -0.3):
        if cancelled():
            raise InterruptedError("S1 physics cancelled")
        system = adapter._build_system(request)
        pocket = pockets[route.pocket.value]
        phi = pt.pot.PottingConfig.default().calculate_angle(
            system.balls["0"], system.balls[str(target)], system.table, system.table.pockets[pocket]
        )
        system.strike(V0=1.5, phi=phi, theta=0, a=0, b=offset)
        simulated = pt.simulate(system, engine=adapter._physics_engine(), max_events=100)
        events = simulated.events
        contacts = [
            event for event in events if str(event.event_type) == "ball_ball" and "0" in event.ids
        ]
        potted = [event.ids for event in events if str(event.event_type) == "ball_pocket"]
        if (
            not contacts
            or set(contacts[0].ids) != {"0", str(target)}
            or (str(target), pocket) not in potted
            or any("0" in pair for pair in potted)
            or any("cushion" in str(event.event_type) and "0" in event.ids for event in events)
        ):
            continue
        mapping = adapter._coordinate_mapping(simulated.table, request)
        cue = simulated.balls["0"]
        endpoint = mapping.to_business_point(cue.state.rvw[0])
        if hypot(endpoint.x_mm - center.x_mm, endpoint.y_mm - center.y_mm) > radius - 20:
            continue
        if int(cue.state.s) != 0:
            continue
        if any(
            hypot(
                mapping.to_business_point(
                    simulated.balls[str(ids[ball.track_id])].state.rvw[0]
                ).x_mm
                - ball.center.x_mm,
                mapping.to_business_point(
                    simulated.balls[str(ids[ball.track_id])].state.rvw[0]
                ).y_mm
                - ball.center.y_mm,
            )
            > 10
            for ball in objects
            if ball.track_id != route.target_track_id
        ):
            continue
        passed += 1
    return passed
