"""Public cloud services, loaded only when their individual exports are requested.

Importing the physics submodule must not load unrelated adjudication services or
their platform-specific dependencies. Re-exports keep the original objects.
"""

from importlib import import_module

_EXPORT_MODULES = {
    "aggregate_session_statistics": "aggregation",
    "calculate_ideal_path": "ideal_path",
    "compose_coaching_report": "reporting",
    "generate_recommendations": "rules",
    "render_report_document": "reporting",
    "ReportCompositionResult": "reporting",
    "revise_report_document": "reporting",
    "score_session_performance": "scoring",
    "select_typical_shots": "typical_shot_selector",
    "bind_visualization_asset": "visualization",
    "canonical_visualization_asset_bytes": "visualization",
    "generate_visualization_asset": "visualization",
}
__all__ = list(_EXPORT_MODULES)


def __getattr__(name):
    module = _EXPORT_MODULES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module}"), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
