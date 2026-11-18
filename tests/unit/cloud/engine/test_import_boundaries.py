"""Physics loading must not eagerly load unrelated cloud adjudication services."""

import importlib
import subprocess
import sys
from pathlib import Path

EXPORTS = {
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


def cold_import(code):
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code, str(Path("src").resolve())],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_engine_package_import_does_not_require_unix_fcntl():
    cold_import("""
import importlib.abc, sys
sys.path.insert(0, sys.argv[1])
class NoFcntl(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'fcntl':
            raise ModuleNotFoundError('fcntl unavailable on Windows', name=fullname)
sys.meta_path.insert(0, NoFcntl())
import billiards_ai.cloud.engine
assert 'billiards_ai.cloud.perception' not in sys.modules
assert 'billiards_ai.cloud.engine.aggregation' not in sys.modules
""")


def test_real_physics_adapter_cold_import_does_not_load_cloud_perception():
    cold_import("""
import importlib.abc, sys
sys.path.insert(0, sys.argv[1])
class NoCloudPerception(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if (fullname == 'billiards_ai.cloud.perception'
                or fullname.startswith('billiards_ai.cloud.perception.')):
            raise ImportError('unrelated adjudication dependency forbidden')
sys.meta_path.insert(0, NoCloudPerception())
from billiards_ai.cloud.engine.pooltool_adapter import PooltoolIdealPathAdapter, pt
assert pt is not None, 'the real installed engine must remain available'
PooltoolIdealPathAdapter()._require_engine()
assert 'billiards_ai.cloud.perception' not in sys.modules
assert 'billiards_ai.cloud.engine.aggregation' not in sys.modules
""")


def test_public_cloud_engine_exports_keep_exact_original_object_identity():
    package = importlib.import_module("billiards_ai.cloud.engine")
    assert package.__all__ == list(EXPORTS)
    namespace = {}
    exec("from billiards_ai.cloud.engine import *", namespace)
    for name, module_name in EXPORTS.items():
        original = getattr(
            importlib.import_module("billiards_ai.cloud.engine." + module_name), name
        )
        assert getattr(package, name) is original
        assert namespace[name] is original
    assert set(EXPORTS).issubset(dir(package))
