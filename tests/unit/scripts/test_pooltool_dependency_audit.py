import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[3] / "scripts" / "algorithm" / "pooltool_dependency_audit.py"


def test_script_help_is_available_without_network() -> None:
    specification = importlib.util.spec_from_file_location("pooltool_dependency_audit", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    parser = module._parser()
    assert "metadata-only pooltool" in parser.description
