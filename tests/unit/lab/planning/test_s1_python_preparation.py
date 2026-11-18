"""S1's interpreter preparation must be local to a fresh owned worker."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


def run_child(code):
    environment = dict(os.environ)
    environment.pop("NUMBA_DISABLE_JIT", None)
    return subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            "import sys; sys.path[:0] = [sys.argv[1], sys.argv[1] + '/src'];\n" + code,
            str(Path.cwd()),
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
    )


@pytest.mark.parametrize(
    "name", ["pooltool", "pooltool.partial", "billiards_ai.cloud.engine.pooltool_adapter"]
)
def test_preloaded_physics_is_rejected_before_import_or_configuration_changes(name):
    result = run_child(f"""
import builtins
from unittest.mock import patch
from numba import config
from billiards_ai.lab.planning.s1_validation import prepare_s1_physics
before = config.DISABLE_JIT
sys.modules[{name!r}] = object()
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'billiards_ai.cloud.engine.pooltool_adapter':
        raise AssertionError('preloaded physics must be rejected before import')
    return original_import(name, *args, **kwargs)
with patch('builtins.__import__', guarded_import):
    try:
        prepare_s1_physics()
    except RuntimeError as error:
        assert 'fresh' in str(error)
    else:
        raise AssertionError('preloaded physics was accepted')
assert config.DISABLE_JIT is before
""")
    assert result.returncode == 0, result.stdout + result.stderr


def test_import_failure_restores_temporary_jit_state_without_loading_models():
    result = run_child("""
import builtins
from unittest.mock import patch
from numba import config
from numba.core.compiler_lock import global_compiler_lock
from billiards_ai.lab.planning.s1_validation import prepare_s1_physics
before = config.DISABLE_JIT
original_import = builtins.__import__
def failing_import(name, *args, **kwargs):
    if name == 'billiards_ai.cloud.engine.pooltool_adapter':
        assert config.DISABLE_JIT == 1
        assert global_compiler_lock.is_locked()
        raise OSError('controlled physics import failure')
    return original_import(name, *args, **kwargs)
with patch('builtins.__import__', failing_import):
    try:
        prepare_s1_physics()
    except OSError as error:
        assert str(error) == 'controlled physics import failure'
    else:
        raise AssertionError('physics failure was swallowed')
assert config.DISABLE_JIT is before
assert not global_compiler_lock.is_locked()
""")
    assert result.returncode == 0, result.stdout + result.stderr
