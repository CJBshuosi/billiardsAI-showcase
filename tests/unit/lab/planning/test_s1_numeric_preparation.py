"""Fresh S1 preparation uses the original engine functions without JIT or a shot."""

import os
import subprocess
import sys
from pathlib import Path


def test_cold_preparation_without_simulation_or_jit_preserves_real_s1_and_parent(tmp_path):
    environment = dict(os.environ)
    environment.update(NUMBA_CACHE_DIR=str(tmp_path / "numba"), MPLCONFIGDIR=str(tmp_path / "mpl"))
    parent_numba = sys.modules.get("numba")
    parent_flag = parent_numba.config.DISABLE_JIT if parent_numba else None
    parent_pooltool = sys.modules.get("pooltool")
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            """
import sys, time
from pathlib import Path
sys.path[:0] = [sys.argv[1], str(Path(sys.argv[1]) / 'src')]
from billiards_ai.lab.field_assistance.snapshot_guidance import _prepare_resources
from billiards_ai.lab.planning.s1_validation import prepare_s1_physics
from numba import config
from numba.core import event
assert 'pooltool' not in sys.modules
before = config.DISABLE_JIT
class Compilations(event.Listener):
    def __init__(self): self.names = []
    def on_start(self, value): self.names.append(value.data['dispatcher'].py_func.__qualname__)
    def on_end(self, value): pass
listener = Compilations()
def no_simulation(frame, event, arg):
    if (event == 'call' and frame.f_code.co_name == 'simulate'
            and frame.f_globals.get('__name__', '').startswith('pooltool.')):
        raise AssertionError('preparation must not simulate')
def numeric_preparer():
    sys.setprofile(no_simulation)
    try:
        prepare_s1_physics()
    finally:
        sys.setprofile(None)
    assert config.DISABLE_JIT is before

def models():
    assert 'pooltool' in sys.modules
    assert config.DISABLE_JIT is before
    return object()
with event.install_listener('numba:compile', listener):
    started = time.monotonic()
    observer, timings = _prepare_resources(models, numeric_preparer)
    prepared_seconds = time.monotonic() - started
    assert type(observer) is object
    assert 0 <= timings['model_preparation_seconds'] <= timings['resource_preparation_seconds']
    from tests.unit.lab.planning.test_s1 import recommend, scene
    started = time.monotonic()
    recommendation = recommend(scene())
    request_seconds = time.monotonic() - started
assert recommendation.route and recommendation.region
assert recommendation.region.physical_witnesses >= 2
assert listener.names == [], listener.names
assert config.DISABLE_JIT is before
assert not list(Path(sys.argv[2]).rglob('*.nbc'))
print('REAL_PREPARED_PYTHON_PHYSICS', prepared_seconds, request_seconds)
""",
            str(Path.cwd()),
            str(tmp_path / "numba"),
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.startswith("REAL_PREPARED_PYTHON_PHYSICS ")
    assert sys.modules.get("numba") is parent_numba
    if parent_numba:
        assert parent_numba.config.DISABLE_JIT is parent_flag
    assert sys.modules.get("pooltool") is parent_pooltool
    print(result.stdout.strip())
