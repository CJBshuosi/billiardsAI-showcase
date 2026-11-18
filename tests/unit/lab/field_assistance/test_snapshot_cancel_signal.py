"""A dead guidance child cannot make the recording stop request wait on its lock."""

import json
import subprocess
import sys
from pathlib import Path


def test_stop_request_returns_after_a_child_abandons_the_actual_cancel_signal(tmp_path):
    program = tmp_path / "abandoned_cancel_signal.py"
    program.write_text(
        """
import json
import multiprocessing
import sys
import threading
import time
from pathlib import Path

sys.path[:0] = [sys.argv[1], str(Path(sys.argv[1]) / 'src')]
from billiards_ai.lab.field_assistance.snapshot_guidance import SnapshotGuidance
from billiards_ai.shared.calibration.homography import calibrate


def abandon_cancel_lock(cancel_signal, connection):
    # A killed child can leave Event.is_set()'s internal condition acquired.
    # Exercise that real primitive when present; a lock-free signal has no
    # critical section to abandon, but follows the same child exit lifecycle.
    condition = getattr(cancel_signal, '_cond', None)
    if condition is not None:
        condition.acquire()
    connection.send('entered')
    connection.close()
    # Deliberately exit without releasing the acquired condition.


if __name__ == '__main__':
    runner = SnapshotGuidance(
        hub=None,
        session_id='cancel-lock-test',
        source_epoch_id='epoch1',
        calibration=calibrate('test', [(0, 239), (319, 239), (319, 0), (0, 0)]),
        observer_factory=object,
        journal_path=Path(sys.argv[2]) / 'guidance.json',
    )
    runner._start_worker()
    cancel_signal = runner._cancel_event
    try:
        deadline = time.monotonic() + 5
        while runner.projection()['model_state'] == 'preparing' and time.monotonic() < deadline:
            runner._poll_preparation()
            time.sleep(.01)
        assert runner.projection()['model_state'] == 'ready'
    finally:
        # Reap the real owned worker before fault injection: even a failing
        # stop request must not leave an orphan when the outer test times out.
        runner._cancel()
    assert runner._process is None

    context = multiprocessing.get_context('spawn')
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=abandon_cancel_lock, args=(cancel_signal, child))
    process.start()
    child.close()
    try:
        assert parent.poll(5) and parent.recv() == 'entered'
        process.join(5)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)
        assert not process.is_alive()
        process.close()
        parent.close()

    # Reproduce request_stop retaining a signal reference while the guidance
    # worker is being reaped. No child process remains alive at this point.
    runner._cancel_event = cancel_signal
    returned = threading.Event()
    errors = []
    def stop_request():
        try:
            runner.request_stop()
        except BaseException as error:
            errors.append(repr(error))
        else:
            returned.set()
    request = threading.Thread(target=stop_request, daemon=True)
    request.start()
    request.join(.5)
    print(json.dumps(dict(stop_returned=returned.is_set(), stop_errors=errors,
                         stop_set=runner._stop.is_set(),
                         owned_worker_reaped=runner._process is None)), flush=True)
""",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-I", "-B", str(program), str(Path.cwd()), str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["owned_worker_reaped"] and report["stop_set"]
    assert report["stop_errors"] == []
    assert report["stop_returned"], "recording stop waited on a lock abandoned by a dead child"
