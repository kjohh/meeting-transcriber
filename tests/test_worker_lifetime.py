"""The whisper worker process must not outlive the app.

Starts a LocalWhisperWorker from a throwaway parent process, kills that parent
with SIGKILL (no cleanup code runs), and checks the worker exits on its own.
Skipped when the bundled model hasn't been fetched (scripts/build-app.sh).
"""
import os
import signal
import subprocess
import sys
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "build-cache", "ggml-small-q5_1.bin")

PARENT = f"""
import sys, time
sys.path.insert(0, {ROOT!r})
import app as A
if __name__ == "__main__":
    w = A.LocalWhisperWorker({MODEL!r})
    w.start()
    print(" ".join(str(p.pid) for p in w._processes()), flush=True)
    time.sleep(120)
"""


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie still answers kill(0); treat it as gone.
    state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)],
                           capture_output=True, text=True).stdout.strip()
    return bool(state) and not state.startswith("Z")


@unittest.skipUnless(os.path.exists(MODEL), "bundled model not fetched")
class WorkerLifetime(unittest.TestCase):
    def test_worker_exits_when_app_is_killed(self):
        parent = subprocess.Popen([sys.executable, "-c", PARENT],
                                  stdout=subprocess.PIPE, text=True)
        pids: list[int] = []
        try:
            pids = [int(p) for p in parent.stdout.readline().split()]
            self.assertTrue(pids, "worker did not start")
            self.assertTrue(all(_alive(p) for p in pids))
            parent.send_signal(signal.SIGKILL)
            parent.wait()
            deadline = time.time() + 8
            while time.time() < deadline and any(_alive(p) for p in pids):
                time.sleep(0.25)
            left = [p for p in pids if _alive(p)]
            self.assertEqual(left, [], f"worker outlived the app: {left}")
        finally:
            if parent.poll() is None:
                parent.kill()
            for p in pids:  # don't leave a regression's orphan behind
                if _alive(p):
                    os.kill(p, signal.SIGKILL)


if __name__ == "__main__":
    unittest.main()
