from __future__ import annotations

import signal
import subprocess
from unittest.mock import Mock, patch

from tools.platform_launcher import ManagedProcess, _kill_process_tree


def test_windows_shutdown_prefers_ctrl_break() -> None:
    proc = Mock()
    proc.pid = 1234
    proc.poll.side_effect = [None, 0]

    with patch("tools.platform_launcher.sys.platform", "win32"), patch(
        "tools.platform_launcher.subprocess.run"
    ) as hard_kill:
        _kill_process_tree(proc, graceful_timeout_seconds=0.1)

    proc.send_signal.assert_called_once_with(signal.CTRL_BREAK_EVENT)
    proc.wait.assert_called_once_with(timeout=0.1)
    hard_kill.assert_not_called()


def test_windows_shutdown_falls_back_to_taskkill_after_timeout() -> None:
    proc = Mock()
    proc.pid = 5678
    proc.poll.return_value = None
    proc.wait.side_effect = subprocess.TimeoutExpired(cmd="uvicorn", timeout=0.1)

    with patch("tools.platform_launcher.sys.platform", "win32"), patch(
        "tools.platform_launcher.subprocess.run"
    ) as hard_kill:
        _kill_process_tree(proc, graceful_timeout_seconds=0.1)

    hard_kill.assert_called_once()
    assert hard_kill.call_args.args[0] == ["taskkill", "/PID", "5678", "/T", "/F"]


def test_managed_process_forwards_graceful_timeout() -> None:
    managed = ManagedProcess("test")
    proc = Mock()
    proc.poll.side_effect = [None, 0, 0]
    managed.proc = proc

    with patch("tools.platform_launcher._kill_process_tree") as stop_tree:
        managed.stop(timeout_seconds=12.0)

    stop_tree.assert_called_once_with(proc, graceful_timeout_seconds=12.0)
    proc.wait.assert_called_once_with(timeout=12.0)
    assert managed.proc is None
