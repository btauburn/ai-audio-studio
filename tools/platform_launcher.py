from __future__ import annotations

import argparse
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import tkinter as tk
import urllib.request
import webbrowser
from pathlib import Path
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _kill_process_tree(
    proc: subprocess.Popen[object],
    *,
    graceful_timeout_seconds: float = 3.0,
) -> None:
    """Stop the managed process group gracefully before using a hard kill.

    On Windows the launcher creates a dedicated process group.  Sending
    CTRL_BREAK lets Uvicorn run the FastAPI lifespan shutdown hook, which in
    turn releases the managed TTS worker and flushes runtime state.  taskkill
    remains a bounded fallback for a wedged process tree.
    """
    if proc.poll() is not None:
        return

    pid = proc.pid
    if pid is None:
        return

    if sys.platform.startswith("win"):
        try:
            proc.send_signal(signal.CTRL_BREAK_EVENT)
            proc.wait(timeout=graceful_timeout_seconds)
            if proc.poll() is not None:
                return
        except (OSError, subprocess.SubprocessError):
            pass
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return

    try:
        proc.terminate()
        proc.wait(timeout=graceful_timeout_seconds)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except OSError:
            pass
    except OSError:
        pass


class ManagedProcess:
    def __init__(self, name: str) -> None:
        self.name = name
        self.proc: subprocess.Popen[str] | None = None
        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self._reader_thread: threading.Thread | None = None

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, cmd: list[str], cwd: Path) -> None:
        if self.is_running():
            return

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("PYTHONUTF8", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")

        popen_kwargs: dict[str, object] = {}
        if sys.platform.startswith("win"):
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True

        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            **popen_kwargs,
        )

        self.proc = proc

        self._reader_thread = threading.Thread(target=self._reader_loop, args=(proc,), daemon=True)
        self._reader_thread.start()

        self.log_queue.put(f"[{self.name}] 已启动 (PID={proc.pid})\n")

    def stop(self, timeout_seconds: float = 15.0) -> None:
        proc = self.proc
        if proc is None:
            return

        if proc.poll() is None:
            self.log_queue.put(f"[{self.name}] 正在停止...\n")
            _kill_process_tree(proc, graceful_timeout_seconds=timeout_seconds)
            try:
                proc.wait(timeout=timeout_seconds)
            except Exception:
                pass

        if proc.poll() is None:
            self.log_queue.put(f"[{self.name}] 停止超时，进程可能仍在运行\n")
            return

        self.proc = None

    def restart(self, cmd: list[str], cwd: Path) -> None:
        self.stop()
        self.start(cmd, cwd)

    def _reader_loop(self, proc: subprocess.Popen[str]) -> None:
        if proc.stdout is None:
            return

        try:
            for line in proc.stdout:
                self.log_queue.put(line)
        finally:
            code = proc.poll()
            self.log_queue.put(f"[{self.name}] 进程已退出 (code={code})\n")


class PlatformLauncherUI:
    def __init__(self, auto_start: bool = True, host: str = "127.0.0.1") -> None:
        if host not in {"127.0.0.1", "0.0.0.0"}:
            raise ValueError("launcher host must be 127.0.0.1 or 0.0.0.0")
        self._host = host
        self.root = tk.Tk()
        self.root.title("配音平台启动器")
        self.root.geometry("800x600")

        self.dubbing = ManagedProcess("Dubbing Service")

        self._auto_browser_opened = False

        # Windows + Python 3.12 下 uvicorn --reload 会切到 SelectorEventLoop，
        # 在高频连接/断开场景可能产生 _write_send 断言噪声。
        # 启动器默认关闭 reload，仅供开发调试时手动开启。
        self._dubbing_reload = tk.BooleanVar(value=False)

        self._build_ui()
        self._poll()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        
        # 自动启动平台
        if auto_start:
            self.root.after(100, self._auto_start)

    def run(self) -> None:
        self.root.mainloop()

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill=tk.X)

        ttk.Button(top, text="启动", command=self._start_dubbing).pack(side=tk.LEFT)
        ttk.Button(top, text="停止", command=self._stop_dubbing).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(top, text="重启", command=self._restart_dubbing).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(top, text="打开平台", command=lambda: webbrowser.open("http://127.0.0.1:9000/web/")).pack(
            side=tk.LEFT, padx=(16, 0)
        )

        ttk.Separator(self.root).pack(fill=tk.X, padx=10, pady=8)

        options = ttk.Frame(self.root, padding=10)
        options.pack(fill=tk.X)

        self.dubbing_status = ttk.Label(options, text="状态：未启动", font=("", 10, "bold"))
        self.dubbing_status.pack(anchor=tk.W)

        ttk.Checkbutton(
            options,
            text="开发模式：代码变更自动重载（Windows 下可能出现 asyncio 噪声日志）",
            variable=self._dubbing_reload,
        ).pack(anchor=tk.W, pady=(6, 0))

        ttk.Label(
            options,
            text="提示：TTS 服务可在配音平台的「平台配置」页面中启动和管理",
            foreground="gray",
        ).pack(anchor=tk.W, pady=(10, 0))

        ttk.Separator(self.root).pack(fill=tk.X, padx=10, pady=8)

        self.dubbing_log = ScrolledText(self.root, wrap=tk.WORD)
        self.dubbing_log.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

    def _dubbing_cmd(self) -> list[str]:
        if self._dubbing_reload.get():
            return [
                sys.executable,
                "-u",
                "-m",
                "uvicorn",
                "dubbing_service.app.main:app",
                "--host",
                self._host,
                "--port",
                "9000",
                "--reload",
                "--reload-dir",
                str(PROJECT_ROOT / "dubbing_service"),
            ]

        return [
            sys.executable,
            "-u",
            "-m",
            "uvicorn",
            "dubbing_service.app.main:app",
            "--host",
            self._host,
            "--port",
            "9000",
        ]

    def _start_dubbing(self) -> None:
        self.dubbing.start(self._dubbing_cmd(), PROJECT_ROOT)

    def _auto_start(self) -> None:
        """自动启动平台并打开浏览器"""
        self._start_dubbing()
        self.root.after(200, lambda: self._open_platform_when_ready(timeout_seconds=60.0))

    @staticmethod
    def _is_http_ready(url: str, timeout_seconds: float = 0.6) -> bool:
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(url, timeout=timeout_seconds) as resp:
                status = getattr(resp, "status", 200)
                return 200 <= int(status) < 400
        except Exception:
            return False

    def _open_platform_when_ready(self, timeout_seconds: float = 60.0) -> None:
        if self._auto_browser_opened:
            return

        deadline = time.time() + timeout_seconds
        self._poll_open_platform(deadline)

    def _poll_open_platform(self, deadline: float) -> None:
        if self._auto_browser_opened:
            return

        url = "http://127.0.0.1:9000/web/"
        ready_url = "http://127.0.0.1:9000/api/v1/ready"

        if self._is_http_ready(ready_url):
            self._auto_browser_opened = True
            webbrowser.open(f"{url}?ts={int(time.time())}")
            return

        if not self.dubbing.is_running():
            self.dubbing.log_queue.put("[Launcher] 未检测到平台服务启动，已取消自动打开浏览器\n")
            return

        if time.time() >= deadline:
            self._auto_browser_opened = True
            self.dubbing.log_queue.put("[Launcher] 等待平台就绪超时，仍将尝试打开浏览器\n")
            webbrowser.open(f"{url}?ts={int(time.time())}")
            return

        self.root.after(300, lambda: self._poll_open_platform(deadline))

    def _stop_dubbing(self) -> None:
        self.dubbing.stop()

    def _restart_dubbing(self) -> None:
        self.dubbing.restart(self._dubbing_cmd(), PROJECT_ROOT)

    def _poll(self) -> None:
        self._drain_logs(self.dubbing, self.dubbing_log)
        self.dubbing_status.configure(text=f"状态：{'运行中' if self.dubbing.is_running() else '未启动'}")
        self.root.after(120, self._poll)

    @staticmethod
    def _drain_logs(proc: ManagedProcess, widget: ScrolledText) -> None:
        changed = False
        while True:
            try:
                line = proc.log_queue.get_nowait()
            except queue.Empty:
                break
            widget.insert(tk.END, line)
            changed = True

        if changed:
            widget.see(tk.END)

    def _on_close(self) -> None:
        self._stop_dubbing()
        self.root.after(200, self.root.destroy)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the local dubbing studio")
    parser.add_argument(
        "--host",
        choices=("127.0.0.1", "0.0.0.0"),
        default="127.0.0.1",
        help="bind address; use 0.0.0.0 only on a firewall-protected trusted LAN",
    )
    args = parser.parse_args()

    if sys.platform.startswith("win"):
        try:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        except Exception:
            pass
        try:
            signal.signal(signal.SIGBREAK, signal.SIG_IGN)
        except Exception:
            pass

    ui = PlatformLauncherUI(host=args.host)
    try:
        ui.run()
    except KeyboardInterrupt:
        ui._stop_dubbing()


if __name__ == "__main__":
    main()
