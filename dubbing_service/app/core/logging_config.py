"""Application-wide logging configuration.

提供统一的日志格式、级别控制和文件轮转。
日志文件输出到 dubbing_service/logs/ 目录。
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path


class _AsyncioSelectorWriteNoiseFilter(logging.Filter):
    """过滤 asyncio 在 Windows 下已知且无害的连接收尾噪声报错。"""

    _TARGET_MSG = "Exception in callback _SelectorSocketTransport._write_send()"
    _TARGET_ASSERT = "Data should not be empty"
    _PROACTOR_TARGET_MSG = "Exception in callback _ProactorBasePipeTransport._call_connection_lost(None)"

    @staticmethod
    def _is_known_connection_reset(exc: BaseException) -> bool:
        if not isinstance(exc, ConnectionResetError):
            return False

        winerror = getattr(exc, "winerror", None)
        errno = getattr(exc, "errno", None)
        first_arg = exc.args[0] if exc.args else None
        return winerror == 10054 or errno == 10054 or first_arg == 10054

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name != "asyncio":
            return True

        message = record.getMessage()
        if not record.exc_info:
            return True

        exc = record.exc_info[1]
        if self._TARGET_MSG in message and isinstance(exc, AssertionError) and self._TARGET_ASSERT in str(exc):
            return False

        if self._PROACTOR_TARGET_MSG in message and self._is_known_connection_reset(exc):
            return False

        return True


def _install_asyncio_noise_filter() -> None:
    asyncio_logger = logging.getLogger("asyncio")
    if getattr(asyncio_logger, "_selector_write_noise_filter_installed", False):
        return
    asyncio_logger.addFilter(_AsyncioSelectorWriteNoiseFilter())
    setattr(asyncio_logger, "_selector_write_noise_filter_installed", True)


def setup_logging(
    level: str | None = None,
    log_dir: str | Path | None = None,
) -> None:
    """Configure application-wide logging.

    Args:
        level: 日志级别，默认从 ``DUBBING_LOG_LEVEL`` 环境变量读取，
               未设置时使用 ``INFO``。
        log_dir: 日志文件目录，默认 ``dubbing_service/logs/``。
    """

    if level is None:
        level = os.getenv("DUBBING_LOG_LEVEL", "INFO")
    log_level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    # 避免重复配置（例如 reload 时）
    if root.handlers:
        _install_asyncio_noise_filter()
        return
    root.setLevel(log_level)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler ──────────────────────────────────────────────
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(log_level)
    console.setFormatter(fmt)
    root.addHandler(console)

    # ── Rotating file handler ────────────────────────────────────────
    if log_dir is None:
        log_dir = Path(__file__).resolve().parents[2] / "logs"
    else:
        log_dir = Path(log_dir)

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "dubbing_service.log",
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
    except Exception:
        # 如果文件日志创建失败（权限等），退化为仅控制台输出
        logging.warning("无法创建日志文件，仅使用控制台输出", exc_info=True)

    # ── 降低第三方库噪音 ─────────────────────────────────────────────
    for noisy in (
        "urllib3",
        "uvicorn.access",
        "httpcore",
        "httpx",
        "watchfiles",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _install_asyncio_noise_filter()
