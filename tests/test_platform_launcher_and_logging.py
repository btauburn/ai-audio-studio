import logging
import unittest
from unittest.mock import patch

from dubbing_service.app.core.logging_config import (
    _AsyncioSelectorWriteNoiseFilter,
    _install_asyncio_noise_filter,
)
from tools import platform_launcher


class _DummyBoolVar:
    def __init__(self, value: bool = False) -> None:
        self._value = bool(value)

    def get(self) -> bool:
        return self._value

    def set(self, value: bool) -> None:
        self._value = bool(value)


class _DummyRoot:
    def title(self, *_args, **_kwargs) -> None:
        return None

    def geometry(self, *_args, **_kwargs) -> None:
        return None

    def after(self, *_args, **_kwargs) -> None:
        return None

    def protocol(self, *_args, **_kwargs) -> None:
        return None

    def destroy(self) -> None:
        return None

    def mainloop(self) -> None:
        return None


class TestLoggingNoiseFilter(unittest.TestCase):
    def _record(self, message: str, exc: Exception | None) -> logging.LogRecord:
        record = logging.LogRecord(
            name="asyncio",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg=message,
            args=(),
            exc_info=None,
        )
        if exc is not None:
            record.exc_info = (type(exc), exc, None)
        return record

    def test_filter_drops_known_selector_write_noise(self) -> None:
        filt = _AsyncioSelectorWriteNoiseFilter()
        rec = self._record(
            "Exception in callback _SelectorSocketTransport._write_send()",
            AssertionError("Data should not be empty"),
        )
        self.assertFalse(filt.filter(rec))

    def test_filter_drops_known_proactor_connection_reset_noise(self) -> None:
        filt = _AsyncioSelectorWriteNoiseFilter()
        rec = self._record(
            "Exception in callback _ProactorBasePipeTransport._call_connection_lost(None)",
            ConnectionResetError(10054, "connection reset by peer"),
        )
        self.assertFalse(filt.filter(rec))

    def test_filter_keeps_other_asyncio_errors(self) -> None:
        filt = _AsyncioSelectorWriteNoiseFilter()

        rec1 = self._record(
            "Exception in callback _SelectorSocketTransport._write_send()",
            AssertionError("another assertion"),
        )
        self.assertTrue(filt.filter(rec1))

        rec2 = self._record(
            "another asyncio callback error",
            AssertionError("Data should not be empty"),
        )
        self.assertTrue(filt.filter(rec2))

    def test_install_filter_is_idempotent(self) -> None:
        logger = logging.getLogger("asyncio")
        original_filters = list(logger.filters)
        original_flag = getattr(logger, "_selector_write_noise_filter_installed", None)

        try:
            logger.filters = [
                f for f in logger.filters if not isinstance(f, _AsyncioSelectorWriteNoiseFilter)
            ]
            if hasattr(logger, "_selector_write_noise_filter_installed"):
                delattr(logger, "_selector_write_noise_filter_installed")

            _install_asyncio_noise_filter()
            _install_asyncio_noise_filter()

            count = sum(
                1 for f in logger.filters if isinstance(f, _AsyncioSelectorWriteNoiseFilter)
            )
            self.assertEqual(count, 1)
        finally:
            logger.filters = original_filters
            if original_flag is None:
                if hasattr(logger, "_selector_write_noise_filter_installed"):
                    delattr(logger, "_selector_write_noise_filter_installed")
            else:
                setattr(logger, "_selector_write_noise_filter_installed", original_flag)


class TestPlatformLauncherDefaults(unittest.TestCase):
    def test_default_launcher_disables_reload_mode(self) -> None:
        with patch("tools.platform_launcher.tk.Tk", return_value=_DummyRoot()), patch(
            "tools.platform_launcher.tk.BooleanVar",
            side_effect=lambda value=False: _DummyBoolVar(value),
        ), patch.object(platform_launcher.PlatformLauncherUI, "_build_ui", lambda self: None), patch.object(
            platform_launcher.PlatformLauncherUI,
            "_poll",
            lambda self: None,
        ):
            ui = platform_launcher.PlatformLauncherUI(auto_start=False)

        self.assertFalse(ui._dubbing_reload.get())
        cmd = ui._dubbing_cmd()
        self.assertNotIn("--reload", cmd)
        self.assertEqual(cmd[cmd.index("--host") + 1], "127.0.0.1")

    def test_launcher_requires_explicit_lan_binding(self) -> None:
        with patch("tools.platform_launcher.tk.Tk", return_value=_DummyRoot()), patch(
            "tools.platform_launcher.tk.BooleanVar",
            side_effect=lambda value=False: _DummyBoolVar(value),
        ), patch.object(platform_launcher.PlatformLauncherUI, "_build_ui", lambda self: None), patch.object(
            platform_launcher.PlatformLauncherUI,
            "_poll",
            lambda self: None,
        ):
            ui = platform_launcher.PlatformLauncherUI(auto_start=False, host="0.0.0.0")

        cmd = ui._dubbing_cmd()
        self.assertEqual(cmd[cmd.index("--host") + 1], "0.0.0.0")

    def test_launcher_can_enable_reload_mode_manually(self) -> None:
        with patch("tools.platform_launcher.tk.Tk", return_value=_DummyRoot()), patch(
            "tools.platform_launcher.tk.BooleanVar",
            side_effect=lambda value=False: _DummyBoolVar(value),
        ), patch.object(platform_launcher.PlatformLauncherUI, "_build_ui", lambda self: None), patch.object(
            platform_launcher.PlatformLauncherUI,
            "_poll",
            lambda self: None,
        ):
            ui = platform_launcher.PlatformLauncherUI(auto_start=False)

        ui._dubbing_reload.set(True)
        cmd = ui._dubbing_cmd()
        self.assertIn("--reload", cmd)

    def test_launcher_poll_uses_health_endpoint_for_readiness(self) -> None:
        with patch("tools.platform_launcher.tk.Tk", return_value=_DummyRoot()), patch(
            "tools.platform_launcher.tk.BooleanVar",
            side_effect=lambda value=False: _DummyBoolVar(value),
        ), patch.object(platform_launcher.PlatformLauncherUI, "_build_ui", lambda self: None), patch.object(
            platform_launcher.PlatformLauncherUI,
            "_poll",
            lambda self: None,
        ):
            ui = platform_launcher.PlatformLauncherUI(auto_start=False)

        ui.dubbing.proc = object()
        with patch.object(
            platform_launcher.PlatformLauncherUI,
            "_is_http_ready",
            return_value=True,
        ) as ready_mock, patch("tools.platform_launcher.webbrowser.open") as open_mock:
            ui._poll_open_platform(deadline=0.0)

        ready_mock.assert_called_once_with("http://127.0.0.1:9000/api/v1/ready")
        open_mock.assert_called_once()
        self.assertTrue(ui._auto_browser_opened)


if __name__ == "__main__":
    unittest.main()
