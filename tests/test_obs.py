"""Tests for src/eval/obs.py — logging layer.

TDD: tests written before/alongside the implementation.

Coverage:
  - configure_run_logging creates run.log in the given dir.
  - logging a message writes to the file.
  - Repeated configure calls don't duplicate handlers (no duplicated lines).
  - Stream handler targets stderr, not stdout.
  - timed() logs START + DONE.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_eval_logger() -> logging.Logger:
    """Return the 'eval' logger with all handlers removed (clean state)."""
    lg = logging.getLogger("eval")
    lg.handlers.clear()
    return lg


# ---------------------------------------------------------------------------
# configure_run_logging
# ---------------------------------------------------------------------------

class TestConfigureRunLogging:
    def test_creates_run_log_file(self, tmp_path: Path) -> None:
        from src.eval.obs import configure_run_logging
        _fresh_eval_logger()
        configure_run_logging(tmp_path)
        assert (tmp_path / "run.log").exists()

    def test_message_written_to_file(self, tmp_path: Path) -> None:
        from src.eval.obs import configure_run_logging
        _fresh_eval_logger()
        log = configure_run_logging(tmp_path)
        log.info("hello world from test")
        content = (tmp_path / "run.log").read_text()
        assert "hello world from test" in content

    def test_idempotent_no_duplicate_handlers(self, tmp_path: Path) -> None:
        """Calling configure_run_logging twice must not duplicate handlers."""
        from src.eval.obs import configure_run_logging
        _fresh_eval_logger()
        configure_run_logging(tmp_path)
        configure_run_logging(tmp_path)

        log = logging.getLogger("eval")
        # Should have exactly 2 handlers (file + stream), not 4.
        assert len(log.handlers) == 2, (
            f"Expected 2 handlers after two configure calls, got {len(log.handlers)}"
        )

    def test_idempotent_no_duplicate_lines_in_file(self, tmp_path: Path) -> None:
        """Repeated calls must not cause the same log line to appear twice."""
        from src.eval.obs import configure_run_logging
        _fresh_eval_logger()
        configure_run_logging(tmp_path)
        configure_run_logging(tmp_path)

        log = logging.getLogger("eval")
        log.info("unique message xyz789")

        content = (tmp_path / "run.log").read_text()
        count = content.count("unique message xyz789")
        assert count == 1, f"Expected message once, found {count} times"

    def test_stream_handler_targets_stderr_not_stdout(self, tmp_path: Path) -> None:
        from src.eval.obs import configure_run_logging
        _fresh_eval_logger()
        configure_run_logging(tmp_path)

        log = logging.getLogger("eval")
        stream_handlers = [
            h for h in log.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]
        assert stream_handlers, "Expected at least one StreamHandler"
        for h in stream_handlers:
            assert h.stream is sys.stderr, (
                f"Stream handler targets {h.stream!r}, expected sys.stderr"
            )

    def test_file_handler_present_at_debug_level(self, tmp_path: Path) -> None:
        from src.eval.obs import configure_run_logging
        _fresh_eval_logger()
        configure_run_logging(tmp_path)

        log = logging.getLogger("eval")
        file_handlers = [h for h in log.handlers if isinstance(h, logging.FileHandler)]
        assert file_handlers, "Expected at least one FileHandler"
        for h in file_handlers:
            assert h.level == logging.DEBUG

    def test_stream_handler_default_level_info(self, tmp_path: Path) -> None:
        from src.eval.obs import configure_run_logging
        _fresh_eval_logger()
        configure_run_logging(tmp_path, verbose=False)

        log = logging.getLogger("eval")
        stream_handlers = [
            h for h in log.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]
        assert stream_handlers
        assert stream_handlers[0].level == logging.INFO

    def test_verbose_stream_handler_level_debug(self, tmp_path: Path) -> None:
        from src.eval.obs import configure_run_logging
        _fresh_eval_logger()
        configure_run_logging(tmp_path, verbose=True)

        log = logging.getLogger("eval")
        stream_handlers = [
            h for h in log.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]
        assert stream_handlers
        assert stream_handlers[0].level == logging.DEBUG

    def test_returns_eval_logger(self, tmp_path: Path) -> None:
        from src.eval.obs import configure_run_logging
        _fresh_eval_logger()
        log = configure_run_logging(tmp_path)
        assert log.name == "eval"

    def test_debug_message_in_file_not_necessarily_stderr(self, tmp_path: Path) -> None:
        """DEBUG messages go to file even when stream is INFO-only."""
        from src.eval.obs import configure_run_logging
        _fresh_eval_logger()
        configure_run_logging(tmp_path, verbose=False)

        log = logging.getLogger("eval")
        log.debug("debug_only_msg_abc")

        content = (tmp_path / "run.log").read_text()
        assert "debug_only_msg_abc" in content


# ---------------------------------------------------------------------------
# get_logger
# ---------------------------------------------------------------------------

class TestGetLogger:
    def test_returns_child_of_eval(self) -> None:
        from src.eval.obs import get_logger
        child = get_logger("run_eval")
        assert child.name == "eval.run_eval"

    def test_different_names_give_different_loggers(self) -> None:
        from src.eval.obs import get_logger
        a = get_logger("a")
        b = get_logger("b")
        assert a is not b
        assert a.name != b.name


# ---------------------------------------------------------------------------
# timed()
# ---------------------------------------------------------------------------

class TestTimed:
    def test_logs_start_and_done(self, tmp_path: Path) -> None:
        from src.eval.obs import configure_run_logging, timed
        _fresh_eval_logger()
        log = configure_run_logging(tmp_path)

        with timed(log, "my_stage"):
            pass  # instant

        content = (tmp_path / "run.log").read_text()
        assert "START my_stage" in content
        assert "DONE my_stage" in content
        assert " ms" in content

    def test_done_includes_elapsed_ms(self, tmp_path: Path) -> None:
        import time
        from src.eval.obs import configure_run_logging, timed
        _fresh_eval_logger()
        log = configure_run_logging(tmp_path)

        with timed(log, "slow_stage"):
            time.sleep(0.05)  # 50 ms

        content = (tmp_path / "run.log").read_text()
        # Should mention "DONE slow_stage in N ms" with N >= 0
        assert "DONE slow_stage in" in content

    def test_yields_none(self, tmp_path: Path) -> None:
        from src.eval.obs import configure_run_logging, timed
        _fresh_eval_logger()
        log = configure_run_logging(tmp_path)

        result = None
        with timed(log, "noop") as ctx:
            result = ctx
        assert result is None

    def test_done_logged_even_on_exception(self, tmp_path: Path) -> None:
        from src.eval.obs import configure_run_logging, timed
        _fresh_eval_logger()
        log = configure_run_logging(tmp_path)

        with pytest.raises(ValueError):
            with timed(log, "boom_stage"):
                raise ValueError("intentional")

        content = (tmp_path / "run.log").read_text()
        assert "START boom_stage" in content
        assert "DONE boom_stage" in content
