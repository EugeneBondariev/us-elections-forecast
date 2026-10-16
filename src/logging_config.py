from __future__ import annotations

import logging
import sys
import types
from datetime import datetime
from pathlib import Path

import colorlog

_OK_LEVEL = 21
_SKIP_LEVEL = 22
logging.addLevelName(_OK_LEVEL, "OK")
logging.addLevelName(_SKIP_LEVEL, "SKIP")

_status_logger = logging.getLogger(__name__)

_LOG_COLORS = {
    "DEBUG": "cyan",
    "INFO": "light_white",
    "OK": "bold_green",
    "SKIP": "yellow",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "bold_red",
}


def log_ok(label: str, detail: str = "") -> None:
    _status_logger.log(_OK_LEVEL, "[OK]   %s%s", label, f" - {detail}" if detail else "")


def log_skip(label: str, reason: str) -> None:
    _status_logger.log(_SKIP_LEVEL, "[SKIP] %s - %s", label, reason)


def log_fail(label: str, msg: object) -> None:
    _status_logger.error("[FAIL] %s - %s", label, msg)


def setup_cli_logging(level: int = logging.INFO) -> None:
    """Console-only logging with color — for CLI entry points that don't need a log file."""
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(fmt="%(log_color)s%(message)s", log_colors=_LOG_COLORS)
    )
    logging.root.setLevel(level)
    logging.root.handlers = [handler]


def setup_logging(level: int = logging.INFO, log_dir: str = "logs/app") -> Path:
    console_handler = colorlog.StreamHandler()
    console_handler.setFormatter(
        colorlog.ColoredFormatter(
            fmt="%(log_color)s%(asctime)s  %(message)s",
            datefmt="%H:%M:%S",
            log_colors=_LOG_COLORS,
        )
    )

    log_path = Path(log_dir) / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s  %(message)s",
            datefmt="%H:%M:%S",
        )
    )

    logging.root.setLevel(level)
    logging.root.handlers = [console_handler, file_handler]

    def _excepthook(
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_tb: types.TracebackType | None,
    ) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        logging.root.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))

    sys.excepthook = _excepthook
    return log_path
