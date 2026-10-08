import colorlog
import logging
from datetime import datetime
from pathlib import Path


def setup_logging(level: int = logging.INFO, log_dir: str = "logs") -> Path:
    console_handler = colorlog.StreamHandler()
    console_handler.setFormatter(
        colorlog.ColoredFormatter(
            fmt="%(log_color)s%(asctime)s  %(message)s",
            datefmt="%H:%M:%S",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "light_green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
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
    return log_path
