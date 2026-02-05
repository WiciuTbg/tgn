# logger_setup.py
import logging
import time
from pathlib import Path
from typing import Optional


def setup_logging(
    name: str = "train",
    log_dir: str = "log",
    run_name: Optional[str] = None,
    level: int = logging.INFO,
    console_level: int = logging.WARN,
) -> logging.Logger:
    """
    Configure and return a logger.
    This function is safe to call multiple times in notebooks (it won't duplicate handlers).
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Use a timestamp as a default run name to avoid overwriting log files
    if run_name is None:
        run_name = str(time.time())

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent adding handlers multiple times (common in Colab when re-running cells)
    if not getattr(logger, "_configured", False):
        fmt = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # File handler (detailed logs)
        fh = logging.FileHandler(f"{log_dir}/{run_name}.log")
        fh.setLevel(level)
        fh.setFormatter(fmt)

        # Console handler (usually warnings+ to keep notebook output clean)
        ch = logging.StreamHandler()
        ch.setLevel(console_level)
        ch.setFormatter(fmt)

        logger.addHandler(fh)
        logger.addHandler(ch)

        # Avoid duplicate logs through the root logger
        logger.propagate = False

        logger._configured = True

    return logger
