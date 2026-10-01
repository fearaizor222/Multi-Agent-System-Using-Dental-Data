"""Logging setup for the Multi-Agent System."""

import logging
import sys
from src.core.config import settings

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    from rich.logging import RichHandler
    from rich.console import Console
    _console = Console(legacy_windows=False)
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    _console = None


def get_logger(name: str = "dental_multi_agent") -> logging.Logger:
    """Get a configured logger instance."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
        logger.setLevel(level)

        if HAS_RICH and _console:
            handler = RichHandler(
                console=_console,
                rich_tracebacks=True,
                markup=True,
                show_time=True,
                show_path=False
            )
        else:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            handler.setFormatter(formatter)

        logger.addHandler(handler)
        logger.propagate = False

    return logger
