import sys
from pathlib import Path
from loguru import logger


def setup_logger(level: str = "DEBUG") -> None:
    logger.remove()

    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    logger.add(
        logs_dir / "app.log",
        rotation="5 MB",
        retention="10 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        backtrace=True,
        diagnose=True,
        enqueue=True,
    )

    logger.add(
        logs_dir / "errors.log",
        rotation="5 MB",
        retention="30 days",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        backtrace=True,
        diagnose=True,
        enqueue=True,
    )

    logger.info(f"Logger initialized with level: {level}")
