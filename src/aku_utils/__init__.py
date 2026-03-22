import logging.config
from pathlib import Path

logger_path = Path(__file__).resolve().parents[2] / "logging.conf"
logging.config.fileConfig(logger_path, disable_existing_loggers=False)
logger = logging.getLogger("lib")
