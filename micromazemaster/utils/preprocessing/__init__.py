import os
from datetime import datetime
from pathlib import Path

from micromazemaster.utils.logging import logger


def create_tmp_dir() -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S.%f")
    CWD = Path(os.getcwd())
    WORKING_DIR = CWD.joinpath("local_data", timestamp)
    try:
        WORKING_DIR.mkdir(parents=True)
        logger.debug(f"Working directory created: {WORKING_DIR}")

    except OSError as e:
        logger.error(f"Failed to create working directory: {e}")
        exit(1)

    return WORKING_DIR
