"""Central logging setup, shared by Home.py and every page.

Streamlit re-executes each page script in the same process, so this guards
against `configure_logging()` piling up duplicate handlers on every rerun.
"""

import logging
import os


def configure_logging() -> None:
    root = logging.getLogger()
    if root.handlers:
        return

    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
