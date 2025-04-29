import logging

logger = logging.getLogger(__name__)

try:
    from ._tipp import TIPP  # type: ignore

    logger.info("TIPP module loaded")
except ImportError:
    # If the Cython module isn't available, use the pure Python version
    from .tipp import TIPP

    logger.info("TIPP module not available, using pure Python version")
