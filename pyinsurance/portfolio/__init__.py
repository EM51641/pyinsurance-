try:
    from .tipp_ import TIPP
except ImportError:
    # If the Cython module isn't available, use the pure Python version
    from .tipp__ import TIPP

__all__ = ["TIPP"]
