"""
Timing utilities.

Provides a decorator `measure_time` that adds an 'inference_time' field
to metadata returned by functions or wraps a raw return value into (result, metadata).
"""
import time
import functools
from typing import Any, Tuple, Dict

def measure_time(func):
    """
    Decorator that measures wall-clock time for a function and ensures
    the wrapped function returns a tuple (result, metadata) where metadata contains 'inference_time'.

    Works when the wrapped function returns:
    - result  -> will be converted to (result, {'inference_time': ...})
    - (result, metadata) -> metadata will be updated with inference_time
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Tuple[Any, Dict]:
        start = time.time()
        out = func(*args, **kwargs)
        end = time.time()
        elapsed = end - start

        # If function returned (result, metadata)
        if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
            result, metadata = out
            metadata = dict(metadata)  # copy
            metadata.setdefault("inference_time", elapsed)
            return result, metadata
        else:
            # Only returned result
            result = out
            metadata = {"inference_time": elapsed}
            return result, metadata
    return wrapper
