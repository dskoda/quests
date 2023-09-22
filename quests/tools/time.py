from time import time
from functools import wraps


def print_log(name, time):
    print(f"[Time] {name} {time * 1000:.3f} ms")


def timetrack(log_fn):
    def _timetrack(fn):
        @wraps(fn)
        def wrapped_fn(*args, **kwargs):
            start_time = time()

            try:
                result = fn(*args, **kwargs)
            finally:
                elapsed_time = time() - start_time

                # log the result
                log_fn(
                    name=fn.__name__,
                    time=elapsed_time,
                )

            return result

        return wrapped_fn

    return _timetrack
