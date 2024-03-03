from functools import wraps
from time import perf_counter, time


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


class Timer:
    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.start

    def __str__(self):
        return f"Time: {self.time:.3f} seconds"
