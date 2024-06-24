import functools
import sys
import time
from collections.abc import Callable
from enum import Enum
from typing import Any, TextIO, TypeVar

_S = TypeVar("_S", bound="StrEnum")


# backport str enum(https://github.com/clbarnes/backports.strenum/blob/main/backports/strenum/strenum.py)
class StrEnum(str, Enum):
    """Enum where members are also (and must be) strings"""

    def __new__(cls: type[_S], *values: str) -> _S:
        if len(values) > 3:
            msg = f"too many arguments for str(): {values!r}"
            raise TypeError(msg)
        if len(values) == 1:
            # it must be a string
            if not isinstance(values[0], str):
                msg = f"{values[0]!r} is not a string"
                raise TypeError(msg)
        if len(values) >= 2:
            # check that encoding argument is a string
            if not isinstance(values[1], str):
                msg = f"encoding must be a string, not {values[1]!r}"
                raise TypeError(msg)
        if len(values) == 3:
            # check that errors argument is a string
            if not isinstance(values[2], str):
                raise TypeError("errors must be a string, not %r" % (values[2]))
        value = str(*values)
        member = str.__new__(cls, value)
        member._value_ = value
        return member

    __str__ = str.__str__

    @staticmethod
    def _generate_next_value_(
        name: str,
        start: int,
        count: int,
        last_values: list[Any],
    ) -> str:
        """Return the lower-cased version of the member name."""
        return name.lower()


def flatten_dict(
    d: dict[str, Any],
    parent_key: str = "",
    sep: str = "__",
    verify_keys_do_not_have_sep: bool = True,
) -> dict:
    """Flatten a nested dictionary"""
    items = []
    for k, v in d.items():
        if verify_keys_do_not_have_sep and sep in k:
            msg = f"Separator {sep!r} is not allowed in keys. Found in {k!r}"
            raise ValueError(
                msg,
            )
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(
                flatten_dict(v, new_key, sep, verify_keys_do_not_have_sep).items(),
            )
        else:
            items.append((new_key, v))
    return dict(items)


F = TypeVar("F", bound=Callable[..., Any])


def retry_func_on_exception(
    exception: type[Exception],
    max_retries: int = 8,
    linear_backoff_factor: float = 3,
    exponential_backoff_factor: float = 2,
    print_output_stream: TextIO = sys.stderr,
    extra_message: str = "",
) -> Callable[[F], F]:
    """
    A decorator that retries a function on a given exception.

    The wait time between retries is calculated as:
    wait_time = exponential_backoff_factor**retries + linear_backoff_factor * retries
    """

    def decorator_retry(func: F) -> F:
        @functools.wraps(func)
        def wrapper_retry(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except exception as e:
                    retries += 1
                    if retries >= max_retries:
                        raise
                    wait_time = (
                        exponential_backoff_factor**retries
                        + linear_backoff_factor * retries
                    )
                    if print_output_stream:
                        print(
                            f"{e.__class__.__name__} occurred. "
                            f"Retrying in {wait_time:.3f} seconds..."
                            + extra_message,
                            file=print_output_stream,
                        )
                    time.sleep(wait_time)

        return wrapper_retry

    return decorator_retry
