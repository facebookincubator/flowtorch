# Copyright (c) Meta Platforms, Inc
import functools
from typing import Any, Callable, List, Sequence

_RECORD_FLOW = True


class _context_manager:
    def __init__(self, value: bool = True) -> None:
        self.value = value
        self.prev: List[bool] = []

    def __call__(self, func: Callable) -> Any:
        @functools.wraps(func)
        def decorate_context(*args: Any, **kwargs: Sequence[Any]) -> Any:
            with self:
                return func(*args, **kwargs)

        return decorate_context

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass


class set_record_flow_graph(_context_manager):
    def __enter__(self) -> None:
        global _RECORD_FLOW
        self.prev.append(_RECORD_FLOW)
        _RECORD_FLOW = self.value

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        global _RECORD_FLOW
        _RECORD_FLOW = self.prev.pop()


def is_record_flow_graph_enabled() -> bool:
    return _RECORD_FLOW


_REQUIRES_LOG_DETJ = True


class set_requires_log_detJ(_context_manager):
    def __enter__(self) -> None:
        global _REQUIRES_LOG_DETJ
        self.prev.append(_REQUIRES_LOG_DETJ)
        _REQUIRES_LOG_DETJ = self.value

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        global _REQUIRES_LOG_DETJ
        _REQUIRES_LOG_DETJ = self.prev.pop()


def requires_log_detJ() -> bool:
    return _REQUIRES_LOG_DETJ
