# Copyright (c) Meta Platforms, Inc
import functools

_RECORD_FLOW = True


class _context_manager():
    def __init__(self, value=True):
        self.value = value
        self.prev = []

    def __call__(self, func):
        @functools.wraps(func)
        def decorate_context(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return decorate_context


class set_record_flow_graph(_context_manager):

    def __enter__(self):
        global _RECORD_FLOW
        self.prev.append(_RECORD_FLOW)
        _RECORD_FLOW = self.value

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _RECORD_FLOW
        _RECORD_FLOW = self.prev.pop()


def is_record_flow_graph_enabled():
    return _RECORD_FLOW

_REQUIRES_LOG_DETJ = True

class set_requires_log_detJ(_context_manager):

    def __enter__(self):
        global _REQUIRES_LOG_DETJ
        self.prev.append(_REQUIRES_LOG_DETJ)
        _REQUIRES_LOG_DETJ = self.value

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _REQUIRES_LOG_DETJ
        _REQUIRES_LOG_DETJ = self.prev.pop()


def requires_log_detJ():
    return _REQUIRES_LOG_DETJ
