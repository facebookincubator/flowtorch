import functools

RECORD_FLOW = True


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
        global RECORD_FLOW
        self.prev.append(RECORD_FLOW)
        RECORD_FLOW = self.value

    def __exit__(self, exc_type, exc_val, exc_tb):
        global RECORD_FLOW
        RECORD_FLOW = self.prev.pop()


def is_record_flow_graph_enabled():
    return RECORD_FLOW
