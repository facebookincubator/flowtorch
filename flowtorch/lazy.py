# Copyright (c) Meta Platforms, Inc

import inspect
from collections import OrderedDict
from typing import Any, Mapping, Tuple


# TODO: Move functions to flowtorch.utils?
def partial_signature(
    sig: inspect.Signature, *args: Any, **kwargs: Any
) -> Tuple[inspect.Signature, Mapping[str, Any]]:
    """
    Given an inspect.Signature object and a dictionary of (name, val) pairs,
    bind the names to the signature and return a new modified signature
    """
    bindings = dict(sig.bind_partial(*args, **kwargs).arguments)

    old_parameters = sig.parameters
    new_parameters = OrderedDict()

    for param_name in old_parameters:
        if param_name not in bindings:
            new_parameters[param_name] = old_parameters[param_name]

    bound_sig = sig.replace(parameters=list(new_parameters.values()))

    return bound_sig, bindings


def count_unbound(sig: inspect.Signature) -> int:
    return len(
        [p for p, v in sig.parameters.items() if v.default is inspect.Parameter.empty]
    )


class LazyMeta(type):
    def __call__(cls: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Intercept instance creation
        """
        # Special behaviour for Lazy class
        if cls.__qualname__ == "Lazy":
            lazy_cls = args[0]
            args = args[1:]
        else:
            lazy_cls = cls

        # Remove first argument (i.e., self) from signature of class' initializer
        sig = inspect.signature(lazy_cls.__init__)
        new_parameters = OrderedDict(
            [(k, v) for idx, (k, v) in enumerate(sig.parameters.items()) if idx != 0]
        )
        sig = sig.replace(parameters=list(new_parameters.values()))

        # Attempt binding arguments to initializer
        bound_sig, bindings = partial_signature(sig, *args, **kwargs)

        # If there are no unbound arguments then instantiate class
        if not count_unbound(bound_sig):
            return type.__call__(lazy_cls, *args, **kwargs)

        # Otherwise, return Lazy instance
        else:
            return type.__call__(Lazy, lazy_cls, bindings, sig, bound_sig)


class Lazy(metaclass=LazyMeta):
    """
    Represents delayed instantiation of a class.
    """

    def __init__(
        self,
        cls: Any,
        bindings: Mapping[str, Any],
        sig: inspect.Signature,
        bound_sig: inspect.Signature,
    ):
        self.cls = cls
        self.bindings = bindings
        self.sig = sig
        self.bound_sig = bound_sig

    def __repr__(self) -> str:
        return f"Lazy(cls={self.cls.__name__}, bindings={self.bindings})"

    def __call__(self, *args: Any, **kwargs: Any) -> "Lazy":
        """
        Apply additional bindings
        """
        new_bindings = dict(self.bound_sig.bind_partial(*args, **kwargs).arguments)
        new_bindings.update(self.bindings)

        # Update args and kwargs
        new_args = []
        new_kwargs = {}
        for n, p in self.sig.parameters.items():
            if n in new_bindings:
                if p.kind == inspect.Parameter.POSITIONAL_ONLY:
                    new_args.append(new_bindings[n])

                else:
                    new_kwargs[n] = new_bindings[n]

        # Attempt object creation
        return Lazy(self.cls, *new_args, **new_kwargs)
