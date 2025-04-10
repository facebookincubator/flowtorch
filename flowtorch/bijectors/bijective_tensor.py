# Copyright (c) Meta Platforms, Inc

# pyre-unsafe
from collections.abc import Iterator
from typing import Any, Optional, Type, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from flowtorch.bijectors.base import Bijector

from torch import Tensor


class BijectiveTensor(Tensor):
    def __repr__(self, *, tensor_contents: Any = None) -> Any:  # type: ignore
        r_str = super().__repr__().replace("tensor", "bijective_tensor")
        return r_str

    def register(
        self,
        input: Tensor,
        output: Tensor,
        context: Tensor | None,
        bijector: "Bijector",
        log_detJ: Tensor | None,
        mode: str,
    ) -> "BijectiveTensor":
        # pyre-fixme[16]: `BijectiveTensor` has no attribute `_input`.
        self._input = input
        # pyre-fixme[16]: `BijectiveTensor` has no attribute `_output`.
        self._output = output
        # pyre-fixme[16]: `BijectiveTensor` has no attribute `_context`.
        self._context = context
        # pyre-fixme[16]: `BijectiveTensor` has no attribute `_bijector`.
        self._bijector = bijector
        # pyre-fixme[16]: `BijectiveTensor` has no attribute `_log_detJ`.
        self._log_detJ = log_detJ
        # pyre-fixme[16]: `BijectiveTensor` has no attribute `_mode`.
        self._mode = mode

        if not (self.from_forward() or self.from_inverse()):
            raise RuntimeError(
                f"BijectiveTensor mode must be either `'forward'` \
or `'inverse'`. got {self._mode}"
            )

        return self

    @classmethod
    def __torch_function__(
        cls: type["BijectiveTensor"],
        func: Any,
        types: Any,
        args: Any = (),
        kwargs: Any = None,
    ) -> Any | Tensor:
        if kwargs is None:
            kwargs = {}
        # we don't want to create a new BijectiveTensor when summing,
        # calling zeros_like etc.
        types = tuple(Tensor if _type is BijectiveTensor else _type for _type in types)
        return Tensor.__torch_function__(func, types, args, kwargs)

    def check_bijector(self, bijector: "Bijector") -> bool:
        is_bijector = bijector in tuple(self.bijectors())
        return is_bijector

    def bijectors(self) -> Iterator["Bijector"]:
        # pyre-fixme[16]: `BijectiveTensor` has no attribute `_bijector`.
        yield self._bijector
        for parent in self.parents():
            if isinstance(parent, BijectiveTensor):
                yield parent._bijector

    def get_parent_from_bijector(self, bijector: "Bijector") -> Tensor:
        # pyre-fixme[16]: `BijectiveTensor` has no attribute `_bijector`.
        if self._bijector is bijector:
            return self.parent
        for parent in self.parents():
            if not isinstance(parent, BijectiveTensor):
                break
            if parent._bijector is bijector:
                return parent.parent
        raise RuntimeError("bijector not found in flow")

    def check_context(self, context: Tensor | None) -> bool:
        # pyre-fixme[16]: `BijectiveTensor` has no attribute `_context`.
        return self._context is context

    def from_forward(self) -> bool:
        # pyre-fixme[16]: `BijectiveTensor` has no attribute `_mode`.
        return self._mode == "forward"

    def from_inverse(self) -> bool:
        # pyre-fixme[16]: `BijectiveTensor` has no attribute `_mode`.
        return self._mode == "inverse"

    def detach_from_flow(self) -> Tensor:
        # pyre-fixme[16]: `BijectiveTensor` has no attribute `_output`.
        # pyre-fixme[16]: `BijectiveTensor` has no attribute `_input`.
        detached_tensor = self._output if self.from_forward() else self._input
        if isinstance(detached_tensor, BijectiveTensor):
            raise RuntimeError("the detached tensor is an instance of BijectiveTensor.")
        return detached_tensor

    def has_ancestor(self, tensor: Tensor) -> bool:
        if tensor is self:
            return False  # self is no parent of self
        # pyre-fixme[16]: `BijectiveTensor` has no attribute `_input`.
        elif self.from_forward() and self._input is tensor:
            return True
        # pyre-fixme[16]: `BijectiveTensor` has no attribute `_output`.
        elif self.from_inverse() and self._output is tensor:
            return True
        elif self.from_forward() and isinstance(self._input, BijectiveTensor):
            return self._input.has_ancestor(tensor)
        elif self.from_inverse() and isinstance(self._output, BijectiveTensor):
            return self._output.has_ancestor(tensor)
        else:
            return False

    @property
    def log_detJ(self) -> Tensor | None:
        # pyre-fixme[16]: `BijectiveTensor` has no attribute `_log_detJ`.
        return self._log_detJ

    @property
    def parent(self) -> Tensor:
        if self.from_forward():
            # pyre-fixme[16]: `BijectiveTensor` has no attribute `_input`.
            return self._input
        else:
            # pyre-fixme[16]: `BijectiveTensor` has no attribute `_output`.
            return self._output

    def parents(self) -> Iterator[Tensor]:
        child: Tensor | BijectiveTensor = self
        while True:
            assert isinstance(child, BijectiveTensor)
            # pyre-fixme[16]: `Tensor` has no attribute `parent`.
            child = parent = child.parent
            yield parent
            if not isinstance(child, BijectiveTensor):
                break


def to_bijective_tensor(
    x: Tensor,
    y: Tensor,
    context: Tensor | None,
    bijector: "Bijector",
    log_detJ: Tensor | None,
    mode: str = "forward",
) -> BijectiveTensor:
    if mode == "inverse":
        x_bij = BijectiveTensor(x)
        x_bij.register(x, y, context, bijector, log_detJ, mode=mode)
        return x_bij
    elif mode == "forward":
        y_bij = BijectiveTensor(y)
        y_bij.register(x, y, context, bijector, log_detJ, mode=mode)
        return y_bij
    else:
        raise NotImplementedError(
            f"mode {mode} is not supported, must be one of 'forward' or 'inverse'."
        )
