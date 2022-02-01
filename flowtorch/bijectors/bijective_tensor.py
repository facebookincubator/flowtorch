# Copyright (c) Meta Platforms, Inc
from typing import Callable, Optional, Iterator

from torch import Tensor


class BijectiveTensor(Tensor):
    def __repr__(self):
        r_str = super(BijectiveTensor, self).__repr__().replace("tensor", "bijective_tensor")
        return r_str

    def register(
            self,
            input: Tensor,
            output: Tensor,
            context: Optional[Tensor],
            bijector: Callable,
            log_detJ: Tensor,
            mode: str,
    ):
        self._input = input
        self._output = output
        self._context = context
        self._bijector = bijector
        self._log_detJ = log_detJ
        self._mode = mode

        if not (self.from_forward() or self.from_inverse()):
            raise RuntimeError(f"BijectiveTensor mode must be either `'forward'` or `'inverse'`. got {self._mode}")

        return self

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        # we don't want to create a new BijectiveTensor when summing, calling zeros_like etc.
        types = tuple(Tensor if _type is BijectiveTensor else _type for _type in types)
        return Tensor.__torch_function__(func, types, args, kwargs)

    def check_bijector(self, bijector):
        is_bijector = bijector in tuple(self.bijectors())
        return is_bijector

    def bijectors(self) -> Iterator["Bijector"]:
        yield self._bijector
        for parent in self.parents():
            if isinstance(parent, BijectiveTensor):
                yield parent._bijector

    def get_parent_from_bijector(self, bijector):
        if self._bijector is bijector:
            return self.parent
        for parent in self.parents():
            if not isinstance(parent, BijectiveTensor):
                break
            if parent._bijector is bijector:
                return parent.parent
        raise RuntimeError("bijector not found in flow")

    def check_context(self, context):
        return self._context is context

    def from_forward(self) -> bool:
        return self._mode == "forward"

    def from_inverse(self) -> bool:
        return self._mode == "inverse"

    def detach_from_flow(self):
        detached_tensor = self._output if self.from_forward() else self._input
        if isinstance(detached_tensor, BijectiveTensor):
            raise RuntimeError("the detached tensor is an instance of BijectiveTensor.")
        return detached_tensor

    def has_ancestor(self, tensor):
        if tensor is self:
            return False  # self is no parent of self
        elif self.from_forward() and self._input is tensor:
            return True
        elif self.from_inverse() and self._output is tensor:
            return True
        elif self.from_forward() and isinstance(self._input, BijectiveTensor):
            return self._input.has_ancestor(tensor)
        elif self.from_inverse() and isinstance(self._output, BijectiveTensor):
            return self._output.has_ancestor(tensor)
        else:
            return False

    @property
    def log_detJ(self):
        return self._log_detJ

    @property
    def parent(self):
        if self.from_forward():
            return self._input
        else:
            return self._output

    def parents(self) -> Iterator[Tensor]:
        child = self
        while True:
            child = parent = child.parent
            yield parent
            if not isinstance(child, BijectiveTensor):
                break

    # TODO: How to adjust this?
    """
    def log_abs_det_jacobian(self, tensor):
        if self.from_forward() and self.has_ancestor(tensor):
            x = tensor
            parent = self.parent
            ldj = self.layer.log_abs_det_jacobian(parent, self)
            if isinstance(parent, BijectiveTensor):
                ldj = ldj + parent.log_abs_det_jacobian(x)
        elif self.from_inverse() and self.has_ancestor(tensor):
            y = tensor
            parent = self.parent
            ldj = self.layer.log_abs_det_jacobian(self, y)
            if isinstance(parent, BijectiveTensor):
                ldj = ldj + parent.log_abs_det_jacobian(y)
        else:
            raise RuntimeError("Called bijective_tensor.log_abs_det_jacobian(tensor) on a tensor that was not"
                               "part of the flow graph.")
        return ldj
    """


def to_bijective_tensor(
        x: Tensor,
        y: Tensor,
        context: Optional[Tensor],
        bijector: "Bijector",
        log_detJ: Optional[Tensor],
        mode: str = "forward"
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
        raise NotImplementedError(f"mode {mode} is not supported, must be one of 'forward' or 'inverse'.")
