# Copyright (c) Meta Platforms, Inc
from typing import Callable, Optional

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
    ):
        self._input = input
        self._output = output
        self._context = context
        self._bijector = bijector
        self._log_detJ = log_detJ

        if not (self.from_forward() or self.from_inverse()):
            raise RuntimeError("BijectiveTensor input or output must be self")

        return self

    def check_bijector(self, bijector):
        return self._bijector is bijector

    def check_context(self, context):
        return self._context is context

    def from_forward(self) -> bool:
        return self._output is self

    def from_inverse(self) -> bool:
        return self._input is self

    def detach_from_flow(self):
        detached_tensor = self._output if self.from_forward() else self._input
        # if self.from_forward() and isinstance(self._input_tensor, BijectiveTensor):
        #     self._input_tensor.detach_from_flow()
        # elif self.from_inverse() and isinstance(self._output_tensor, BijectiveTensor):
        #     self._output_tensor.detach_from_flow()
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
        bijector: Callable,
        log_detJ: Optional[Tensor],
        mode: str = "forward"
) -> BijectiveTensor:
    if mode == "inverse":
        x = BijectiveTensor(x)
        x.register(x, y, context, bijector, log_detJ)
        return x
    elif mode == "forward":
        y = BijectiveTensor(y)
        y.register(x, y, context, bijector, log_detJ)
        return y
    else:
        raise NotImplementedError(f"mode {mode} is not supported, must be one of 'forward' or 'inverse'.")
