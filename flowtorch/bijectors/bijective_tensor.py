from typing import Callable

from torch import Tensor


class BijectiveTensor(Tensor):
    def __repr__(self):
        r_str = super(BijectiveTensor, self).__repr__().replace("tensor", "bijective_tensor")
        return r_str

    def register_layer(
            self,
            layer: Callable,
            input: Tensor,
            output: Tensor
    ):
        self._input_tensor = input
        self._output_tensor = output
        if not (self.from_forward() or self.from_inverse()):
            raise RuntimeError("BijectiveTensor input or output must be self")
        self._layer = layer
        return self

    def check_layer(self, layer):
        return self._layer is layer

    def from_forward(self) -> bool:
        return self._output_tensor is self

    def from_inverse(self) -> bool:
        return self._input_tensor is self

    def detach_from_flow(self):
        detached_tensor = self._output_tensor if self.from_forward() else self._input_tensor
        # if self.from_forward() and isinstance(self._input_tensor, BijectiveTensor):
        #     self._input_tensor.detach_from_flow()
        # elif self.from_inverse() and isinstance(self._output_tensor, BijectiveTensor):
        #     self._output_tensor.detach_from_flow()
        return detached_tensor

    def has_ancestor(self, tensor):
        if tensor is self:
            return False  # self is no parent of self
        elif self.from_forward() and self._input_tensor is tensor:
            return True
        elif self.from_inverse() and self._output_tensor is tensor:
            return True
        elif self.from_forward() and isinstance(self._input_tensor, BijectiveTensor):
            return self._input_tensor.has_ancestor(tensor)
        elif self.from_inverse() and isinstance(self._output_tensor, BijectiveTensor):
            return self._output_tensor.has_ancestor(tensor)
        else:
            return False

    @property
    def parent(self):
        if self.from_forward():
            return self._input_tensor
        else:
            return self._output_tensor

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


def to_bijective_tensor(x: Tensor, y: Tensor, layer: Callable, mode: str = "forward"):
    if mode == "inverse":
        x = BijectiveTensor(x)
        x.register_layer(layer, x, y)
        return x
    elif mode == "forward":
        y = BijectiveTensor(y)
        y.register_layer(layer, x, y)
        return y
    else:
        raise NotImplementedError(f"mode {mode} is not supported, must be one of 'forward' or 'inverse'.")
