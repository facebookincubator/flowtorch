from .base import Bijector

class Conv1x1Bijector(Bijector):

    def __init__(
            self,
            params_fn: Optional[flowtorch.Lazy] = None,
            *,
            shape: torch.Size,
            context_shape: Optional[torch.Size] = None,
            LU_decompose: bool=False,
            double_solve: bool=False,
    ):
        if params_fn is None:
            params_fn = Conv1x1Params(LU_decompose)

        super().__init__(
            params_fn=params_fn,
            shape=shape,
            context_shape=context_shape,
        )
        self._LU = LU_decompose
        self._double_solve = double_solve

    def _forward(
        self,
        x: torch.Tensor,
        params: Optional[Sequence[torch.Tensor]],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        weight, logdet = params
        z = F.conv2d(input, weight)
        return z, logdet

    def _inverse(
        self,
        y: torch.Tensor,
        params: Optional[Sequence[torch.Tensor]],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        if self._LU:
            p, l, u, logdet = params
            output_view = output.permute(0,2,3,1).unsqueeze(-1)
            if self._double_solve:
                dtype = l.dtype
                l = l.double()
                p = p.double()
                output_view = output_view.double()
                u = u.double()

            z_view = torch.triangular_solve(
                torch.triangular_solve(
                    p.transpose(-1, -2) @ output_view, l, upper=False)[0],
                u, upper=True)[0]

            if self._double_solve:
                z_view = z_view.to(dtype)

            z = z_view.squeeze(-1).permute(0, 3, 1, 2)
            return z, logdet.expand_as(z.sum(self.dims))

        weight, logdet = self.get_weight(output, True)
        z = F.conv2d(output, weight)
        return z, logdet.expand_as(z.sum(self.dims))

    def param_shapes(self, shape: torch.Size) -> Sequence[torch.Size]:
        shape = torch.Size(shape[-3], shape[-3], 1, 1)
        if not self._LU:
            return shape
        else;
            return shape, shape, shape