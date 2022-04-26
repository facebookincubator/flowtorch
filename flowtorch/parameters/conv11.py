from typing import Optional, Sequence, Union, List

import torch
from flowtorch.parameters import Parameters
from scipy import linalg as scipy_linalg  # type: ignore
from torch import nn
from torch.nn import functional as F


def _pixels(tensor: torch.Tensor) -> int:
    return int(tensor.shape[-2] * tensor.shape[-1])


def _sum(
    tensor: torch.Tensor,
    dim: Optional[Union[int, List[int]]] = None,
    keepdim: bool = False,
) -> torch.Tensor:
    if dim is None:
        # sum up all dim
        return torch.sum(tensor)
    else:
        if isinstance(dim, int):
            dims = [dim]
        else:
            dims = dim
        dims = sorted(dims)
        for d in dims:
            tensor = tensor.sum(dim=d, keepdim=True)
        if not keepdim:
            for i, d in enumerate(dims):
                tensor.squeeze_(d - i)
        return tensor


class Conv1x1Params(Parameters):
    BIAS_SOFTPLUS = 0.54

    def __init__(
        self,
        LU_decompose: bool,
        param_shapes: Sequence[torch.Size],
        input_shape: torch.Size,
        context_shape: Optional[torch.Size],
        zero_init: bool = True,
    ) -> None:
        self.LU = LU_decompose
        self.zero_init = zero_init
        super().__init__(
            param_shapes=param_shapes,
            input_shape=input_shape,
            context_shape=context_shape,
        )
        self._build()

    def _get_permutation_matrix(self, num_channels: int) -> torch.Tensor:
        nz = [
            torch.zeros(num_channels - k).scatter_(
                -1,
                torch.multinomial(
                    torch.randn(num_channels - k).softmax(-1), 1, replacement=False
                ),
                1.0,
            )
            for k in range(num_channels)
        ]
        np_p0 = torch.zeros((num_channels, num_channels))
        allidx = torch.arange(0, num_channels)
        for i, _nz in enumerate(nz):
            np_p0[:, i][allidx] = _nz
            allidx = allidx[~_nz.bool()]
        return np_p0

    def _build(self) -> None:
        self.num_channels = num_channels = self.param_shapes[0][-3]
        w_shape = torch.Size([num_channels, num_channels])

        np_p0 = self._get_permutation_matrix(num_channels)
        if self.zero_init:
            w_init = None
        else:
            w = torch.randn(w_shape) + torch.eye(num_channels) * 1e-4
            w_init = torch.linalg.qr(w)[
                0
            ]  # np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
        if not self.LU:
            # Sample a random orthogonal matrix:
            self.register_buffer("p", np_p0)
            weight = np_p0 @ w_init
            self.weight = nn.Parameter(weight.clone().requires_grad_())
        else:
            if w_init is not None:
                np_p, np_l, np_u = [
                    torch.tensor(_v) for _v in scipy_linalg.lu(w_init.numpy())
                ]
            else:
                np_p = np_p0  # torch.zeros(w_shape).scatter_(-1, torch.multinomial(torch.randn(w_shape).softmax(-1), 1), 1.0)
                np_l = torch.eye(w_shape[0])
                np_u = torch.eye(w_shape[0])
            self.register_buffer("p", np_p)
            if self.zero_init:
                np_s = torch.ones(num_channels).expm1().log() - self.BIAS_SOFTPLUS
                np_s_sign = torch.ones_like(np_s, dtype=torch.int)
            else:
                np_s = abs(np_u.diag()).expm1().log() - self.BIAS_SOFTPLUS
                np_s_sign = np_u.diag().sign()
            self.log_s = nn.Parameter(np_s.requires_grad_())
            self.low = nn.Parameter(np_l.requires_grad_())
            self.up = nn.Parameter(np_u.requires_grad_())

            self.register_buffer("s_sign", np_s_sign)
            l_mask = torch.ones(w_shape).tril(-1)
            self.register_buffer("l_mask", l_mask)
            eye = torch.eye(*w_shape)
            self.register_buffer("eye", eye)

    def _forward(
        self,
        input: torch.Tensor,
        inverse: bool,
        context: Optional[torch.Tensor] = None,
    ) -> Optional[Sequence[torch.Tensor]]:
        num_channels = self.num_channels
        if not self.LU:
            pixels = _pixels(input)
            weight = self.weight
            dlogdet = torch.slogdet(weight)[1] * pixels
            if not inverse:
                weight_out = self.weight.view(num_channels, num_channels, 1, 1)  # type: ignore
            else:
                weight_double = weight.double()
                dtype = weight.dtype
                assert isinstance(dtype, torch.dtype)
                weight_out = (
                    torch.inverse(weight_double)
                    .to(dtype)
                    .view(num_channels, num_channels, 1, 1)
                )
            return weight_out, dlogdet
        else:
            low = self.low
            l_mask = self.l_mask
            assert isinstance(l_mask, torch.Tensor)
            eye = self.eye
            assert isinstance(eye, torch.Tensor)
            low_out = low * l_mask + eye

            log_s = self.log_s
            s_sign = self.s_sign
            s = F.softplus(self.BIAS_SOFTPLUS + log_s) * s_sign
            assert isinstance(s, torch.Tensor)

            up = self.up
            assert isinstance(up, torch.Tensor)
            l_mask_transpose = l_mask.transpose(-1, -2)  # type: ignore
            up_out = up * l_mask_transpose + s.diag_embed()
            dlogdet = _sum(abs(s).clamp_min(1e-5).log()) * _pixels(input)
            if not inverse:
                w = self.p @ low_out @ up_out
                return w.view(num_channels, num_channels, 1, 1), dlogdet
            else:
                p = self.p
                assert isinstance(p, torch.Tensor)
                return p, low_out, up_out, dlogdet
