from typing import Tuple, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

import flowtorch
from . import Bijector
from .reshape import ReshapeBijector


class SplitBijector(ReshapeBijector):
    BIAS_SOFTPLUS = 0.54

    def __init__(
        self,
        params_fn: Optional[flowtorch.Lazy] = None,
        *,
        shape: torch.Size,
        transform: Bijector,
        chunk_dim: int = -3,
        context_shape: Optional[torch.Size] = None,
    ) -> None:
        super().__init__(params_fn, transform=transform, shape=shape, context_shape=context_shape)
        self.chunk_dim = chunk_dim
        self._split_prior = nn.Conv2d(10, 20, (3, 3), padding=(1, 1))

    def _reshape_in(
        self,
        x: Tensor,
        mode: str
    ) -> Tuple[Tensor, Tensor, Tensor]:
        x1, x2 = x.chunk(2, dim=self.chunk_dim)
        ldj = 0.0
        if mode == 'forward':
            m, logs = self._split_prior(x1).chunk(2, dim=self.chunk_dim)
            sd = F.softplus(self.BIAS_SOFTPLUS + logs)
            sd_rec = sd.clamp_min(1e-5).reciprocal()
            x2 = (x2 - m) * sd_rec
            ldj = sd_rec.log()
        return x1, x2, ldj

    def _reshape_out(
        self,
        x1: Tensor,
        mode: str,
        log_detJ: Tensor,
        x2: Tensor,
        ldj: Tensor,
    ) -> Tuple[Tensor, ...]:
        if mode == "inverse":
            m, logs = self._split_prior(x1)
            sd = F.softplus(self.BIAS_SOFTPLUS + logs)
            sd = sd.clamp_min(1e-5)
            x2 = m + x2 * sd
            ldj = - sd.log()
        if isinstance(ldj, Tensor) and isinstance(log_detJ, Tensor):
            if not ldj.shape == log_detJ.shape:
                raise RuntimeError(
                    "SplitBijector and transform log-abs-det Jacobian shapes"
                    f"mismatch: {ldj.shape} vs {log_detJ.shape}")
        ldj = ldj + log_detJ
        y = torch.cat([x1, x2], dim=self.chunk_dim)
        return y, ldj
