# Copyright (c) Meta Platforms, Inc

# pyre-unsafe
import time

import flowtorch.parameters as params
import pytest
import torch
from flowtorch.bijectors import AffineAutoregressive, Compose
from flowtorch.bijectors.utils import set_record_flow_graph

dim_x = 32


def get_net() -> AffineAutoregressive:
    ar = Compose(
        [
            AffineAutoregressive(params.DenseAutoregressive()),
            AffineAutoregressive(params.DenseAutoregressive()),
            AffineAutoregressive(params.DenseAutoregressive()),
        ]
    )
    ar = ar(
        shape=torch.Size(
            [
                dim_x,
            ]
        )
    )
    return ar


def test_forward():
    ar = get_net()
    x = torch.randn(50, dim_x, requires_grad=True)
    y = ar.forward(x)
    assert ar.inverse(y) is x
    assert ar.forward(y) is not x

    with set_record_flow_graph(False):
        y = ar.forward(x)
        assert ar.inverse(y) is not x
        assert ar.forward(y) is not x


def test_backward():
    ar = get_net()
    x = torch.randn(50, dim_x, requires_grad=True)
    y = ar.inverse(x)
    assert ar.forward(y) is x
    assert ar.inverse(y) is not x

    with set_record_flow_graph(False):
        y = ar.inverse(x)
        assert ar.forward(y) is not x
        assert ar.inverse(y) is not x


@pytest.mark.parametrize("mode", ["forward", "inverse"])
def test_gradient_matching(mode):
    ar = get_net()

    print("test with bijective tensor")
    t0 = time.time()
    with set_record_flow_graph(True):
        x = torch.randn(50, dim_x, requires_grad=True)
        t1 = time.time()
        if mode == "forward":
            y = ar.forward(x)
            xinv = ar.inverse(y)
            ldj = ar.log_abs_det_jacobian(x, y).sum()
        else:
            y = ar.inverse(x)
            xinv = ar.forward(y)
            ldj = ar.log_abs_det_jacobian(y, x).sum()
        assert xinv is x
        print(f"op with bij tensor took {time.time() - t1} for mode={mode}")
        ldj.backward()
        g_bijtensor = x.grad.clone()
    bij_time = time.time() - t0
    print("bij tensor time: ", bij_time)

    print("test with regular tensor")
    t0 = time.time()
    with set_record_flow_graph(False):
        x.grad = None
        t1 = time.time()
        if mode == "forward":
            y = ar.forward(x)
            xinv = ar.inverse(y)
            ldj = ar.log_abs_det_jacobian(x, y).sum()
        else:
            y = ar.inverse(x)
            xinv = ar.forward(y)
            ldj = ar.log_abs_det_jacobian(y, x).sum()
        assert xinv is not x
        print(f"op with regular tensor took {time.time() - t1} for mode={mode}")
        ldj.backward()
        g_tensor = x.grad.clone()
    tensor_time = time.time() - t0
    print("regular tensor time: ", tensor_time)

    print("diff between grads: ", (g_bijtensor - g_tensor).norm(2))
    torch.testing.assert_allclose(g_bijtensor, g_tensor)

    # This is flacky and should probably not be merged, but it's a good
    # soundness check locally
    assert bij_time < tensor_time, f"Bijective tensor {mode}+backprop took longer"


if __name__ == "__main__":
    pytest.main([__file__, "--capture", "no"])
