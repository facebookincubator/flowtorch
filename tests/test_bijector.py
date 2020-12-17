import simplex
import simplex.bijectors
import simplex.params


def test_bijector_constructor():
    param_fn = simplex.params.DenseAutoregressive()
    b = simplex.bijectors.AffineAutoregressive(param_fn=param_fn)
    assert b is not None
