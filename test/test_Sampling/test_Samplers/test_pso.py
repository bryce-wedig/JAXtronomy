__author__ = "ahuang314"

from functools import partial
from jax import jit, lax, numpy as jnp, vmap
import numpy as np
import numpy.testing as npt
import pytest

from jaxtronomy.Sampling.Samplers.pso import ParticleSwarmOptimizer
from jaxtronomy.Sampling.Samplers.pso_jit import ParticleSwarmOptimizerJIT


@jit
@vmap
def logL(x):
    # Minimum at x = 0.6
    return -jnp.sum((x - 0.6) ** 2)


class TestPSO(object):
    """Tests the PSO class with a simple logL function."""

    def setup_method(self):
        args_lower = np.array([0.0])
        args_upper = np.array([10.1])
        args = (args_lower, args_upper)
        self.pso = ParticleSwarmOptimizer(logL, *args, particle_count=10)

    def test_run(self):
        # Tests to see if the PSO gets close to the true answer
        final_result, _ = self.pso.optimize(max_iter=100)
        npt.assert_array_almost_equal(final_result, [0.6], decimal=3)

    def test_result_dtype(self, disable_x64):
        # The returned parameters must be usable as Python floats downstream;
        # see TestPSOJIT.test_result_dtype
        pso = ParticleSwarmOptimizer(
            logL, np.array([0.0]), np.array([10.1]), particle_count=10
        )
        final_result, _ = pso.optimize(max_iter=10)
        assert all(isinstance(value, float) for value in final_result)


class TestPSOJIT(object):
    """Tests the PSO JIT class with a simple logL function."""

    def setup_method(self):
        args_lower = np.array([0.0])
        args_upper = np.array([10.1])
        args = (args_lower, args_upper)
        self.pso = ParticleSwarmOptimizerJIT(logL, *args, particle_count=10)

    def test_run(self):
        # Tests to see if the PSO gets close to the true answer
        final_result, _ = self.pso.optimize(max_iter=100)
        npt.assert_array_almost_equal(final_result, [0.6], decimal=3)

        final_result, _ = self.pso.optimize(max_iter=100, early_stop_tolerance=1e-5)
        npt.assert_array_almost_equal(final_result, [0.6], decimal=3)

    def test_result_dtype(self, disable_x64):
        # JAX computes in float32 unless x64 is enabled. Since np.float32 is not a
        # subclass of Python float (unlike np.float64), returning the raw float32
        # values would make best-fit parameters fail isinstance(value, float) checks
        # on the GPU backend only, where this JIT PSO is used. The result must
        # therefore be float64, matching the non-JIT PSO and lenstronomy.
        pso = ParticleSwarmOptimizerJIT(
            logL, np.array([0.0]), np.array([10.1]), particle_count=10
        )
        final_result, _ = pso.optimize(max_iter=10)
        assert final_result.dtype == np.float64
        assert all(isinstance(value, float) for value in final_result)


if __name__ == "__main__":
    pytest.main()
