import jax
import pytest


@pytest.fixture
def disable_x64():
    """Runs a test with 64 bit precision disabled, which is the JAX default.

    JAXtronomy never enables x64 itself, so this is the configuration most users run in,
    but the test suite is run with JAX_ENABLE_X64=True. Tests that assert on the dtype
    of values returned to the user therefore need this fixture, otherwise x64 hides any
    float32 leaking out of the samplers.
    """
    x64_enabled = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", x64_enabled)
