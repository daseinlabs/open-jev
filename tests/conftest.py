import pytest


@pytest.fixture(scope="session")
def renderer():
    """One offscreen renderer for the whole run (Panda allows one ShowBase per process)."""
    from openworld.render import Renderer
    return Renderer()
