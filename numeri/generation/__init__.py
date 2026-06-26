# Register bundled generation backends.
from . import prompts  # noqa: F401  (prompt-builder, not a backend)
from .diffusion import DiffusionBackend  # noqa: F401