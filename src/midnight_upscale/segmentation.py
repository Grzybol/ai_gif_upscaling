"""Optional AI background removal.

The base install does not need any segmentation library. ``BackgroundRemover``
is the interface; ``rembg`` is one implementation. Backends are registered by
name and take a model name, so neither is baked into the pipeline. Importing
this module never imports a heavy dependency.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Callable
from typing import Protocol

import numpy as np
from PIL import Image

from midnight_upscale.utils import PipelineError

NOT_INSTALLED = "AI background removal is not installed."
INSTALL_COMMAND = 'pip install -e ".[bgremove]"'
INSTALL_HELP = (
    f"{NOT_INSTALLED}\n"
    f"Install the optional dependency group and restart the page:\n  {INSTALL_COMMAND}\n"
    "The first run downloads the selected model once. Chroma Key and None keep working "
    "without it."
)

DEFAULT_BACKEND = "rembg"
# General foreground model. Person-specific models are listed as alternatives.
DEFAULT_MODEL = os.environ.get("MIDNIGHT_BGREMOVE_MODEL", "isnet-general-use")
MODEL_CHOICES = (
    "isnet-general-use",
    "u2net_human_seg",
    "u2net",
    "isnet-anime",
    "birefnet-general",
    "birefnet-portrait",
)


class BackgroundRemoverUnavailable(PipelineError):
    """The AI backend or its model is missing. The message says how to fix it."""


class BackgroundRemover(Protocol):
    """Turns one RGB frame into a soft foreground mask."""

    name: str

    def alpha(self, rgb: np.ndarray) -> np.ndarray:
        """Return ``HxW`` uint8 alpha for an ``HxWx3`` uint8 image."""

    def close(self) -> None:
        """Release the model."""


RemoverFactory = Callable[[str], BackgroundRemover]


class RembgRemover:
    name = "rembg"

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        if not rembg_installed():
            raise BackgroundRemoverUnavailable(INSTALL_HELP)
        from rembg import new_session

        self.model = model
        try:
            self._session = new_session(model)
        except Exception as exc:  # model download or load failure
            raise BackgroundRemoverUnavailable(
                f"AI background model {model!r} could not be loaded: {exc}\n"
                "The first run needs a network connection to download the model."
            ) from exc

    def alpha(self, rgb: np.ndarray) -> np.ndarray:
        from rembg import remove

        mask = remove(Image.fromarray(rgb[..., :3]), session=self._session, only_mask=True)
        return np.asarray(mask.convert("L"), dtype=np.uint8)

    def close(self) -> None:
        self._session = None


def rembg_installed() -> bool:
    return (
        importlib.util.find_spec("rembg") is not None
        and importlib.util.find_spec("onnxruntime") is not None
    )


_FACTORIES: dict[str, RemoverFactory] = {"rembg": RembgRemover}
_AVAILABILITY: dict[str, Callable[[], bool]] = {"rembg": rembg_installed}


def register_remover(
    name: str, factory: RemoverFactory, available: Callable[[], bool] = lambda: True
) -> None:
    """Add or replace a backend. Tests use this to plug in a fake model."""

    _FACTORIES[name] = factory
    _AVAILABILITY[name] = available


def backend_names() -> tuple[str, ...]:
    return tuple(_FACTORIES)


def ai_available(backend: str = DEFAULT_BACKEND) -> bool:
    check = _AVAILABILITY.get(backend)
    return bool(check and check())


def create_remover(backend: str = DEFAULT_BACKEND, model: str = DEFAULT_MODEL) -> BackgroundRemover:
    factory = _FACTORIES.get(backend)
    if factory is None:
        raise PipelineError(f"Unknown background-removal backend {backend!r}")
    if not ai_available(backend):
        raise BackgroundRemoverUnavailable(INSTALL_HELP)
    return factory(model)
