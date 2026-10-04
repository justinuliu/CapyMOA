"""Data stream representations and related utilities."""

from capymoa._optional import lazy_torch_attrs

# `_stream` and `_csv_stream` must be imported before `drift`/`generator`/
# `preprocessing`: those submodules (transitively) import names such as
# `MOAStream` from `capymoa.stream`, which requires this module's own
# re-exports to already be bound. isort would normally alphabetize `from .
# import ...` above these, which reintroduces that circular import, hence
# `isort: skip`.
from ._stream import (  # isort: skip
    ARFFStream,
    FiniteStream,
    MOAStream,
    NumpyStream,
    Schema,
    Stream,
)
from ._csv_stream import CSVStream  # isort: skip
from . import drift, generator, preprocessing
from ._stream_from_file import stream_from_file

__all__ = [
    "ARFFStream",
    "CSVStream",
    "FiniteStream",
    "MOAStream",
    "NumpyStream",
    "Schema",
    "Stream",
    "TorchStream",
    "drift",
    "generator",
    "preprocessing",
    "stream_from_file",
]


#: Names that need PyTorch. Imported on first access so ``import capymoa`` stays
#: torch-free -- see :mod:`capymoa._optional`.
_LAZY = {
    "TorchStream": ".torch",
}

__getattr__, __dir__ = lazy_torch_attrs(__name__, _LAZY, "TorchStream", __all__)
