from importlib.metadata import version

from src.tabpfn.classifier import TabPFNClassifier
from src.tabpfn.errors import TabPFNCUDAOutOfMemoryError, TabPFNMPSOutOfMemoryError
from src.tabpfn.misc.debug_versions import display_debug_info
from src.tabpfn.model_loading import (
    load_fitted_tabpfn_model,
    save_fitted_tabpfn_model,
)
from src.tabpfn.regressor import TabPFNRegressor

try:
    __version__ = version(__name__)
except ImportError:
    __version__ = "unknown"

__all__ = [
    "TabPFNCUDAOutOfMemoryError",
    "TabPFNClassifier",
    "TabPFNMPSOutOfMemoryError",
    "TabPFNRegressor",
    "__aversion__",
    "display_debug_info",
    "load_fitted_tabpfn_model",
    "save_fitted_tabpfn_model",
]
