"""Compatibilidad de DeepChecks 0.19.1 con las versiones fijadas en `requirements.txt`.

DeepChecks ya no se publica para NumPy 2.x ni scikit-learn 1.9, y falla al importarse:
- usa `np.Inf`, eliminado en NumPy 2.0;
- pide el scorer `max_error`, que scikit-learn renombró a `neg_max_error`;
- importa `pkg_resources` (de setuptools) solo para `parse_version` y la lista de paquetes
  instalados. Se simula con `packaging` e `importlib.metadata` para no depender de setuptools,
  cuyas versiones con `pkg_resources` (<81) tienen vulnerabilidades conocidas.
Se agregan estos alias antes de importar `deepchecks`. Solo se usan chequeos de datos, que no
dependen de esos nombres.
"""

import importlib.metadata as metadata
import logging
import sys
import types
import warnings

from packaging.version import parse as _parse_version
import numpy as np
import sklearn.metrics as metrics
import sklearn.metrics._scorer as scorer

np.Inf = np.inf  # noqa: N816
_get_scorer = scorer.get_scorer


def _get_scorer_compat(nombre):
    return _get_scorer("neg_max_error" if nombre == "max_error" else nombre)


scorer.get_scorer = metrics.get_scorer = _get_scorer_compat
warnings.filterwarnings("ignore", module="deepchecks")
logging.getLogger("deepchecks").setLevel(logging.ERROR)


if "pkg_resources" not in sys.modules:
    try:
        import pkg_resources  # noqa: F401
    except ModuleNotFoundError:
        _pkg = types.ModuleType("pkg_resources")
        _pkg.parse_version = _parse_version
        _pkg.working_set = [
            types.SimpleNamespace(key=d.metadata["Name"].lower()) for d in metadata.distributions()
        ]
        sys.modules["pkg_resources"] = _pkg
