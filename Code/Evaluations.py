"""Metrics for comparing an imputed log-return panel against the ground truth.

Each metric is a plain function ``metric(true, imp, mask=None) -> float`` where
``true`` and ``imp`` are ``(T, N)`` panels (``numpy`` arrays or ``pandas``
frames).  ``mask`` is an optional boolean ``(T, N)`` array that is ``True`` on
the cells that were originally missing (and therefore imputed); the pointwise
errors (MAE, RMSE) are restricted to those cells, while the structural metrics
(covariance, volatility, VaR, Wasserstein, signature) compare the full panels.

The :class:`Evaluator` bundles a set of metrics and applies them in a
"serialized" way -- across the several missingness levels (e.g. the keys of
``Sparses`` / ``Gaps`` / ``Stressed``) and the several models -- returning one
tidy ``DataFrame`` indexed by ``(dataset, model)``.

Nothing here is wired into ``Analysis.ipynb``; it is meant to be imported and
applied there later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

__all__ = [
    "mae", "rmse", "cov_error", "vol_error", "var_error",
    "wasserstein_error", "truncated_signature", "signature_error",
    "Evaluator",
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _arr(x):
    """Return ``x`` as a float ``numpy`` array."""
    if isinstance(x, (pd.DataFrame, pd.Series)):
        x = x.to_numpy()
    return np.asarray(x, dtype=float)


def _aligned(true, imp, mask):
    """Difference ``true - imp`` restricted to ``mask`` (or all aligned cells).

    Returns ``None`` when the shapes don't match (e.g. row-deletion changed the
    panel size), so the pointwise metrics can report ``nan`` instead of crashing.
    """
    true, imp = _arr(true), _arr(imp)
    if true.shape != imp.shape:
        return None
    diff = true - imp
    return diff[mask] if mask is not None else diff.ravel()


def _degenerate(*panels):
    """True if any panel has fewer than 2 rows.

    The structural metrics aren't defined then -- e.g. ``DeletionImputer`` drops
    every row that has a missing value, so at high missingness it can return an
    empty panel.  Those metrics return ``nan`` in that case instead of crashing.
    """
    return any(_arr(p).shape[0] < 2 for p in panels)


# --------------------------------------------------------------------------- #
# Pointwise reconstruction error (on the imputed cells)
# --------------------------------------------------------------------------- #
def mae(true, imp, mask=None):
    """Mean absolute error on the imputed cells."""
    d = _aligned(true, imp, mask)
    return float(np.mean(np.abs(d))) if d is not None else np.nan


def rmse(true, imp, mask=None):
    """Root mean squared error on the imputed cells."""
    d = _aligned(true, imp, mask)
    return float(np.sqrt(np.mean(d ** 2))) if d is not None else np.nan


# --------------------------------------------------------------------------- #
# Second-moment structure
# --------------------------------------------------------------------------- #
def cov_error(true, imp, mask=None):
    """Relative Frobenius error of the asset covariance matrix."""
    if _degenerate(true, imp):
        return np.nan
    ct = np.cov(_arr(true), rowvar=False)
    ci = np.cov(_arr(imp), rowvar=False)
    return float(np.linalg.norm(ci - ct) / (np.linalg.norm(ct) + 1e-12))


def vol_error(true, imp, mask=None):
    """Mean relative error of the per-asset volatility (standard deviation)."""
    if _degenerate(true, imp):
        return np.nan
    vt = _arr(true).std(axis=0)
    vi = _arr(imp).std(axis=0)
    return float(np.mean(np.abs(vi - vt) / (np.abs(vt) + 1e-12)))


# --------------------------------------------------------------------------- #
# Tail risk
# --------------------------------------------------------------------------- #
def empirical_var(x, level=0.05):
    """Empirical Value-at-Risk per asset: the (positive) ``level``-tail loss."""
    return -np.quantile(_arr(x), level, axis=0)


def var_error(true, imp, mask=None, level=0.05):
    """Mean absolute error of the per-asset empirical VaR (in return units)."""
    if _degenerate(true, imp):
        return np.nan
    return float(np.mean(np.abs(empirical_var(imp, level) - empirical_var(true, level))))


# --------------------------------------------------------------------------- #
# Marginal distribution
# --------------------------------------------------------------------------- #
def wasserstein_error(true, imp, mask=None):
    """Mean 1-Wasserstein distance between the per-asset return distributions."""
    if _degenerate(true, imp):
        return np.nan
    true, imp = _arr(true), _arr(imp)
    return float(np.mean([
        wasserstein_distance(true[:, j], imp[:, j]) for j in range(true.shape[1])
    ]))


# --------------------------------------------------------------------------- #
# Path signature (truncated)
# --------------------------------------------------------------------------- #
def truncated_signature(increments, depth=2):
    """Truncated path signature of the cumulative path of ``increments``.

    The panel rows are treated as the increments of a path (log-returns -> the
    log-price path).  Returns the signature levels as a list of arrays:
    level 1 is the total increment ``(N,)`` and level 2 the matrix of iterated
    integrals ``int P^i dP^j`` ``(N, N)`` (captures cross-asset covariation).
    Depth > 2 is impractical for a 50-asset path (it grows as ``N**depth``) and
    would need a dedicated library (e.g. iisignature).
    """
    if depth > 2:
        raise ValueError("truncated_signature supports depth <= 2 (use iisignature for more)")
    dX = _arr(increments)
    path = np.cumsum(dX, axis=0)                       # P_t, with P_0 = 0 implied
    levels = [path[-1]]                                # level 1: total increment
    if depth >= 2:
        prev = np.vstack([np.zeros((1, dX.shape[1])), path[:-1]])  # P_{t-1}
        levels.append(prev.T @ dX)                     # level 2: \int P dP
    return levels


def signature_error(true, imp, mask=None, depth=2):
    """Mean per-level relative error of the truncated path signatures."""
    if _degenerate(true, imp):
        return np.nan
    st = truncated_signature(true, depth)
    si = truncated_signature(imp, depth)
    errs = [np.linalg.norm(b - a) / (np.linalg.norm(a) + 1e-12) for a, b in zip(st, si)]
    return float(np.mean(errs))


# --------------------------------------------------------------------------- #
# Serialized evaluation
# --------------------------------------------------------------------------- #
class Evaluator:
    """Apply a set of metrics across many datasets and models.

    Parameters mirror the metric knobs: ``var_level`` for the VaR tail and
    ``sig_depth`` for the signature truncation.  ``metrics`` can be overridden
    with any ``{name: function}`` mapping following the ``(true, imp, mask)``
    signature.
    """

    def __init__(self, metrics=None, var_level=0.05, sig_depth=2):
        self.metrics = metrics or {
            "MAE":    mae,
            "RMSE":   rmse,
            "CovErr": cov_error,
            "VolErr": vol_error,
            f"VaR{int(var_level * 100)}Err": lambda t, i, m: var_error(t, i, m, level=var_level),
            "WassErr": wasserstein_error,
            "SigErr": lambda t, i, m: signature_error(t, i, m, depth=sig_depth),
        }

    def evaluate(self, true, imp, mask=None):
        """Return ``{metric_name: value}`` comparing ``imp`` against ``true``."""
        return {name: fn(true, imp, mask) for name, fn in self.metrics.items()}

    def evaluate_runs(self, runs, trues, masks=None):
        """Score every model on every dataset.

        ``runs``  : ``{dataset_key: {model_name: imputed_panel}}``.
        ``trues`` : a single ground-truth panel, or ``{dataset_key: panel}`` when
                    the truth differs per dataset (e.g. standardized per panel).
        ``masks`` : optional ``{dataset_key: missing_mask}`` (``True`` where the
                    panel was originally missing); needed for MAE / RMSE.

        Returns a ``DataFrame`` indexed by ``(dataset, model)`` with one column
        per metric.
        """
        rows = {}
        for key, models in runs.items():
            true = trues[key] if isinstance(trues, dict) else trues
            mask = None if masks is None else masks.get(key)
            for model, imp in models.items():
                rows[(key, model)] = self.evaluate(true, imp, mask)
        out = pd.DataFrame(rows).T
        out.index.names = ["dataset", "model"]
        return out
