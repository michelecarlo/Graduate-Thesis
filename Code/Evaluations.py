"""Metrics for comparing an imputed log-return panel against the ground truth.

Each metric is a plain function ``metric(true, imp, mask=None) -> float`` where
``true`` and ``imp`` are ``(T, N)`` panels (``numpy`` arrays or ``pandas``
frames).  ``mask`` is an optional boolean ``(T, N)`` array that is ``True`` on
the cells that were originally missing (and therefore imputed); the pointwise
errors (MAE, RMSE) are restricted to those cells, while the structural metrics
(covariance, VaR, Wasserstein, ACF of absolute returns, leverage effect)
compare the full panels.

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
    "mae", "rmse", "cov_error", "var_error", "wasserstein_error",
    "acf", "leverage",
    "acf_abs_error", "leverage_error",
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
# Temporal structure: volatility clustering and the leverage effect
# --------------------------------------------------------------------------- #
def acf(x, nlags=20):
    """Autocorrelation of the series ``x`` at lags ``1..nlags``."""
    x = _arr(x).ravel()
    x = x - x.mean()
    denom = float(x @ x) + 1e-12
    return np.array([float(x[:-k] @ x[k:]) / denom for k in range(1, nlags + 1)])


def leverage(x, nlags=20):
    """Leverage curve ``corr(r_t, |r_{t+k}|)`` at lags ``1..nlags``.

    Negative values are the leverage effect: a negative return today raises
    volatility over the following days.
    """
    x = _arr(x).ravel()
    a = np.abs(x)
    xc, ac = x - x.mean(), a - a.mean()
    scale = xc.std() * ac.std() + 1e-12
    return np.array([
        float(xc[:-k] @ ac[k:]) / ((len(x) - k) * scale) for k in range(1, nlags + 1)
    ])


def _curve_error(true, imp, curve, nlags):
    """Mean absolute gap between per-asset ``curve``s of the two panels."""
    true, imp = _arr(true), _arr(imp)
    if min(true.shape[0], imp.shape[0]) < nlags + 2:
        return np.nan
    return float(np.mean([
        np.abs(curve(imp[:, j], nlags) - curve(true[:, j], nlags)).mean()
        for j in range(true.shape[1])
    ]))


def acf_abs_error(true, imp, mask=None, nlags=20):
    """Mean absolute error of the ACF of absolute returns (volatility clustering)."""
    return _curve_error(true, imp, lambda x, n: acf(np.abs(x), n), nlags)


def leverage_error(true, imp, mask=None, nlags=20):
    """Mean absolute error of the leverage curve ``corr(r_t, |r_{t+k}|)``."""
    return _curve_error(true, imp, leverage, nlags)


# --------------------------------------------------------------------------- #
# Serialized evaluation
# --------------------------------------------------------------------------- #
class Evaluator:
    """Apply a set of metrics across many datasets and models.

    ``var_level`` sets the VaR tail and ``nlags`` the horizon of the temporal
    curves (ACF of absolute returns, leverage); ``metrics`` can be
    overridden with any ``{name: function}`` mapping following the
    ``(true, imp, mask)`` signature.
    """

    def __init__(self, metrics=None, var_level=0.05, nlags=20):
        self.metrics = metrics or {
            "MAE":    mae,
            "RMSE":   rmse,
            "CovErr": cov_error,
            f"VaR{int(var_level * 100)}Err": lambda t, i, m: var_error(t, i, m, level=var_level),
            "WassErr": wasserstein_error,
            "AbsACFErr":  lambda t, i, m: acf_abs_error(t, i, m, nlags=nlags),
            "LevErr":     lambda t, i, m: leverage_error(t, i, m, nlags=nlags),
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
