"""Downstream evaluation: the classical mean-variance (tangency) portfolio.

The metrics in :mod:`Evaluations` compare an imputed panel against the ground
truth directly.  This module scores an imputation the other way round, by the
quality of the *decision* a downstream model takes from it.  The downstream
model here is Markowitz mean-variance optimisation: from an imputed panel we
estimate ``(mu, Sigma)``, form the fully invested maximum-Sharpe (tangency)
portfolio

    w = Sigma^{-1} mu / (1' Sigma^{-1} mu),

and then realise it on the TRUE returns.  The realised mean, variance and
Sharpe ratio measure how good the allocation implied by the imputation turns
out to be.  The reference is the same portfolio estimated on the uncorrupted
panel (the oracle): it is the best in-sample allocation, and every imputation
is compared to it and to the others, not to a naive benchmark.

Everything is deliberately kept parallel to :mod:`Evaluations`: plain
functions plus a :class:`PortfolioEvaluator` that serialises the computation
across missingness levels and models and returns a tidy ``DataFrame``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "moments", "tangency_weights", "min_variance_weights", "realized",
    "PortfolioEvaluator",
]

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _arr(x):
    """Return ``x`` as a float ``numpy`` array."""
    if isinstance(x, (pd.DataFrame, pd.Series)):
        x = x.to_numpy()
    return np.asarray(x, dtype=float)


def moments(R):
    """Sample mean vector and covariance matrix of the ``(T, N)`` panel ``R``."""
    R = _arr(R)
    return R.mean(axis=0), np.cov(R, rowvar=False)


# --------------------------------------------------------------------------- #
# Portfolio weights
# --------------------------------------------------------------------------- #
def _ridged(Sigma, ridge):
    """Add a small multiple of the identity for numerical stability."""
    if not ridge:
        return Sigma
    N = Sigma.shape[0]
    return Sigma + ridge * (np.trace(Sigma) / N) * np.eye(N)


def tangency_weights(mu, Sigma, ridge=0.0):
    """Fully invested maximum-Sharpe portfolio ``w = Sigma^{-1} mu / 1' Sigma^{-1} mu``.

    Weights sum to one and may be negative (long-short), as the classical
    unconstrained problem allows.
    """
    Sigma = _ridged(np.asarray(Sigma, float), ridge)
    z = np.linalg.solve(Sigma, np.asarray(mu, float))
    return z / (np.ones(Sigma.shape[0]) @ z)


def min_variance_weights(Sigma, ridge=0.0):
    """Global minimum-variance portfolio ``w = Sigma^{-1} 1 / 1' Sigma^{-1} 1``.

    Kept here so other downstream evaluations can reuse it; not used by the
    optimal-portfolio table, which reports the tangency portfolio.
    """
    Sigma = _ridged(np.asarray(Sigma, float), ridge)
    ones = np.ones(Sigma.shape[0])
    z = np.linalg.solve(Sigma, ones)
    return z / (ones @ z)


# --------------------------------------------------------------------------- #
# Realisation on the true returns
# --------------------------------------------------------------------------- #
def realized(weights, R_true, periods=TRADING_DAYS):
    """Realise ``weights`` on the true panel ``R_true``.

    Returns ``(stats, series)`` where ``series`` is the daily portfolio log
    return ``R_true @ weights`` and ``stats`` holds its annualised mean,
    variance, volatility and Sharpe ratio.
    """
    r = _arr(R_true) @ np.asarray(weights, float)
    mean = float(r.mean()) * periods
    var = float(r.var(ddof=1)) * periods
    vol = np.sqrt(var)
    stats = {
        "Mean": mean,
        "Variance": var,
        "Volatility": vol,
        "Sharpe": mean / vol if vol > 0 else np.nan,
    }
    return stats, r


# --------------------------------------------------------------------------- #
# Serialized evaluation
# --------------------------------------------------------------------------- #
class PortfolioEvaluator:
    """Estimate and realise the tangency portfolio across models and datasets.

    ``periods`` annualises the moments (252 trading days); ``ridge`` optionally
    regularises the covariance before it is inverted.  ``true_label`` names the
    oracle row estimated on the uncorrupted panel.
    """

    def __init__(self, periods=TRADING_DAYS, ridge=0.0, true_label="True"):
        self.periods = periods
        self.ridge = ridge
        self.true_label = true_label

    def _weights(self, panel):
        mu, Sigma = moments(panel)
        return tangency_weights(mu, Sigma, ridge=self.ridge)

    def evaluate(self, panel, R_true):
        """Return ``(stats, weights, series)`` for one imputed ``panel``."""
        w = self._weights(panel)
        stats, series = realized(w, R_true, self.periods)
        return stats, w, series

    def evaluate_runs(self, runs, R_true, include_true=True, min_rows=2):
        """Score every model on every dataset against the true panel ``R_true``.

        ``runs`` is ``{dataset_key: {model_name: imputed_panel}}``.  The oracle
        (portfolio estimated on ``R_true`` itself) is added once per dataset
        under ``true_label`` when ``include_true`` is set.

        Returns ``(table, weights, series)``:
          * ``table``   : ``DataFrame`` indexed by ``(dataset, model)`` with the
                          realised Mean, Variance, Volatility and Sharpe, plus
                          ``L1ToTrue``, the gross distance of the weights from
                          the oracle;
          * ``weights`` : ``{(dataset, model): weight vector}``;
          * ``series``  : ``{(dataset, model): realised daily return series}``.
        """
        R_true = _arr(R_true)
        w_true = self._weights(R_true)
        s_true, r_true = realized(w_true, R_true, self.periods)

        rows, weights, series = {}, {}, {}
        for key, models in runs.items():
            if include_true:
                idx = (key, self.true_label)
                rows[idx] = {**s_true, "L1ToTrue": 0.0}
                weights[idx] = w_true
                series[idx] = r_true
            for model, panel in models.items():
                if _arr(panel).shape[0] < min_rows:
                    idx = (key, model)
                    rows[idx] = {k: np.nan for k in
                                 ("Mean", "Variance", "Volatility", "Sharpe", "L1ToTrue")}
                    weights[idx] = None
                    series[idx] = None
                    continue
                stats, w, r = self.evaluate(panel, R_true)
                idx = (key, model)
                rows[idx] = {**stats, "L1ToTrue": float(np.abs(w - w_true).sum())}
                weights[idx] = w
                series[idx] = r

        table = pd.DataFrame(rows).T
        table.index.names = ["dataset", "model"]
        return table, weights, series
