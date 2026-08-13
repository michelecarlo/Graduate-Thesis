"""Downstream evaluation: the statistical factor structure of the panel.

The optimal portfolio of :mod:`Portfolios` scores an imputation by the decision
a mean-variance optimiser takes from it.  That decision passes through
``Sigma^{-1}``, so a distorted covariance is not only wrong but amplified.  This
module isolates the covariance itself.  The downstream model is principal
component analysis, the statistical factor model of the panel: from an imputed
panel we estimate ``Sigma``, take its leading eigenvectors as the factor
loadings, and ask how far that factor space is from the one the uncorrupted
panel gives.

Two things can go wrong and they are reported separately.  The *spectrum* can be
distorted, so that the imputation puts too much or too little variance in the
dominant factor; this is measured by the explained-variance shares.  The
*subspace* can be rotated, so that the factors are the wrong linear combinations
of assets; this is measured by the principal angles between the estimated and
the true leading eigenspaces.  Neither involves inverting anything, so what they
report is the structural damage alone.

Everything is kept parallel to :mod:`Portfolios` and :mod:`Evaluations`: plain
functions plus a :class:`FactorEvaluator` that serialises the computation across
missingness levels and models and returns a tidy ``DataFrame``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "covariance", "spectrum", "explained_variance_ratio", "loadings",
    "principal_angles", "subspace_distance", "FactorEvaluator",
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _arr(x):
    """Return ``x`` as a float ``numpy`` array."""
    if isinstance(x, (pd.DataFrame, pd.Series)):
        x = x.to_numpy()
    return np.asarray(x, dtype=float)


def covariance(R):
    """Sample covariance matrix of the ``(T, N)`` panel ``R``."""
    return np.cov(_arr(R), rowvar=False)


# --------------------------------------------------------------------------- #
# Spectrum
# --------------------------------------------------------------------------- #
def spectrum(Sigma):
    """Eigenvalues and eigenvectors of ``Sigma`` in decreasing eigenvalue order.

    ``Sigma`` is symmetric, so ``eigh`` is used and the eigenvectors are
    orthonormal.  Returns ``(eigenvalues, eigenvectors)`` with the eigenvectors
    as columns.
    """
    vals, vecs = np.linalg.eigh(np.asarray(Sigma, float))
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order]


def explained_variance_ratio(eigvals):
    """Share of total variance carried by each principal component."""
    eigvals = np.asarray(eigvals, float)
    total = eigvals.sum()
    return eigvals / total if total > 0 else np.full_like(eigvals, np.nan)


def loadings(R, k):
    """Leading ``k`` eigenvectors of the covariance of ``R``, as columns."""
    _, vecs = spectrum(covariance(R))
    return vecs[:, :k]


# --------------------------------------------------------------------------- #
# Distance between factor spaces
# --------------------------------------------------------------------------- #
def principal_angles(V1, V2):
    """Principal angles, in radians, between the column spans of ``V1``, ``V2``.

    Both matrices are assumed to have orthonormal columns, as eigenvectors of a
    symmetric matrix do.  The cosines of the angles are the singular values of
    ``V1' V2``, which makes the comparison invariant to the sign and to the
    ordering of the individual eigenvectors: only the subspace they span
    matters.
    """
    s = np.linalg.svd(np.asarray(V1, float).T @ np.asarray(V2, float),
                      compute_uv=False)
    return np.arccos(np.clip(s, -1.0, 1.0))


def subspace_distance(V1, V2):
    """Normalised chordal distance between the spans of ``V1`` and ``V2``.

    This is the root mean square of the sines of the principal angles,

        D = || sin(Theta) ||_F / sqrt(k),

    which is ``0`` when the two subspaces coincide and ``1`` when they are
    orthogonal.  Normalising by ``sqrt(k)`` keeps the values comparable across
    different numbers of factors.
    """
    V1, V2 = np.asarray(V1, float), np.asarray(V2, float)
    k = V1.shape[1]
    s = np.clip(np.linalg.svd(V1.T @ V2, compute_uv=False), 0.0, 1.0)
    return float(np.sqrt(max(k - float((s ** 2).sum()), 0.0) / k))


# --------------------------------------------------------------------------- #
# Serialized evaluation
# --------------------------------------------------------------------------- #
class FactorEvaluator:
    """Compare the factor structure of imputed panels with the true one.

    ``factors`` lists the numbers of factors at which the subspace distance is
    reported; ``max_k`` is how far the explained-variance and distance profiles
    are computed for plotting.  ``true_label`` names the reference row estimated
    on the uncorrupted panel.
    """

    def __init__(self, factors=(1, 5), max_k=15, true_label="True"):
        self.factors = tuple(factors)
        self.max_k = max_k
        self.true_label = true_label

    # -- one panel ---------------------------------------------------------- #
    def profile(self, panel):
        """Return ``(eigenvalues, eigenvectors, explained-variance ratios)``."""
        vals, vecs = spectrum(covariance(panel))
        return vals, vecs, explained_variance_ratio(vals)

    def evaluate(self, panel, vecs_true):
        """Score one imputed ``panel`` against the true eigenvectors.

        Returns ``(stats, ratios, curve)`` where ``stats`` holds the shares of
        variance in the leading factors and the subspace distances, ``ratios``
        is the explained-variance profile and ``curve`` the subspace distance
        as a function of the number of factors.
        """
        vals, vecs, ratios = self.profile(panel)

        stats = {"Lambda1": 100.0 * ratios[0],
                 "Top5": 100.0 * ratios[:5].sum()}
        for k in self.factors:
            stats[f"D{k}"] = subspace_distance(vecs_true[:, :k], vecs[:, :k])

        curve = np.array([subspace_distance(vecs_true[:, :k], vecs[:, :k])
                          for k in range(1, self.max_k + 1)])
        return stats, ratios, curve

    # -- every panel -------------------------------------------------------- #
    def evaluate_runs(self, runs, R_true, include_true=True, min_rows=2):
        """Score every model on every dataset against the true panel ``R_true``.

        ``runs`` is ``{dataset_key: {model_name: imputed_panel}}``.  The
        reference row, computed on ``R_true`` itself, is added once per dataset
        under ``true_label`` when ``include_true`` is set; its distances are
        zero by construction.

        Returns ``(table, ratios, curves)``:
          * ``table``  : ``DataFrame`` indexed by ``(dataset, model)`` with the
                         variance share of the first factor, that of the first
                         five, and the subspace distances at each ``factors``;
          * ``ratios`` : ``{(dataset, model): explained-variance profile}``;
          * ``curves`` : ``{(dataset, model): subspace distance vs k}``.
        """
        R_true = _arr(R_true)
        vals_t, vecs_t, ratios_t = self.profile(R_true)
        stats_t = {"Lambda1": 100.0 * ratios_t[0],
                   "Top5": 100.0 * ratios_t[:5].sum()}
        stats_t.update({f"D{k}": 0.0 for k in self.factors})
        curve_t = np.zeros(self.max_k)

        columns = list(stats_t)
        rows, ratios, curves = {}, {}, {}
        for key, models in runs.items():
            if include_true:
                idx = (key, self.true_label)
                rows[idx] = dict(stats_t)
                ratios[idx], curves[idx] = ratios_t, curve_t
            for model, panel in models.items():
                idx = (key, model)
                panel = _arr(panel)
                if panel.shape[0] < min_rows or not np.isfinite(panel).all():
                    rows[idx] = {c: np.nan for c in columns}
                    ratios[idx] = curves[idx] = None
                    continue
                stats, ratio, curve = self.evaluate(panel, vecs_t)
                rows[idx] = stats
                ratios[idx], curves[idx] = ratio, curve

        table = pd.DataFrame(rows).T
        table.index.names = ["dataset", "model"]
        return table, ratios, curves
