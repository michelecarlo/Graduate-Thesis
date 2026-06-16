"""Reusable imputation models for a daily log-return panel of shape ``(T, N)``.

The module exposes a small, scikit-learn-like family of imputers used to compare
how different missing-data methods reconstruct financial time-series panels.  A
complete return matrix is meant to be artificially masked elsewhere, imputed with
one of these models, and then evaluated (MSE, covariance/volatility preservation,
forecasting, portfolio metrics) in a separate script.

Every model follows the same minimal interface::

    model = ModelName(**params)
    model.fit(X_train)
    X_imp = model.transform(X)
    X_imp = model.fit_transform(X_train)

``X`` is a 2D NumPy array or pandas DataFrame with missing values encoded as
``np.nan``.  All models return NumPy arrays.  Except for ``DeletionImputer`` the
output keeps the input shape and preserves observed entries exactly.

Models (in fixed order):
    1. DeletionImputer       -- naive baseline, drops incomplete rows/columns
    2. StudentTEMImputer     -- multivariate Student-t EM (heavy-tailed returns)
    3. KNNReturnImputer      -- nonparametric KNN baseline
    4. BRITSImputer          -- recurrent (bidirectional) deep imputation
    5. SAITSImputer          -- self-attention deep imputation
    6. CSDIImputer           -- conditional diffusion deep imputation

The deep-learning models are thin wrappers around PyPOTS implementations and are
imported lazily, so the module is usable even when PyPOTS is absent.
"""

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

__all__ = [
    "DeletionImputer",
    "StudentTEMImputer",
    "KNNReturnImputer",
    "BRITSImputer",
    "SAITSImputer",
    "CSDIImputer",
]


# --------------------------------------------------------------------------- #
# Input / output helpers
# --------------------------------------------------------------------------- #
def _to_numpy(X):
    """Return a float NumPy copy of ``X`` (accepts DataFrame or ndarray)."""
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    return np.array(X, dtype=float, copy=True)


def _to_3d(X):
    """Reshape a 2D panel ``(T, N)`` into the ``(1, T, N)`` batch PyPOTS expects."""
    X = _to_numpy(X)
    return X.reshape(1, X.shape[0], X.shape[1])


def _from_3d(X_3d):
    """Collapse a ``(n_samples, T, N)`` backend output back to ``(T, N)``."""
    X_3d = np.asarray(X_3d)
    if X_3d.ndim == 3:
        return X_3d[0]
    return X_3d


def _preserve_observed(X_original, X_imputed):
    """Overwrite imputed entries with the original observed values where present."""
    X_original = _to_numpy(X_original)
    out = np.asarray(X_imputed, dtype=float).copy()
    observed = ~np.isnan(X_original)
    out[observed] = X_original[observed]
    return out


# --------------------------------------------------------------------------- #
# 1. Deletion baseline
# --------------------------------------------------------------------------- #
class DeletionImputer:
    """Naive complete-case baseline -- it discards incomplete observations.

    Deletion is not a real imputation method: it shows what happens when rows or
    columns containing any missing value are simply dropped.  It therefore
    changes the shape of the dataset and is *not* directly comparable with the
    shape-preserving imputers; it is included only as a reference point.

    Parameters
    ----------
    strategy : {"row", "column"}
        ``"row"`` removes every row with at least one missing value;
        ``"column"`` removes every column with at least one missing value.
    """

    def __init__(self, strategy="row"):
        if strategy not in ("row", "column"):
            raise ValueError("strategy must be 'row' or 'column'")
        self.strategy = strategy

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = _to_numpy(X)
        missing = np.isnan(X)
        if self.strategy == "row":
            return X[~missing.any(axis=1)]
        return X[:, ~missing.any(axis=0)]

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


# --------------------------------------------------------------------------- #
# 2. Multivariate Student-t EM imputer
# --------------------------------------------------------------------------- #
class StudentTEMImputer:
    """Model-based imputer assuming a multivariate Student-t return vector.

    Financial returns are heavy-tailed, so a Student-t model is more robust than
    a Gaussian one.  Fitting uses an EM scheme that treats the t-distribution as
    a Gaussian scale mixture: each observation receives a weight that downplays
    extreme rows, and missing entries are filled by their conditional
    expectation under the current location ``mu_`` and scatter ``Sigma_``.

    Parameters
    ----------
    max_iter : int
        Maximum number of EM iterations.
    tol : float
        Convergence tolerance on the change of ``mu_`` / ``Sigma_``.
    nu : float
        Degrees of freedom of the Student-t (smaller => heavier tails).
    ridge : float
        Diagonal loading added to ``Sigma_`` for numerical stability.

    Attributes
    ----------
    mu_ : ndarray of shape (N,)
        Estimated location vector.
    Sigma_ : ndarray of shape (N, N)
        Estimated scatter (scale) matrix.
    """

    def __init__(self, max_iter=100, tol=1e-5, nu=5.0, ridge=1e-6):
        self.max_iter = max_iter
        self.tol = tol
        self.nu = nu
        self.ridge = ridge

    def fit(self, X, y=None):
        X = _to_numpy(X)
        T, N = X.shape
        obs = ~np.isnan(X)
        eye = self.ridge * np.eye(N)

        # Initialise from column means / sample covariance of mean-filled data.
        col_mean = np.nanmean(X, axis=0)
        col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
        filled = np.where(obs, np.nan_to_num(X), col_mean)
        mu = filled.mean(axis=0)
        Sigma = np.atleast_2d(np.cov(filled, rowvar=False)) + eye
        nu = self.nu

        for _ in range(self.max_iter):
            s_tau = 0.0                 # sum of weights
            s1 = np.zeros(N)            # sum w_i * xhat_i
            S2 = np.zeros((N, N))       # sum w_i * xhat_i xhat_i' + cond. cov
            for t in range(T):
                o = obs[t]
                p_o = int(o.sum())
                xhat = mu.copy()
                C = np.zeros((N, N))
                if p_o == 0:                      # nothing observed
                    w = 1.0
                    C = Sigma
                else:
                    m = ~o
                    So = Sigma[np.ix_(o, o)]
                    d = X[t, o] - mu[o]
                    Soi_d = np.linalg.solve(So, d)
                    delta = float(d @ Soi_d)
                    w = (nu + p_o) / (nu + delta)
                    xhat[o] = X[t, o]
                    if m.any():                   # regress missing on observed
                        Som = Sigma[np.ix_(o, m)]
                        xhat[m] = mu[m] + Som.T @ Soi_d
                        C[np.ix_(m, m)] = Sigma[np.ix_(m, m)] - Som.T @ np.linalg.solve(So, Som)
                s_tau += w
                s1 += w * xhat
                S2 += w * np.outer(xhat, xhat) + C

            mu_new = s1 / s_tau
            Sigma_new = (S2 - s_tau * np.outer(mu_new, mu_new)) / T
            Sigma_new = 0.5 * (Sigma_new + Sigma_new.T) + eye

            change = max(np.max(np.abs(mu_new - mu)),
                         np.max(np.abs(Sigma_new - Sigma)))
            mu, Sigma = mu_new, Sigma_new
            if change < self.tol:
                break

        self.mu_ = mu
        self.Sigma_ = Sigma
        return self

    def transform(self, X):
        X = _to_numpy(X)
        obs = ~np.isnan(X)
        out = X.copy()
        for t in range(X.shape[0]):
            o = obs[t]
            m = ~o
            if not m.any():
                continue
            if not o.any():
                out[t, m] = self.mu_[m]
                continue
            So = self.Sigma_[np.ix_(o, o)]
            Som = self.Sigma_[np.ix_(o, m)]
            d = X[t, o] - self.mu_[o]
            out[t, m] = self.mu_[m] + Som.T @ np.linalg.solve(So, d)
        return _preserve_observed(X, out)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


# --------------------------------------------------------------------------- #
# 3. KNN baseline
# --------------------------------------------------------------------------- #
class KNNReturnImputer:
    """Nonparametric KNN imputer (thin wrapper around ``sklearn`` KNNImputer).

    Missing returns at a given time point are filled from the most similar time
    points in the observed feature space, using a distance-weighted average of
    the ``n_neighbors`` nearest rows.

    Parameters
    ----------
    n_neighbors : int
        Number of neighbouring rows used for each imputation.
    weights : {"uniform", "distance"}
        Neighbour weighting scheme passed to ``sklearn.impute.KNNImputer``.
    """

    def __init__(self, n_neighbors=5, weights="distance"):
        self.n_neighbors = n_neighbors
        self.weights = weights

    def fit(self, X, y=None):
        X = _to_numpy(X)
        self.imputer_ = KNNImputer(n_neighbors=self.n_neighbors, weights=self.weights)
        self.imputer_.fit(X)
        return self

    def transform(self, X):
        X = _to_numpy(X)
        out = self.imputer_.transform(X)
        return _preserve_observed(X, out)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


# --------------------------------------------------------------------------- #
# PyPOTS plumbing shared by the deep-learning wrappers
# --------------------------------------------------------------------------- #
def _seed_everything(random_state):
    """Seed NumPy and (if present) PyTorch for reproducible deep training."""
    if random_state is None:
        return
    np.random.seed(random_state)
    try:
        import torch
        torch.manual_seed(random_state)
    except ImportError:
        pass


def _import_pypots(model_name):
    """Return ``pypots.imputation.<model_name>`` or raise a clear ImportError."""
    try:
        import pypots.imputation as imputation
    except ImportError as exc:  # PyPOTS not installed at all
        raise ImportError(
            f"{model_name} requires the optional package 'pypots'. "
            "Install it with `pip install pypots`."
        ) from exc
    model_cls = getattr(imputation, model_name, None)
    if model_cls is None:      # installed, but this model is unavailable
        raise ImportError(
            f"'{model_name}' is not available in the installed version of pypots. "
            "Upgrade pypots or install an implementation that provides it."
        )
    return model_cls


def _instantiate(model_cls, kwargs, learning_rate):
    """Build a PyPOTS model, tolerating minor cross-version API differences.

    Handles the ``learning_rate`` vs ``optimizer`` and ``d_inner`` vs ``d_ffn``
    naming changes that differ between PyPOTS releases.
    """
    lr_styles = [{"learning_rate": learning_rate}]
    try:
        from pypots.optim import Adam
        lr_styles.append({"optimizer": Adam(lr=learning_rate)})
    except Exception:
        pass
    lr_styles.append({})

    last_err = None
    for lr_kw in lr_styles:
        for rename in (False, True):
            kw = dict(kwargs)
            kw.update(lr_kw)
            if rename and "d_inner" in kw:
                kw["d_ffn"] = kw.pop("d_inner")
            try:
                return model_cls(**kw)
            except TypeError as exc:
                last_err = exc
    raise last_err


def _run_imputation(model, dataset, predict_kwargs=None):
    """Return the backend imputation array across PyPOTS API variants."""
    predict_kwargs = predict_kwargs or {}
    if hasattr(model, "predict"):
        try:
            result = model.predict(dataset, **predict_kwargs)
        except TypeError:
            result = model.predict(dataset)
        if isinstance(result, dict):
            return result.get("imputation", result)
        return result
    return model.impute(dataset)  # older PyPOTS


class _BaseDeepImputer:
    """Common fit/transform logic for the PyPOTS-backed wrappers."""

    _PYPOTS_NAME = None  # set by subclasses

    def _model_kwargs(self):
        """Subclass hook: constructor kwargs excluding the learning rate."""
        raise NotImplementedError

    def _predict_kwargs(self):
        """Subclass hook: extra kwargs for the prediction call."""
        return {}

    def fit(self, X, y=None):
        X = _to_numpy(X)
        T, N = X.shape
        if self.n_steps is None:
            self.n_steps = T
        if self.n_features is None:
            self.n_features = N
        _seed_everything(self.random_state)
        model_cls = _import_pypots(self._PYPOTS_NAME)
        self.model_ = _instantiate(model_cls, self._model_kwargs(), self.learning_rate)
        self.model_.fit({"X": _to_3d(X)})
        return self

    def transform(self, X):
        X = _to_numpy(X)
        out = _run_imputation(self.model_, {"X": _to_3d(X)}, self._predict_kwargs())
        out = np.asarray(out)
        if out.ndim == 4:                       # (n_samples, n_sampling_times, T, N)
            out = out.mean(axis=1)
        return _preserve_observed(X, _from_3d(out))

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


# --------------------------------------------------------------------------- #
# 4. BRITS
# --------------------------------------------------------------------------- #
class BRITSImputer(_BaseDeepImputer):
    """Bidirectional recurrent imputation (BRITS) -- thin PyPOTS wrapper.

    BRITS learns missing values jointly with the temporal dynamics of the series
    using a bidirectional RNN.  The 2D panel ``(T, N)`` is treated as a single
    long sample of shape ``(1, T, N)`` before being passed to the backend.

    Parameters mirror ``pypots.imputation.BRITS``; ``n_steps`` and ``n_features``
    are inferred from the data during ``fit`` when left as ``None``.
    """

    _PYPOTS_NAME = "BRITS"

    def __init__(self, n_steps=None, n_features=None, rnn_hidden_size=64,
                 epochs=100, batch_size=1, learning_rate=1e-3, random_state=42):
        self.n_steps = n_steps
        self.n_features = n_features
        self.rnn_hidden_size = rnn_hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state

    def _model_kwargs(self):
        return dict(
            n_steps=self.n_steps,
            n_features=self.n_features,
            rnn_hidden_size=self.rnn_hidden_size,
            batch_size=self.batch_size,
            epochs=self.epochs,
        )


# --------------------------------------------------------------------------- #
# 5. SAITS
# --------------------------------------------------------------------------- #
class SAITSImputer(_BaseDeepImputer):
    """Self-Attention Imputation for Time Series (SAITS) -- thin PyPOTS wrapper.

    SAITS uses stacked self-attention blocks, letting it capture long-range
    temporal dependence as well as cross-sectional dependence across assets.
    The 2D panel ``(T, N)`` is reshaped to ``(1, T, N)`` for the backend.

    Parameters mirror ``pypots.imputation.SAITS``; ``n_steps`` and ``n_features``
    are inferred from the data during ``fit`` when left as ``None``.
    """

    _PYPOTS_NAME = "SAITS"

    def __init__(self, n_steps=None, n_features=None, n_layers=2, d_model=64,
                 d_inner=128, n_heads=4, d_k=16, d_v=16, dropout=0.1,
                 epochs=100, batch_size=1, learning_rate=1e-3, random_state=42):
        self.n_steps = n_steps
        self.n_features = n_features
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_inner = d_inner
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state

    def _model_kwargs(self):
        return dict(
            n_steps=self.n_steps,
            n_features=self.n_features,
            n_layers=self.n_layers,
            d_model=self.d_model,
            d_inner=self.d_inner,
            n_heads=self.n_heads,
            d_k=self.d_k,
            d_v=self.d_v,
            dropout=self.dropout,
            batch_size=self.batch_size,
            epochs=self.epochs,
        )


# --------------------------------------------------------------------------- #
# 6. CSDI
# --------------------------------------------------------------------------- #
class CSDIImputer(_BaseDeepImputer):
    """Conditional Score-based Diffusion Imputation (CSDI) -- thin PyPOTS wrapper.

    CSDI is probabilistic: instead of a single point estimate it models the
    conditional distribution of missing values given the observed ones, which is
    valuable in finance where uncertainty about missing returns feeds into
    covariance estimation, risk models and portfolio decisions.  When several
    samples are drawn (``n_sampling_times > 1``) their mean is returned.

    Parameters mirror ``pypots.imputation.CSDI``; ``n_steps`` and ``n_features``
    are inferred from the data during ``fit`` when left as ``None``.
    """

    _PYPOTS_NAME = "CSDI"

    def __init__(self, n_steps=None, n_features=None, n_layers=2, n_channels=64,
                 n_heads=4, d_time_embedding=128, d_feature_embedding=16,
                 d_diffusion_embedding=128, n_diffusion_steps=50,
                 epochs=100, batch_size=1, learning_rate=1e-3,
                 n_sampling_times=1, random_state=42):
        self.n_steps = n_steps
        self.n_features = n_features
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.n_heads = n_heads
        self.d_time_embedding = d_time_embedding
        self.d_feature_embedding = d_feature_embedding
        self.d_diffusion_embedding = d_diffusion_embedding
        self.n_diffusion_steps = n_diffusion_steps
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_sampling_times = n_sampling_times
        self.random_state = random_state

    def _model_kwargs(self):
        return dict(
            n_steps=self.n_steps,
            n_features=self.n_features,
            n_layers=self.n_layers,
            n_channels=self.n_channels,
            n_heads=self.n_heads,
            d_time_embedding=self.d_time_embedding,
            d_feature_embedding=self.d_feature_embedding,
            d_diffusion_embedding=self.d_diffusion_embedding,
            n_diffusion_steps=self.n_diffusion_steps,
            batch_size=self.batch_size,
            epochs=self.epochs,
        )

    def _predict_kwargs(self):
        return dict(n_sampling_times=self.n_sampling_times)
