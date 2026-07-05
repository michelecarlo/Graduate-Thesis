"""Missing-data imputers for financial log-return panels.

Every imputer follows the same small scikit-learn-style interface::

    model.fit(X)              # learn parameters from X
    Xfilled = model.transform(X)
    Xfilled = model.fit_transform(X)

``X`` is a ``(T, N)`` panel (``pandas.DataFrame`` or ``numpy`` array) of
``T`` time steps and ``N`` assets, with ``numpy.nan`` marking the missing
entries.  Unless noted, ``transform`` returns a **complete** ``(T, N)`` array
in which the originally observed cells are preserved exactly and only the
missing cells are filled.

Implemented here, from the ground up:

    DeletionImputer  -- drop every row that contains a missing value
    ZeroImputer      -- replace missing entries with 0 (the "no move" return)
    EMImputer        -- Gaussian expectation-maximisation (Little & Rubin)
    KNNImputer       -- k-nearest-neighbour fill on co-observed features
    BRITSImputer     -- Bidirectional Recurrent Imputation (PyTorch)
    CSDIImputer      -- Conditional Score-based Diffusion Imputation (PyTorch)

The two neural models (BRITS, CSDI) are written in PyTorch.  They are
deliberately compact -- faithful to the published architectures but small
enough to train on CPU.
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

__all__ = [
    "DeletionImputer",
    "ZeroImputer",
    "EMImputer",
    "KNNImputer",
    "MahalanobisKNNImputer",
    "BRITSImputer",
    "CSDIImputer",
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _to_numpy(X):
    """Return ``X`` as a float64 ``numpy`` array with NaNs intact."""
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    return np.asarray(X, dtype=float)


def _standardize(X, observed):
    """Per-asset standardisation using observed entries only.

    Returns the standardised panel (missing cells set to 0), plus the mean
    and std used, so the result can be mapped back to the original scale.
    """
    masked = np.where(observed, X, np.nan)
    mu = np.nanmean(masked, axis=0)
    sd = np.nanstd(masked, axis=0)
    mu = np.nan_to_num(mu, nan=0.0)
    sd = np.where((sd == 0) | np.isnan(sd), 1.0, sd)
    Xs = np.where(observed, (X - mu) / sd, 0.0)
    return Xs, mu, sd


def _to_windows(arr, length, stride=None):
    """Cut a ``(T, ...)`` array into ``(n_windows, length, ...)`` blocks.

    With ``stride is None`` (or ``stride >= length``) the windows tile the series
    without overlap, zero-padding the end so its length is a multiple of
    ``length`` -- used at reconstruction time, where every step is filled once.
    With ``stride < length`` the windows overlap, yielding many more densely
    sampled training windows (a final window flush with the end is appended so
    the tail is covered).  Returns the windows and the padding added (0 when
    overlapping).
    """
    T = arr.shape[0]
    if stride is None or stride >= length:
        pad = (-T) % length
        if pad:
            arr = np.concatenate([arr, np.zeros((pad,) + arr.shape[1:], arr.dtype)], axis=0)
        return arr.reshape(-1, length, *arr.shape[1:]), pad
    starts = list(range(0, T - length + 1, stride))
    if not starts or starts[-1] != T - length:
        starts.append(T - length)
    return np.stack([arr[s:s + length] for s in starts], axis=0), 0


def _from_windows(win, T):
    """Inverse of :func:`_to_windows`; keep the first ``T`` rows."""
    flat = win.reshape(-1, win.shape[-1])
    return flat[:T]


class _BaseImputer:
    """Shared ``fit_transform`` glue."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        raise NotImplementedError

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def save(self, path):
        """Pickle the whole fitted imputer to ``path``.

        Captures everything needed to restore the model later -- for the neural
        imputers that means the network weights *and* the optimizer state, plus
        the standardisation stats and ``loss_history_`` -- so a fresh kernel can
        reload it and (with ``warm_start=True``) continue training from here.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return self

    @staticmethod
    def load(path):
        """Reload an imputer previously written with :meth:`save`."""
        with open(path, "rb") as f:
            return pickle.load(f)

    def _log(self, epoch, loss):
        """Record an epoch's mean training loss (and optionally print it)."""
        self.loss_history_.append(loss)
        if getattr(self, "verbose", False):
            every = max(1, self.epochs // 10)
            if epoch == 0 or (epoch + 1) % every == 0:
                name = type(self).__name__.replace("Imputer", "")
                print(f"{name:5s} epoch {epoch + 1:4d}/{self.epochs}  loss {loss:.5f}")


# --------------------------------------------------------------------------- #
# 1. Deletion
# --------------------------------------------------------------------------- #
class DeletionImputer(_BaseImputer):
    """Complete-case analysis: drop every row holding at least one NaN.

    This is the only imputer that changes the panel shape -- the result has
    ``(T', N)`` rows with ``T' <= T`` and no missing values.
    """

    def transform(self, X):
        X = _to_numpy(X)
        keep = ~np.isnan(X).any(axis=1)
        return X[keep]


# --------------------------------------------------------------------------- #
# 2. Zero fill
# --------------------------------------------------------------------------- #
class ZeroImputer(_BaseImputer):
    """Replace every missing entry with 0 (a flat, "no change" log return)."""

    def transform(self, X):
        X = _to_numpy(X)
        return np.where(np.isnan(X), 0.0, X)


# --------------------------------------------------------------------------- #
# 3. Gaussian EM
# --------------------------------------------------------------------------- #
class EMImputer(_BaseImputer):
    """Expectation-maximisation under a multivariate-normal model.

    Each row is treated as an i.i.d. draw from ``N(mu, Sigma)``.  The E-step
    fills missing entries with their conditional expectation given the
    observed ones; the M-step re-estimates ``mu`` and ``Sigma`` (including the
    conditional-covariance correction).  Rows are grouped by missingness
    pattern so each pattern needs a single matrix solve per iteration.
    """

    def __init__(self, max_iter=50, tol=1e-4, ridge=1e-6, verbose=False):
        self.max_iter = max_iter
        self.tol = tol
        self.ridge = ridge
        self.verbose = verbose

    def fit(self, X, y=None):
        X = _to_numpy(X)
        T, N = X.shape
        observed = ~np.isnan(X)

        mu = np.nan_to_num(np.nanmean(np.where(observed, X, np.nan), axis=0))
        Xf = np.where(observed, X, mu)
        Sigma = np.cov(Xf, rowvar=False) + self.ridge * np.eye(N)
        self.loss_history_ = []  # relative change in the covariance per iteration (-> 0 at convergence)

        missing = ~observed
        patterns, inverse = np.unique(missing, axis=0, return_inverse=True)
        eye = np.eye(N)

        for it in range(self.max_iter):
            Sigma_old = Sigma.copy()
            correction = np.zeros((N, N))

            for p, pattern in enumerate(patterns):
                rows = np.where(inverse == p)[0]
                mi = np.where(pattern)[0]                 # missing columns
                if mi.size == 0:                          # fully observed
                    continue
                oi = np.where(~pattern)[0]                # observed columns
                if oi.size == 0:                          # fully missing row(s)
                    Xf[np.ix_(rows, mi)] = mu[mi]
                    correction[np.ix_(mi, mi)] += Sigma[np.ix_(mi, mi)] * rows.size
                    continue

                Soo = Sigma[np.ix_(oi, oi)] + self.ridge * np.eye(oi.size)
                Som = Sigma[np.ix_(oi, mi)]
                B = np.linalg.solve(Soo, Som)             # (|o|, |m|)
                resid = X[np.ix_(rows, oi)] - mu[oi]
                Xf[np.ix_(rows, mi)] = mu[mi] + resid @ B
                cond_cov = Sigma[np.ix_(mi, mi)] - Som.T @ B
                correction[np.ix_(mi, mi)] += cond_cov * rows.size

            mu = Xf.mean(axis=0)
            centered = Xf - mu
            Sigma = (centered.T @ centered + correction) / T + self.ridge * eye

            delta = np.linalg.norm(Sigma - Sigma_old) / (np.linalg.norm(Sigma_old) + 1e-12)
            self.loss_history_.append(delta)
            if self.verbose:
                print(f"EM    iter {it + 1:3d}/{self.max_iter}  cov update {delta:.2e}")
            if delta < self.tol:
                break

        self.mu_, self.Sigma_ = mu, Sigma
        return self

    def transform(self, X):
        X = _to_numpy(X)
        N = X.shape[1]
        observed = ~np.isnan(X)
        Xf = np.where(observed, X, 0.0)
        missing = ~observed
        patterns, inverse = np.unique(missing, axis=0, return_inverse=True)

        for p, pattern in enumerate(patterns):
            rows = np.where(inverse == p)[0]
            mi = np.where(pattern)[0]
            if mi.size == 0:
                continue
            oi = np.where(~pattern)[0]
            if oi.size == 0:
                Xf[np.ix_(rows, mi)] = self.mu_[mi]
                continue
            Soo = self.Sigma_[np.ix_(oi, oi)] + self.ridge * np.eye(oi.size)
            Som = self.Sigma_[np.ix_(oi, mi)]
            B = np.linalg.solve(Soo, Som)
            resid = X[np.ix_(rows, oi)] - self.mu_[oi]
            Xf[np.ix_(rows, mi)] = self.mu_[mi] + resid @ B
        return Xf


# --------------------------------------------------------------------------- #
# 4. k-nearest neighbours
# --------------------------------------------------------------------------- #
class KNNImputer(_BaseImputer):
    """Fill each missing cell from its ``k`` nearest rows.

    Distances are the NaN-aware Euclidean distance over the features two rows
    both observe (scaled by the number of co-observed features).  For a given
    missing column the value is the (optionally distance-weighted) average of
    the nearest neighbours that actually observe that column.
    """

    def __init__(self, n_neighbors=5, weights="distance"):
        self.n_neighbors = n_neighbors
        self.weights = weights

    def fit(self, X, y=None):
        self.train_ = _to_numpy(X)
        self.col_mean_ = np.nan_to_num(np.nanmean(self.train_, axis=0))
        return self

    def _distances(self, diff):
        """Squared Euclidean distance per training row (``diff`` already zeroed
        on the features the two rows don't both observe)."""
        return (diff ** 2).sum(axis=1)

    def transform(self, X):
        X = _to_numpy(X)
        A = self.train_
        A_obs = ~np.isnan(A)
        N = X.shape[1]
        out = X.copy()

        for i in range(X.shape[0]):
            row = X[i]
            miss = np.isnan(row)
            if not miss.any():
                continue
            obs = ~miss

            both = A_obs & obs                              # (T_train, N)
            count = both.sum(axis=1)
            diff = np.where(both, A - row, 0.0)
            d2 = self._distances(diff) * (obs.sum() / np.maximum(count, 1))
            d2[count == 0] = np.inf
            order = np.argsort(d2)

            for j in np.where(miss)[0]:
                col = A[order, j]
                valid = order[~np.isnan(col)][: self.n_neighbors]
                if valid.size == 0:
                    out[i, j] = self.col_mean_[j]
                elif self.weights == "distance":
                    w = 1.0 / (np.sqrt(d2[valid]) + 1e-8)
                    out[i, j] = np.sum(w * A[valid, j]) / np.sum(w)
                else:
                    out[i, j] = A[valid, j].mean()
        return out


# --------------------------------------------------------------------------- #
# 5. k-nearest neighbours, Mahalanobis distance
# --------------------------------------------------------------------------- #
class MahalanobisKNNImputer(KNNImputer):
    """KNN fill using the Mahalanobis distance instead of the Euclidean one.

    Same scheme as :class:`KNNImputer`, but two rows are compared with
    ``(a - b) Sigma^{-1} (a - b)`` rather than ``(a - b)(a - b)``, so directions
    along strongly correlated assets are down-weighted instead of double-counted.
    The precision ``Sigma^{-1}`` is estimated once from the (mean-imputed)
    training panel; over the features two rows both observe this uses the
    matching block of the precision matrix.
    """

    def __init__(self, n_neighbors=5, weights="distance", ridge=1e-6):
        super().__init__(n_neighbors=n_neighbors, weights=weights)
        self.ridge = ridge

    def fit(self, X, y=None):
        super().fit(X)
        Xf = np.where(np.isnan(self.train_), self.col_mean_, self.train_)
        N = Xf.shape[1]
        Sigma = np.cov(Xf, rowvar=False) + self.ridge * np.eye(N)
        self.precision_ = np.linalg.inv(Sigma)
        return self

    def _distances(self, diff):
        """Mahalanobis squared distance over the co-observed features."""
        return ((diff @ self.precision_) * diff).sum(axis=1)


# --------------------------------------------------------------------------- #
# 6. BRITS  (Bidirectional Recurrent Imputation for Time Series)
# --------------------------------------------------------------------------- #
def _time_gaps(mask):
    """BRITS delta: time since each feature was last observed, per window.

    ``mask`` is ``(B, L, N)`` with 1 = observed.  ``delta[:, 0] = 0`` and
    ``delta[:, t] = 1 + (1 - mask[:, t-1]) * delta[:, t-1]``.
    """
    B, L, N = mask.shape
    delta = np.zeros((B, L, N), dtype=np.float32)
    for t in range(1, L):
        delta[:, t] = 1.0 + (1.0 - mask[:, t - 1]) * delta[:, t - 1]
    return delta


def _masked_mae(pred, target, mask):
    return (torch.abs(pred - target) * mask).sum() / (mask.sum() + 1e-5)


class _RITS(nn.Module):
    """One temporal direction of BRITS."""

    def __init__(self, n_features, hidden):
        super().__init__()
        self.hidden = hidden
        self.rnn = nn.LSTMCell(2 * n_features, hidden)
        self.decay_h = nn.Linear(n_features, hidden)      # hidden-state decay
        self.decay_x = nn.Linear(n_features, n_features)  # input decay
        self.hist = nn.Linear(hidden, n_features)         # history regression
        self.feat = nn.Linear(n_features, n_features)     # feature regression
        self.beta = nn.Linear(2 * n_features, n_features)  # combination gate

    def forward(self, x, m, delta):
        B, L, N = x.shape
        h = x.new_zeros(B, self.hidden)
        c = x.new_zeros(B, self.hidden)
        loss = 0.0
        imputations = []

        feat_w = self.feat.weight - torch.diag(torch.diag(self.feat.weight))

        for t in range(L):
            xt, mt, dt = x[:, t], m[:, t], delta[:, t]
            gamma_h = torch.exp(-torch.relu(self.decay_h(dt)))
            gamma_x = torch.exp(-torch.relu(self.decay_x(dt)))
            h = h * gamma_h

            x_hat = self.hist(h)
            loss = loss + _masked_mae(x_hat, xt, mt)

            x_comp = mt * xt + (1 - mt) * x_hat
            z_hat = torch.nn.functional.linear(x_comp, feat_w, self.feat.bias)
            loss = loss + _masked_mae(z_hat, xt, mt)

            beta = torch.sigmoid(self.beta(torch.cat([gamma_x, mt], dim=1)))
            c_hat = beta * z_hat + (1 - beta) * x_hat
            loss = loss + _masked_mae(c_hat, xt, mt)

            c_comp = mt * xt + (1 - mt) * c_hat
            h, c = self.rnn(torch.cat([c_comp, mt], dim=1), (h, c))
            imputations.append(c_comp)

        return torch.stack(imputations, dim=1), loss


class _BRITSNet(nn.Module):
    def __init__(self, n_features, hidden):
        super().__init__()
        self.forward_rits = _RITS(n_features, hidden)
        self.backward_rits = _RITS(n_features, hidden)

    def forward(self, x, m, delta_f, delta_b):
        imp_f, loss_f = self.forward_rits(x, m, delta_f)
        rev = lambda z: torch.flip(z, dims=[1])
        imp_b, loss_b = self.backward_rits(rev(x), rev(m), delta_b)
        imp_b = rev(imp_b)
        consistency = torch.abs(imp_f - imp_b).mean()
        imputation = (imp_f + imp_b) / 2
        return imputation, loss_f + loss_b + 0.1 * consistency


class BRITSImputer(_BaseImputer):
    """BRITS: a bidirectional RNN with temporal decay and feature regression.

    The long panel is cut into windows of ``window`` steps; the network is
    trained to reconstruct the observed entries of every window, then used to
    fill the missing ones.
    """

    def __init__(self, hidden=64, window=100, epochs=300, lr=1e-3,
                 batch_size=16, random_state=0, warm_start=False,
                 device="cpu", verbose=False):
        self.hidden = hidden
        self.window = window
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state
        self.warm_start = warm_start
        self.device = device
        self.verbose = verbose

    def fit(self, X, y=None):
        X = _to_numpy(X)
        observed = (~np.isnan(X)).astype(np.float32)
        dev = torch.device(self.device)

        # warm start: keep the existing weights, optimizer state and scaling and
        # train further on top of them; otherwise build everything fresh.
        resume = self.warm_start and getattr(self, "net_", None) is not None
        if resume:
            Xs = np.where(observed.astype(bool), (X - self.mu_) / self.sd_, 0.0)
        else:
            torch.manual_seed(self.random_state)
            Xs, self.mu_, self.sd_ = _standardize(X, observed.astype(bool))
            self.net_ = _BRITSNet(X.shape[1], self.hidden).to(dev)
            self.opt_ = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
            self.loss_history_ = []

        xw, _ = _to_windows(Xs.astype(np.float32), self.window)
        mw, _ = _to_windows(observed, self.window)
        dw_f = _time_gaps(mw)
        dw_b = _time_gaps(mw[:, ::-1].copy())

        x = torch.tensor(xw, device=dev)
        m = torch.tensor(mw, device=dev)
        df = torch.tensor(dw_f, device=dev)
        db = torch.tensor(dw_b, device=dev)

        opt = self.opt_
        n = x.shape[0]
        self.net_.train()
        for epoch in range(self.epochs):
            perm = torch.randperm(n, device=dev)
            total = 0.0
            for s in range(0, n, self.batch_size):
                idx = perm[s: s + self.batch_size]
                opt.zero_grad()
                _, loss = self.net_(x[idx], m[idx], df[idx], db[idx])
                loss.backward()
                opt.step()
                total += loss.item()
            self._log(epoch, total / max(1, math.ceil(n / self.batch_size)))
        return self

    def transform(self, X):
        X = _to_numpy(X)
        observed = (~np.isnan(X)).astype(np.float32)
        Xs = np.where(observed.astype(bool), (X - self.mu_) / self.sd_, 0.0)

        xw, _ = _to_windows(Xs.astype(np.float32), self.window)
        mw, _ = _to_windows(observed, self.window)
        dw_f = _time_gaps(mw)
        dw_b = _time_gaps(mw[:, ::-1].copy())

        dev = torch.device(self.device)
        self.net_.eval()
        with torch.no_grad():
            imp, _ = self.net_(
                torch.tensor(xw, device=dev),
                torch.tensor(mw, device=dev),
                torch.tensor(dw_f, device=dev),
                torch.tensor(dw_b, device=dev),
            )
        imp = imp.cpu().numpy()
        filled = _from_windows(imp, X.shape[0]) * self.sd_ + self.mu_
        return np.where(observed.astype(bool), X, filled)


# --------------------------------------------------------------------------- #
# 6. CSDI  (Conditional Score-based Diffusion Imputation)
# --------------------------------------------------------------------------- #
class _ResBlock(nn.Module):
    """A CSDI residual block: temporal attention, then feature attention."""

    def __init__(self, channels, n_heads):
        super().__init__()
        kw = dict(d_model=channels, nhead=n_heads, dim_feedforward=2 * channels,
                  batch_first=True, activation="gelu")
        self.time_attn = nn.TransformerEncoderLayer(**kw)
        self.feat_attn = nn.TransformerEncoderLayer(**kw)
        self.gate = nn.Linear(channels, 2 * channels)
        self.out = nn.Linear(channels, 2 * channels)

    def forward(self, h, diff_bias):
        B, L, N, C = h.shape
        y = h + diff_bias

        yt = y.permute(0, 2, 1, 3).reshape(B * N, L, C)
        yt = self.time_attn(yt).reshape(B, N, L, C).permute(0, 2, 1, 3)

        yf = yt.reshape(B * L, N, C)
        yf = self.feat_attn(yf).reshape(B, L, N, C)

        a, b = self.gate(yf).chunk(2, dim=-1)
        gated = torch.tanh(a) * torch.sigmoid(b)
        res, skip = self.out(gated).chunk(2, dim=-1)
        return (h + res) / math.sqrt(2.0), skip


class _CSDINet(nn.Module):
    """Noise-prediction network conditioned on the observed entries."""

    def __init__(self, n_features, window, channels=32, n_layers=2,
                 n_heads=4, d_time=16, d_feat=16):
        super().__init__()
        self.in_proj = nn.Linear(2, channels)             # [conditioning, noisy]
        self.feat_emb = nn.Embedding(n_features, d_feat)
        self.side_proj = nn.Linear(d_time + d_feat, channels)
        self.diff_mlp = nn.Sequential(
            nn.Linear(channels, channels), nn.SiLU(), nn.Linear(channels, channels)
        )
        self.blocks = nn.ModuleList(_ResBlock(channels, n_heads) for _ in range(n_layers))
        self.out_proj = nn.Linear(channels, 1)
        self.channels = channels

        pos = torch.arange(window)[:, None]
        div = torch.exp(torch.arange(0, d_time, 2) * (-math.log(10000.0) / d_time))
        pe = torch.zeros(window, d_time)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("time_pe", pe)

    def _diff_embedding(self, t):
        half = self.channels // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device) * (-math.log(10000.0) / half)
        )
        args = t.float()[:, None] * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(self, cond, noisy, t):
        B, L, N = noisy.shape
        h = self.in_proj(torch.stack([cond, noisy], dim=-1))   # (B, L, N, C)

        time = self.time_pe[:L, None, :].expand(L, N, -1)
        feat = self.feat_emb.weight[None, :, :].expand(L, N, -1)
        side = self.side_proj(torch.cat([time, feat], dim=-1))
        h = h + side[None]

        diff_bias = self.diff_mlp(self._diff_embedding(t))[:, None, None, :]

        skips = 0.0
        for block in self.blocks:
            h, skip = block(h, diff_bias)
            skips = skips + skip
        out = skips / math.sqrt(len(self.blocks))
        return self.out_proj(out).squeeze(-1)              # (B, L, N)


class CSDIImputer(_BaseImputer):
    """CSDI: conditional denoising-diffusion imputation.

    A DDPM is trained self-supervised -- part of the observed entries are hidden
    and the network learns to denoise them given the rest.  At inference the
    reverse process is run with the observed cells held fixed, so sampling is
    conditioned on what we actually know.  Each reverse pass is one stochastic
    draw; following the paper, the imputation is the median over ``n_samples``
    such draws (more samples -> less noise, proportionally more time).
    """

    def __init__(self, window=32, stride=None, channels=32, n_layers=2, n_heads=4,
                 n_steps=50, epochs=100, lr=1e-3, batch_size=16,
                 n_samples=1, random_state=0, warm_start=False, device="cpu",
                 verbose=False):
        self.window = window
        # overlapping training windows -> many more, more densely sampled training
        # samples (so more gradient steps per epoch and a smoother loss curve).
        self.stride = stride if stride is not None else max(1, window // 2)
        self.channels = channels
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_steps = n_steps
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.random_state = random_state
        self.warm_start = warm_start
        self.device = device
        self.verbose = verbose

    def _schedule(self, dev):
        # quadratic beta schedule from ~1e-4 to 0.5 (as in CSDI)
        beta = torch.linspace(1e-4 ** 0.5, 0.5 ** 0.5, self.n_steps, device=dev) ** 2
        alpha = 1.0 - beta
        return beta, alpha, torch.cumprod(alpha, dim=0)

    def fit(self, X, y=None):
        X = _to_numpy(X)
        observed = ~np.isnan(X)
        dev = torch.device(self.device)

        # warm start: keep the existing weights, optimizer state and scaling and
        # train further on top of them; otherwise build everything fresh.
        resume = self.warm_start and getattr(self, "net_", None) is not None
        if resume:
            Xs = np.where(observed, (X - self.mu_) / self.sd_, 0.0)
        else:
            torch.manual_seed(self.random_state)
            Xs, self.mu_, self.sd_ = _standardize(X, observed)
            self.net_ = _CSDINet(X.shape[1], self.window, self.channels,
                                 self.n_layers, self.n_heads).to(dev)
            self.opt_ = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
            self.loss_history_ = []

        # overlapping windows for training (many more samples than a clean tiling)
        xw, _ = _to_windows(Xs.astype(np.float32), self.window, self.stride)
        mw, _ = _to_windows(observed.astype(np.float32), self.window, self.stride)

        x0 = torch.tensor(xw, device=dev)
        om = torch.tensor(mw, device=dev)
        beta, alpha, abar = self._schedule(dev)

        opt = self.opt_
        n = x0.shape[0]
        self.net_.train()
        for epoch in range(self.epochs):
            perm = torch.randperm(n, device=dev)
            total, nb = 0.0, 0
            for s in range(0, n, self.batch_size):
                idx = perm[s: s + self.batch_size]
                xb, mb = x0[idx], om[idx]
                # random conditioning ratio per window, so training spans the full
                # range of observed fractions (incl. the high ones seen at inference)
                # instead of a fixed 50% -- this is what keeps the sampler calibrated.
                ratio = torch.rand(xb.shape[0], 1, 1, device=dev)
                cond_mask = mb * (torch.rand_like(mb) < ratio)
                target = mb * (1 - cond_mask)
                if target.sum() == 0:
                    continue

                t = torch.randint(0, self.n_steps, (xb.shape[0],), device=dev)
                a = abar[t][:, None, None]
                eps = torch.randn_like(xb)
                noisy = torch.sqrt(a) * xb + torch.sqrt(1 - a) * eps
                # conditioning cells are given to the model clean (as at inference);
                # only the non-conditioning cells are actually noised.
                noisy = cond_mask * xb + (1 - cond_mask) * noisy

                eps_hat = self.net_(cond_mask * xb, noisy, t)
                loss = (((eps_hat - eps) * target) ** 2).sum() / (target.sum() + 1e-5)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item()
                nb += 1
            self._log(epoch, total / max(1, nb))
        return self

    @torch.no_grad()
    def transform(self, X):
        X = _to_numpy(X)
        observed = ~np.isnan(X)
        Xs = np.where(observed, (X - self.mu_) / self.sd_, 0.0)

        xw, _ = _to_windows(Xs.astype(np.float32), self.window)
        mw, _ = _to_windows(observed.astype(np.float32), self.window)

        dev = torch.device(self.device)
        n_win = xw.shape[0]
        # tile the windows so all n_samples draws run through the net together
        x0 = torch.tensor(xw, device=dev).repeat(self.n_samples, 1, 1)
        om = torch.tensor(mw, device=dev).repeat(self.n_samples, 1, 1)
        beta, alpha, abar = self._schedule(dev)

        self.net_.eval()
        x = om * x0 + (1 - om) * torch.randn_like(x0)       # observed clean, missing ~ noise
        for t in reversed(range(self.n_steps)):
            tt = torch.full((x.shape[0],), t, device=dev, dtype=torch.long)
            eps_hat = self.net_(om * x0, x, tt)
            mean = (x - beta[t] / torch.sqrt(1 - abar[t]) * eps_hat) / torch.sqrt(alpha[t])
            if t > 0:
                x = mean + torch.sqrt(beta[t]) * torch.randn_like(x)
            else:
                x = mean
            x = om * x0 + (1 - om) * x                       # keep observed clean, evolve only missing

        samples = x.reshape(self.n_samples, n_win, self.window, -1)
        x = samples.median(dim=0).values                    # median over the draws
        imp = _from_windows(x.cpu().numpy(), X.shape[0]) * self.sd_ + self.mu_
        return np.where(observed, X, imp)
