import pandas as pd
import numpy as np

def _add_sparse(df, rate=0.02):        # isolated entries: one asset at scattered dates
    return df.mask(rng.random(df.shape) < rate)

def _add_gaps(df, n=25, mean_len=15):  # contiguous gaps: long runs per asset
    X = df.copy(); T, N = X.shape
    for _ in range(n):
        j, t0 = rng.integers(N), rng.integers(T)
        X.iloc[t0:t0 + max(1, int(rng.exponential(mean_len))), j] = np.nan
    return X

def _stress_mask(
    X: pd.DataFrame,
    base_q: float = 0.05,
    stress_q: float = 0.25,
    stress_quantile: float = 0.90,
    seed: int = 3
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    X = np.log(X).diff().iloc[1:]  # log returns

    market_proxy = X.mean(axis=1)

    stress_score = market_proxy ** 2

    threshold = stress_score.quantile(stress_quantile)
    stress_days = stress_score >= threshold

    probs = pd.Series(float(base_q), index=X.index)
    probs.loc[stress_days] = stress_q

    random_matrix = rng.random(X.shape)
    prob_matrix = np.repeat(probs.values[:, None], X.shape[1], axis=1)

    mask = random_matrix < prob_matrix
    return pd.DataFrame(mask, index=X.index, columns=X.columns)

#def _convert_ret_to_price(

class DataHandler:
    '''Take as input a pd.DataFrame PRICES dataset and return the data with missing values.'''
    def __init__(self, data: pd.DataFrame):
        self.prices = data
        self.log_returns = np.log(self.prices / self.prices.shift(1)).dropna()
        self.seed = 69

    def getSparse(self, rates:list[float]=[0.02, 0.05, 0.1]):
        missing_prices = {r: _add_sparse(self.prices, rate=r) for r in rates}
        missing_returns = {r: np.log(X / X.shift(1)).dropna() for r, X in missing_prices.items()}
        return missing_prices, missing_returns
    
    def getGaps(self, args:list[tuple[int, int]]=[(25, 15), (50, 30), (100, 60)]):
        missing_prices = {f"{n} gaps": _add_gaps(self.prices, n=n, mean_len=l) for n, l in args}
        missing_returns = {k: np.log(X / X.shift(1)).dropna() for k, X in missing_prices.items()}
        return missing_prices, missing_returns
    
    def getStress(self, args:list[tuple[float, float, float]]=[(0.05, 0.25, 0.90), (0.02, 0.10, 0.95), (0.01, 0.05, 0.99)]):
        missing_prices = {}
        for base_q, stress_q, stress_quantile in args:
            mask = _stress_mask(self.prices, base_q=base_q, stress_q=stress_q, stress_quantile=stress_quantile, seed=self.seed)
            missing_prices[f"Stress {int(stress_quantile*100)}%"] = self.prices.mask(mask)
        missing_returns = {k: np.log(X / X.shift(1)).dropna() for k, X in missing_prices.items()}
        return missing_prices, missing_returns
    
