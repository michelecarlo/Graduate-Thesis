from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf
from pytickersymbols import PyTickerSymbols

START = "2010-01-01"
END = "2026-06-01"  # yfinance `end` is exclusive -> includes 2026-05-31
OUT_DIR = Path(__file__).resolve().parent
OUT_FILE = OUT_DIR / "sp500_prices.csv"


def sp500_tickers() -> list[str]:
    """Current S&P 500 constituents (Yahoo ticker format)."""
    stocks = PyTickerSymbols().get_stocks_by_index("S&P 500")
    tickers = {d["symbol"].replace(".", "-") for d in stocks}
    return sorted(tickers)


def fetch_prices(tickers: list[str]) -> pd.DataFrame:
    """Adjusted close prices, dates x tickers."""
    data = yf.download(
        tickers,
        start=START,
        end=END,
        auto_adjust=True,
        progress=True,
        threads=True,
    )
    return data["Close"].sort_index().sort_index(axis=1)


def main() -> None:
    tickers = sp500_tickers()
    prices = fetch_prices(tickers)
    prices.to_csv(OUT_FILE)
    n_missing = int(prices.isna().sum().sum())


if __name__ == "__main__":
    main()