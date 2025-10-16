"""Analytical routines relating the shanzhai factor to short-side performance."""
from __future__ import annotations

from typing import Dict, Iterable, Tuple, Union

import pandas as pd
from pandas import DataFrame, Series

from .factor_model import ShanzhaiFactorModel
from .regime import MarketRegime, MarketRegimeClassifier


def _future_returns(prices: Union[Series, DataFrame], horizons: Iterable[int]) -> Dict[int, Series]:
    """Compute forward returns for a set of horizons."""

    returns: Dict[int, Series] = {}
    for horizon in horizons:
        shifted = prices.groupby(level=1).shift(-horizon)
        current = prices
        fwd = (shifted - current) / current

        if isinstance(fwd, Series):
            series = fwd
        else:
            if fwd.shape[1] == 1:
                series = fwd.iloc[:, 0]
            else:
                series = fwd.stack()

        series.name = f"fwd_{horizon}d"
        returns[horizon] = series.dropna()
    return returns


def factor_return_analysis(
    factor_model: ShanzhaiFactorModel,
    token_snapshot: "TokenSnapshot",
    prices: DataFrame,
    market: "MarketSnapshots",
    horizons: Tuple[int, ...] = (1, 3, 7),
) -> pd.DataFrame:
    """Analyse the relationship between the factor and short returns by regime."""

    from .data_structures import MarketSnapshots, TokenSnapshot  # Local import to avoid circular deps

    if not isinstance(token_snapshot, TokenSnapshot):
        raise TypeError("token_snapshot must be a TokenSnapshot instance")
    if not isinstance(market, MarketSnapshots):
        raise TypeError("market must be a MarketSnapshots instance")

    factor_series = factor_model.compute(token_snapshot)
    factor_percentiles = factor_model.rank_percentile(factor_series)

    returns = _future_returns(prices, horizons)
    returns_df = pd.concat(returns.values(), axis=1)

    aligned = pd.concat([factor_series, factor_percentiles, returns_df], axis=1, join="inner")
    aligned.columns = ["factor", "percentile", *[f"fwd_{h}d" for h in horizons]]

    classifier = MarketRegimeClassifier()
    regimes = classifier.classify(market)
    regime_aligned = regimes.reindex(aligned.index.get_level_values(0))
    aligned["regime"] = regime_aligned.values

    records = []
    for regime in MarketRegime:
        regime_slice = aligned[aligned["regime"] == regime]
        if regime_slice.empty:
            continue
        for horizon in horizons:
            col = f"fwd_{horizon}d"
            # Short returns are negative future spot returns.
            short_returns = -regime_slice[col]
            correlation = regime_slice[["factor", col]].corr().iloc[0, 1]
            percentile_q = regime_slice.groupby(
                pd.qcut(regime_slice["percentile"], 5, duplicates="drop"), observed=False
            )[col].mean()
            records.append(
                {
                    "regime": regime.name,
                    "horizon": horizon,
                    "correlation": correlation,
                    "short_mean": short_returns.mean(),
                    "short_sharpe": short_returns.mean() / short_returns.std() if short_returns.std() else float("nan"),
                    "percentile_breakdown": percentile_q.to_dict(),
                }
            )

    return pd.DataFrame.from_records(records)
