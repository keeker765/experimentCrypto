import pandas as pd

from shanzhai_factor.analysis import _future_returns, factor_return_analysis
from shanzhai_factor.factor_model import ShanzhaiFactorModel, ShanzhaiFactorWeights

from shanzhai_factor.synthetic import SyntheticConfig, build_snapshots


def test_future_returns_accepts_series_and_dataframe() -> None:
    cfg = SyntheticConfig(periods=30)
    snapshot, market, prices = build_snapshots(cfg)
    series_returns = _future_returns(prices, (1,))
    assert 1 in series_returns
    assert isinstance(series_returns[1], pd.Series)

    dataframe_returns = _future_returns(prices.to_frame("close"), (1,))
    assert 1 in dataframe_returns
    assert isinstance(dataframe_returns[1], pd.Series)
    assert series_returns[1].index.names == dataframe_returns[1].index.names


def test_factor_return_analysis_generates_regime_rows() -> None:
    cfg = SyntheticConfig(periods=120)
    snapshot, market, prices = build_snapshots(cfg)
    model = ShanzhaiFactorModel(weights=ShanzhaiFactorWeights(), zscore_window=20, min_history=10)

    result = factor_return_analysis(model, snapshot, prices, market, horizons=(1, 3))
    assert set(result["regime"]) <= {"BTC_DOMINANT", "ALT_SEASON", "TRANSITION"}
    assert {"horizon", "correlation", "short_mean"}.issubset(result.columns)
