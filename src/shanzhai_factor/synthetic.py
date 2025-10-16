"""Synthetic data generation utilities for shanzhai factor experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from .data_structures import MarketSnapshots, TokenSnapshot
from .regime import MarketRegime, MarketRegimeClassifier


@dataclass
class SyntheticConfig:
    """Configuration options for reproducible synthetic datasets."""

    tokens: Tuple[str, ...] = (
        "ALPHA",
        "BETA",
        "GAMMA",
        "DELTA",
        "EPSILON",
        "ZETA",
        "THETA",
        "IOTA",
        "KAPPA",
        "LAMBDA",
    )
    start: str = "2021-01-01"
    periods: int = 240
    seed: int = 7


def _make_index(cfg: SyntheticConfig) -> pd.MultiIndex:
    dates = pd.date_range(cfg.start, periods=cfg.periods, freq="D")
    return pd.MultiIndex.from_product([dates, cfg.tokens], names=["timestamp", "token"])


def _generate_latent_factor(index: pd.MultiIndex, rng: np.random.Generator) -> pd.Series:
    latent: Dict[str, np.ndarray] = {}
    dates = index.get_level_values("timestamp").unique()
    for token in index.get_level_values("token").unique():
        shocks = rng.normal(loc=0.0, scale=0.8, size=len(dates))
        values = np.zeros(len(dates))
        decay = rng.uniform(0.4, 0.7)
        level = rng.normal(loc=0.0, scale=0.5)
        for i, shock in enumerate(shocks):
            if i == 0:
                values[i] = level + shock
            else:
                values[i] = decay * values[i - 1] + shock
        latent[token] = values

    stacked = np.concatenate(list(latent.values()))
    return pd.Series(stacked, index=index, name="latent_shanzhai")


def _feature_frame(
    index: pd.MultiIndex,
    latent: pd.Series,
    prefix: str,
    loadings: Iterable[Tuple[str, float]],
    rng: np.random.Generator,
) -> pd.DataFrame:
    data = {}
    for feature, loading in loadings:
        noise = rng.normal(loc=0.0, scale=0.6, size=len(index))
        data[f"{prefix}__{feature}"] = latent.values * loading + noise
    return pd.DataFrame(data, index=index)


def _simulate_market(cfg: SyntheticConfig, rng: np.random.Generator) -> MarketSnapshots:
    dates = pd.date_range(cfg.start, periods=cfg.periods, freq="D")
    trend = np.linspace(-1.0, 1.0, len(dates))
    seasonal = np.sin(np.linspace(0, 6 * np.pi, len(dates))) * 0.08
    dominance = 0.74 + 0.06 * trend / np.max(np.abs(trend)) + seasonal
    dominance = np.clip(dominance, 0.6, 0.86)
    noise = rng.normal(scale=0.01, size=len(dates))
    dominance = dominance + noise

    dominance_series = pd.Series(dominance, index=dates, name="btc_dominance")
    total_volume = pd.Series(
        rng.lognormal(mean=9.0, sigma=0.2, size=len(dates)), index=dates, name="total_volume"
    )
    return MarketSnapshots(btc_dominance=dominance_series, total_volume=total_volume)


def _simulate_prices(
    index: pd.MultiIndex,
    latent: pd.Series,
    market: MarketSnapshots,
    rng: np.random.Generator,
) -> pd.Series:
    dates = index.get_level_values("timestamp").unique()
    tokens = index.get_level_values("token").unique()
    classifier = MarketRegimeClassifier()
    regimes = classifier.classify(market)

    prices = []
    for token in tokens:
        token_latent = latent.xs(token, level="token")
        level = rng.lognormal(mean=1.5, sigma=0.5)
        series = np.zeros(len(dates))
        series[0] = level
        for i in range(1, len(dates)):
            regime = regimes.iloc[i - 1]
            base_drift = {
                MarketRegime.BTC_DOMINANT: -0.002,
                MarketRegime.ALT_SEASON: -0.006,
                MarketRegime.TRANSITION: -0.004,
            }[regime]
            latent_effect = -0.015 * token_latent.iloc[i - 1]
            vol = 0.02 + 0.01 * abs(token_latent.iloc[i - 1])
            shock = rng.normal(loc=0.0, scale=vol)
            ret = base_drift + latent_effect + shock
            series[i] = max(series[i - 1] * (1 + ret), 0.5)
        prices.append(series)

    stacked = np.concatenate(prices)
    return pd.Series(stacked, index=index, name="close")


def build_snapshots(cfg: SyntheticConfig) -> Tuple[TokenSnapshot, MarketSnapshots, pd.Series]:
    rng = np.random.default_rng(cfg.seed)
    index = _make_index(cfg)
    latent = _generate_latent_factor(index, rng)

    fundamentals = _feature_frame(
        index,
        latent,
        "fundamentals",
        [("whitepaper_similarity", 0.9), ("code_reuse", 0.7)],
        rng,
    )
    tokenomics = _feature_frame(
        index,
        latent,
        "tokenomics",
        [("unlock_pressure", 1.1), ("whale_concentration", 0.8)],
        rng,
    )
    market_micro = _feature_frame(
        index,
        latent,
        "market_microstructure",
        [("funding_rate", 1.0), ("borrow_availability", 0.6)],
        rng,
    )
    sentiment = _feature_frame(
        index,
        latent,
        "sentiment",
        [("social_hype", 0.9), ("news_buzz", 0.75)],
        rng,
    )

    token_snapshot = TokenSnapshot(
        fundamentals=fundamentals,
        tokenomics=tokenomics,
        market_microstructure=market_micro,
        sentiment=sentiment,
    )

    market_snapshot = _simulate_market(cfg, rng)
    prices = _simulate_prices(index, latent, market_snapshot, rng)

    return token_snapshot, market_snapshot, prices
