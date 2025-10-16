"""Data container definitions for the shanzhai factor pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd


@dataclass
class TokenSnapshot:
    """Holds per-token features required to compute the shanzhai factor.

    Attributes
    ----------
    fundamentals:
        DataFrame indexed by token symbol and timestamp with columns describing
        qualitative and quantitative features (e.g., whitepaper similarity,
        code reuse score).
    tokenomics:
        DataFrame with concentration metrics, unlock schedules, and inflation
        estimates. Expected columns include ``top10_holder_share`` and
        ``unlock_pressure``.
    market_microstructure:
        DataFrame providing liquidity and derivatives data such as funding
        rates and borrow costs.
    sentiment:
        DataFrame capturing social media or news derived sentiment scores.
    """

    fundamentals: pd.DataFrame
    tokenomics: pd.DataFrame
    market_microstructure: pd.DataFrame
    sentiment: pd.DataFrame

    def aligned_features(self) -> pd.DataFrame:
        """Return a feature matrix where all components share the same index.

        The method performs an inner join across the component DataFrames using
        their multi-index of ``(timestamp, token)``. Missing data is left to the
        caller to handle explicitly because forward filling or imputation can
        materially distort the factor.
        """

        frames: Mapping[str, pd.DataFrame] = {
            "fundamentals": self.fundamentals,
            "tokenomics": self.tokenomics,
            "market_microstructure": self.market_microstructure,
            "sentiment": self.sentiment,
        }

        # Ensure all inputs use a consistent sort order for reproducibility.
        aligned = [frame.sort_index() for frame in frames.values()]
        base = aligned[0]
        for frame in aligned[1:]:
            base = base.join(frame, how="inner")
        return base


@dataclass
class MarketSnapshots:
    """Container for aggregate market statistics used in regime detection."""

    btc_dominance: pd.Series
    total_volume: pd.Series

    def dominance_share(self) -> pd.Series:
        """Return the BTC dominance time series as a percentage."""

        return self.btc_dominance.sort_index()
