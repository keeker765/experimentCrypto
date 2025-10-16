"""Market regime classification utilities."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Tuple

import pandas as pd

from .data_structures import MarketSnapshots


class MarketRegime(Enum):
    """Enumeration of market regimes used in the study."""

    BTC_DOMINANT = auto()
    ALT_SEASON = auto()
    TRANSITION = auto()


@dataclass
class MarketRegimeClassifier:
    """Classify market regimes based on BTC dominance share."""

    dominant_threshold: float = 0.75
    transition_band: Tuple[float, float] = (0.65, 0.85)

    def classify(self, market: MarketSnapshots) -> pd.Series:
        """Return the regime label for each timestamp."""

        dominance = market.dominance_share()
        low, high = self.transition_band
        regimes = pd.Series(index=dominance.index, dtype="object")
        regimes[dominance >= self.dominant_threshold] = MarketRegime.BTC_DOMINANT
        regimes[dominance < self.dominant_threshold] = MarketRegime.ALT_SEASON
        regimes[(dominance >= low) & (dominance <= high)] = MarketRegime.TRANSITION
        return regimes
