"""Implementation of the shanzhai factor construction pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

from pandas import DataFrame, Series

from .data_structures import TokenSnapshot


@dataclass
class ShanzhaiFactorWeights:
    """Weights assigned to each thematic component of the shanzhai factor."""

    fundamentals: float = 0.25
    tokenomics: float = 0.25
    market_microstructure: float = 0.25
    sentiment: float = 0.25

    def as_dict(self) -> Dict[str, float]:
        return {
            "fundamentals": self.fundamentals,
            "tokenomics": self.tokenomics,
            "market_microstructure": self.market_microstructure,
            "sentiment": self.sentiment,
        }


class ShanzhaiFactorModel:
    """Compute a composite shanzhai factor from token level snapshots."""

    def __init__(
        self,
        weights: Optional[ShanzhaiFactorWeights] = None,
        zscore_window: int = 60,
        min_history: int = 30,
    ) -> None:
        """Create a new factor model.

        Parameters
        ----------
        weights:
            Optional custom weight configuration. If ``None`` equal weights are
            used.
        zscore_window:
            Rolling window (in observations) used to z-score each raw feature.
        min_history:
            Minimum number of valid observations required before emitting a
            factor value for a token. This guards against unstable readings when
            a token first lists.
        """

        self.weights = weights or ShanzhaiFactorWeights()
        self.zscore_window = zscore_window
        self.min_history = min_history

    def _zscore_features(self, features: DataFrame) -> DataFrame:
        """Apply rolling z-score normalisation to raw feature values."""

        def compute_group_zscores(group: DataFrame) -> DataFrame:
            return (group - group.rolling(self.zscore_window, min_periods=self.min_history).mean()) / group.rolling(
                self.zscore_window, min_periods=self.min_history
            ).std()

        zscored = features.groupby(level=1, group_keys=False).apply(compute_group_zscores)
        return zscored.dropna(how="all")

    def compute(self, snapshot: TokenSnapshot) -> Series:
        """Compute the composite factor series for each token.

        Parameters
        ----------
        snapshot:
            TokenSnapshot containing the aligned feature sets. Every DataFrame
            in the snapshot must share the same multi-index of (timestamp,
            token).
        """

        aligned = snapshot.aligned_features()
        components = {
            key: aligned.filter(like=f"{key}__") for key in self.weights.as_dict().keys()
        }

        zscored_components = {}
        for key, frame in components.items():
            if frame.empty:
                continue
            zscored_components[key] = self._zscore_features(frame)

        weighted_components: Iterable[Series] = []
        total_weight = 0.0
        for key, zscores in zscored_components.items():
            weight = self.weights.as_dict()[key]
            if weight == 0:
                continue
            weighted_components.append(zscores.mean(axis=1) * weight)
            total_weight += weight

        if total_weight == 0 or not weighted_components:
            raise ValueError("No components available to compute the factor.")

        factor = sum(weighted_components) / total_weight
        factor.name = "shanzhai_factor"
        return factor.dropna()

    def rank_percentile(self, factor: Series) -> Series:
        """Convert factor values to cross-sectional percentiles per timestamp."""

        def percentile(group: Series) -> Series:
            return group.rank(pct=True)

        return factor.groupby(level=0, group_keys=False).apply(percentile)
