# Shanzhai Factor Research Toolkit

This repository bootstraps the quantitative research workflow required to
construct a "山寨因子" (shanzhai factor) for crypto assets and to evaluate the
factor's effectiveness for short-side strategies across different market
regimes.

## Project structure

```
├── src/shanzhai_factor
│   ├── __init__.py               # Public package exports
│   ├── analysis.py               # Factor/return analytics by market regime
│   ├── data_structures.py        # Typed containers for aligned datasets
│   ├── factor_model.py           # Factor construction and weighting logic
│   └── regime.py                 # BTC dominance based regime classification
```

All modules rely on pandas objects and expect time-series indexed by a
``(timestamp, token)`` multi-index. They are intentionally data-source agnostic
so that the research team can plug in proprietary or vendor datasets without
changing the core logic.

## Usage overview

1. **Assemble snapshots** – Load your cleaned feature matrices into a
   `TokenSnapshot`. Columns should be prefixed with thematic namespaces, e.g.
   `fundamentals__whitepaper_similarity`, `tokenomics__unlock_pressure`.
2. **Compute the factor** – Instantiate `ShanzhaiFactorModel` with optional
   weights and call `.compute(snapshot)` to obtain a composite factor score per
   token and timestamp.
3. **Classify regimes** – Use `MarketRegimeClassifier` with a
   `MarketSnapshots` instance populated with BTC dominance data to label each
   observation as BTC 主导期、山寨期或过渡期。
4. **Analyse performance** – Call `factor_return_analysis` with forward price
   data to obtain regime-specific correlation, mean short returns, and Sharpe
   ratios across multiple horizons.

### Reproducing the synthetic experiment

The repository ships with a self-contained script that generates a synthetic
dataset, computes the shanzhai factor, evaluates factor/return correlations by
market regime,并在高分位山寨币上进行做空策略回测：

```bash
python scripts/run_experiment.py
```

结果 CSV 会写入 `reports/outputs/`，并自动生成 Markdown 报告
[`reports/shanzhai_factor_strategy_report.md`](reports/shanzhai_factor_strategy_report.md)，
其中汇总了因子相关性与策略表现的关键发现。

## Next steps

- Connect to real data sources (exchanges, on-chain, sentiment) and implement
  ETL routines that populate the snapshot containers.
- Extend `ShanzhaiFactorModel` with alternative normalisation schemes or
  data-driven weights (e.g. PCA, regression-based loadings).
- Integrate the analytics output with a backtesting framework to convert factor
  insights into executable trading signals.

## Development

Install dependencies and run tests (once implemented):

```bash
pip install -e .[dev]
pytest
```

Testing is not yet configured because the repository currently focuses on the
research pipeline scaffolding.
