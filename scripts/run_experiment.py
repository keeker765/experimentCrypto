"""Run a full shanzhai factor experiment using synthetic data."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from shanzhai_factor.analysis import factor_return_analysis
from shanzhai_factor.factor_model import ShanzhaiFactorModel, ShanzhaiFactorWeights
from shanzhai_factor.regime import MarketRegime, MarketRegimeClassifier
from shanzhai_factor.synthetic import SyntheticConfig, build_snapshots


def _strategy_performance(
    factor: pd.Series,
    percentiles: pd.Series,
    prices: pd.Series,
    market: "MarketSnapshots",
    horizon: int = 3,
) -> pd.DataFrame:
    from shanzhai_factor.data_structures import MarketSnapshots  # Local import to avoid cycle in type checking

    if not isinstance(market, MarketSnapshots):
        raise TypeError("market must be a MarketSnapshots instance")

    classifier = MarketRegimeClassifier()
    regimes = classifier.classify(market)

    future_price = prices.groupby(level="token").shift(-horizon)
    current_price = prices
    short_returns = (current_price - future_price) / current_price

    frame = pd.concat(
        {
            "factor": factor,
            "percentile": percentiles,
            "short_return": short_returns,
        },
        axis=1,
    ).dropna()

    selections = frame[frame["percentile"] >= 0.8]
    daily_returns = selections.groupby(level="timestamp")["short_return"].mean()
    aligned_regimes = regimes.reindex(daily_returns.index)

    summary = []
    for regime in MarketRegime:
        mask = aligned_regimes == regime
        if not mask.any():
            continue
        regime_returns = daily_returns[mask]
        mean = regime_returns.mean()
        std = regime_returns.std()
        sharpe = mean / std if std else float("nan")
        hit_rate = (regime_returns > 0).mean()
        summary.append(
            {
                "regime": regime.name,
                "trading_days": int(mask.sum()),
                "mean_return": mean,
                "std_return": std,
                "sharpe": sharpe,
                "hit_rate": hit_rate,
            }
        )

    overall = daily_returns.mean()
    overall_std = daily_returns.std()
    summary.append(
        {
            "regime": "OVERALL",
            "trading_days": int(len(daily_returns)),
            "mean_return": overall,
            "std_return": overall_std,
            "sharpe": overall / overall_std if overall_std else float("nan"),
            "hit_rate": (daily_returns > 0).mean(),
        }
    )

    return pd.DataFrame(summary)


def _markdown_table(
    frame: pd.DataFrame,
    *,
    index_label: str,
    column_suffix: str = "",
    percent_columns: Iterable[str] | None = None,
    int_columns: Iterable[str] | None = None,
    precision: int = 2,
) -> str:
    """Render a simple Markdown table with optional percent formatting."""

    percent_columns = set(percent_columns or [])
    int_columns = set(int_columns or [])
    headers = [index_label, *[f"{col}{column_suffix}" for col in frame.columns]]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for index, row in frame.iterrows():
        formatted = []
        for col, value in row.items():
            if pd.isna(value):
                formatted.append("N/A")
                continue
            if col in percent_columns:
                formatted.append(f"{value * 100:.{precision}f}%")
            elif col in int_columns:
                formatted.append(f"{int(round(value))}")
            else:
                formatted.append(f"{value:.{precision}f}")
        lines.append("| " + " | ".join([str(index), *formatted]) + " |")

    return "\n".join(lines)


def _write_markdown_report(
    output_dir: Path,
    analysis: pd.DataFrame,
    strategy: pd.DataFrame,
    cfg: SyntheticConfig,
    factor_model: ShanzhaiFactorModel,
    weights: ShanzhaiFactorWeights,
) -> Path:
    """Create a Markdown report that summarises factor correlations and strategy tests."""

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir.parent / "shanzhai_factor_strategy_report.md"

    corr_table = analysis.pivot(index="regime", columns="horizon", values="correlation").sort_index()
    mean_table = analysis.pivot(index="regime", columns="horizon", values="short_mean").sort_index()
    sharpe_table = analysis.pivot(index="regime", columns="horizon", values="short_sharpe").sort_index()

    order = [regime.name for regime in MarketRegime] + ["OVERALL"]
    strategy_sorted = strategy.set_index("regime").reindex(order).dropna(how="all")

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")

    top_corr_row = analysis.loc[analysis["correlation"].idxmax()]
    weakest_corr_row = analysis.loc[analysis["correlation"].idxmin()]
    best_short_row = analysis.loc[analysis["short_mean"].idxmax()]

    regime_only = strategy[strategy["regime"] != "OVERALL"]
    top_sharpe = (
        regime_only.loc[regime_only["sharpe"].idxmax()] if not regime_only.empty else None
    )
    worst_sharpe = (
        regime_only.loc[regime_only["sharpe"].idxmin()] if not regime_only.empty else None
    )

    lines = [
        "# 山寨因子相关性与策略测试报告",
        "",
        f"- 数据集：{len(cfg.tokens)} 个代币、{cfg.periods} 个交易日的合成样本",
        f"- 因子构建：滚动窗口 {factor_model.zscore_window} 日标准化，最小历史 {factor_model.min_history} 日，权重设定 {weights}",
        f"- 报告生成时间：{generated_at}",
        "",
        "## 因子相关性分析",
        "",  # spacing
        "### 相关系数",
        _markdown_table(corr_table, index_label="市场时期", column_suffix="日"),
        "",
        "### 做空收益均值",
        _markdown_table(mean_table, index_label="市场时期", column_suffix="日", percent_columns=set(mean_table.columns)),
        "",
        "### 做空收益 Sharpe",
        _markdown_table(sharpe_table, index_label="市场时期", column_suffix="日"),
        "",
        "#### 重点发现",
        f"- 最高相关性出现在 **{top_corr_row['regime']}** 的 **{int(top_corr_row['horizon'])} 日** 窗口，系数为 {top_corr_row['correlation']:.2f}，显示高山寨因子与做空收益强烈联动。",
        f"- 最弱相关性出现在 **{weakest_corr_row['regime']}** 的 **{int(weakest_corr_row['horizon'])} 日** 窗口，系数为 {weakest_corr_row['correlation']:.2f}，提示该 regime 中需谨慎依赖因子。",
        f"- 做空收益均值最高的组合来自 **{best_short_row['regime']}** 在 **{int(best_short_row['horizon'])} 日** 窗口，平均收益 {best_short_row['short_mean'] * 100:.2f}% 。",
        "",
        "## 策略测试表现",
        "",
        "策略规则：每日选取山寨因子分位 ≥80% 的代币，等权做空并持有 3 日。",
        "",
    ]

    strategy_percent_cols = {"mean_return", "std_return", "hit_rate"}
    lines.append(
        _markdown_table(
            strategy_sorted,
            index_label="市场时期",
            percent_columns=strategy_percent_cols,
            int_columns={"trading_days"},
            column_suffix="",
            precision=2,
        )
    )

    lines.extend(
        [
            "",
            "### 策略解读",
        ]
    )

    if top_sharpe is not None:
        lines.append(
            f"- Sharpe 比率最高的 regime 为 **{top_sharpe['regime']}** (Sharpe {top_sharpe['sharpe']:.2f})，命中率 {top_sharpe['hit_rate'] * 100:.1f}% 。"
        )
    if worst_sharpe is not None:
        lines.append(
            f"- 表现最弱的 regime 为 **{worst_sharpe['regime']}** (Sharpe {worst_sharpe['sharpe']:.2f})，建议降低仓位或暂停交易。"
        )

    overall_row = strategy_sorted.loc["OVERALL"] if "OVERALL" in strategy_sorted.index else None
    if overall_row is not None:
        lines.append(
            f"- 全局来看，策略平均收益 {overall_row['mean_return'] * 100:.2f}% 、Sharpe {overall_row['sharpe']:.2f}，命中率 {overall_row['hit_rate'] * 100:.1f}% 。"
        )

    lines.extend(
        [
            "",
            "## 后续优化建议",
            "- 引入真实市场数据验证因子稳定性，并与资金费率、借币成本合并评估净收益。",
            "- 结合 regime 过滤调节仓位，过渡期可缩短持有期或动态对冲主流币风险。",
            "- 增补风控模块，例如波动率目标和尾部保护 (期权/止损)。",
            "",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    cfg = SyntheticConfig()
    token_snapshot, market_snapshot, prices = build_snapshots(cfg)

    weights = ShanzhaiFactorWeights(
        fundamentals=0.2,
        tokenomics=0.3,
        market_microstructure=0.3,
        sentiment=0.2,
    )
    factor_model = ShanzhaiFactorModel(weights=weights, zscore_window=45, min_history=20)

    price_frame = prices.to_frame("close")

    analysis = factor_return_analysis(
        factor_model=factor_model,
        token_snapshot=token_snapshot,
        prices=price_frame,
        market=market_snapshot,
        horizons=(1, 3, 7),
    )

    factor_series = factor_model.compute(token_snapshot)
    percentiles = factor_model.rank_percentile(factor_series)
    strategy = _strategy_performance(factor_series, percentiles, price_frame["close"], market_snapshot)

    output_dir = Path(__file__).resolve().parents[1] / "reports" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    factor_output = output_dir / "factor_regime_analysis.csv"
    strategy_output = output_dir / "short_strategy_performance.csv"

    analysis.to_csv(factor_output, index=False)
    strategy.to_csv(strategy_output, index=False)
    report_path = _write_markdown_report(output_dir, analysis, strategy, cfg, factor_model, weights)

    print("Factor/regime analysis saved to", factor_output)
    print("Strategy performance saved to", strategy_output)
    print("Markdown report saved to", report_path)


if __name__ == "__main__":
    main()
