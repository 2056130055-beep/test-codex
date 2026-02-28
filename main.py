from __future__ import annotations

from dataclasses import asdict

import numpy as np
from tqdm import tqdm

from analysis.backtest import backtest
from analysis.cointegration import engle_granger_test, is_recently_cointegrated
from analysis.correlation import correlation_matrix, correlation_stability, filter_high_correlation_pairs, full_correlation, rolling_correlation
from analysis.risk import longest_extreme_streak, max_adverse_move, zscore_extremes
from analysis.spread import (
    beta_stability,
    calc_half_life,
    calc_ols_hedge_ratio,
    calc_spread,
    calc_zscore,
    half_life_trend,
    rolling_beta,
    rolling_half_life,
)
from config import Config
from dashboard.builder import build_dashboard
from dashboard.charts import downsample_series
from data_loader import get_price_matrix, get_returns_matrix, load_csv
from signals.generator import generate_signal


def calculate_score(metrics: dict) -> float:
    score = 0.0
    if metrics["pval"] <= 0.01:
        score += 30
    elif metrics["pval"] <= 0.05:
        score += 22
    elif metrics["pval"] <= 0.10:
        score += 12
    roll_corr = metrics.get("roll_corr") or metrics["corr"]
    score += min(25, roll_corr * 25)
    hl = metrics["recent_hl"]
    if 7 <= hl <= 45:
        score += 15
    elif 45 < hl <= 90:
        score += 8
    score += min(10, max(0, metrics["sharpe"] * 4))
    if metrics["recent_coint_stable"]:
        score += 8
    score += metrics["corr_stability"] * 7
    score += metrics["beta_stability"] * 5
    return round(score, 1)


def analyze_pair(t1, t2, log_prices, returns, corr):
    p1 = log_prices[t1]
    p2 = log_prices[t2]
    joined = p1.to_frame("p1").join(p2.to_frame("p2")).dropna()
    if len(joined) < Config.MIN_COMMON_DAYS:
        return None

    eg = engle_granger_test(joined["p1"], joined["p2"])
    spread = eg["spread"]
    z = calc_zscore(spread, Config.ZSCORE_WINDOW).dropna()
    if z.empty:
        return None
    hl = calc_half_life(spread)
    if hl > 999:
        return None

    rb = rolling_beta(joined["p1"], joined["p2"], window=Config.HEDGE_RATIO_WINDOW, step=20)
    rhl = rolling_half_life(spread, window=Config.ROLLING_COINT_WINDOW, step=20)
    btest = backtest(z, Config.ENTRY_ZSCORE, Config.EXIT_ZSCORE, Config.STOP_LOSS_ZSCORE, Config.TRANSACTION_COST)
    ext = zscore_extremes(z)

    metrics = {
        "t1": t1,
        "t2": t2,
        "corr": float(corr),
        "roll_corr": float(rolling_correlation(returns[t1], returns[t2], Config.ROLLING_CORR_WINDOW).mean() or corr),
        "corr_stability": correlation_stability(returns[t1], returns[t2], Config.ROLLING_CORR_WINDOW),
        "pval": eg["pval"],
        "recent_coint_stable": is_recently_cointegrated(spread, Config.ROLLING_COINT_WINDOW, 3),
        "beta": eg["beta"],
        "alpha": eg["intercept"],
        "beta_stability": beta_stability(joined["p1"], joined["p2"], Config.HEDGE_RATIO_WINDOW),
        "half_life": hl,
        "recent_hl": float(rhl["half_life"].iloc[-1]) if not rhl.empty else hl,
        "hl_trend": half_life_trend(rhl["half_life"].tolist()) if not rhl.empty else "stable",
        "zscore": float(z.iloc[-1]),
        "z_max": ext["z_max"],
        "z_min": ext["z_min"],
        "max_streak_days": longest_extreme_streak(z, 2.0),
        "max_adverse_z": max_adverse_move(z, "BUY" if z.iloc[-1] < 0 else "SELL"),
        "sharpe": btest.sharpe_ratio,
        "win_rate": btest.win_rate,
        "n_trades": btest.n_trades,
        "max_dd": btest.max_drawdown,
        "equity_curve": btest.equity_curve[::5],
        "eq_dates": [d.strftime("%Y-%m-%d") for d in z.index[: len(btest.equity_curve)][::5]],
        "p1_last": float(np.exp(joined["p1"].iloc[-1])),
        "p2_last": float(np.exp(joined["p2"].iloc[-1])),
        "n_obs": int(len(joined)),
    }
    metrics["dates"], metrics["zvals"] = downsample_series(z, 3)
    metrics["rb_dates"], metrics["rb_vals"] = ([], []) if rb.empty else (
        [d.strftime("%Y-%m-%d") for d in rb["date"].iloc[::2]],
        [float(v) for v in rb["beta"].iloc[::2]],
    )
    metrics["rhl_dates"], metrics["rhl_vals"] = ([], []) if rhl.empty else (
        [d.strftime("%Y-%m-%d") for d in rhl["date"].iloc[::2]],
        [float(v) for v in rhl["half_life"].iloc[::2]],
    )
    metrics["score"] = calculate_score(metrics)
    return metrics


def run_pipeline(csv_path: str, output_html: str):
    print("Step 1/6: Loading data...")
    df = load_csv(csv_path)
    log_prices = get_price_matrix(df, min_obs=Config.MIN_OBSERVATIONS)
    returns = get_returns_matrix(log_prices)

    print("Step 2/6: Computing correlations...")
    corr_mat = correlation_matrix(returns)
    candidate_pairs = filter_high_correlation_pairs(corr_mat, threshold=Config.MIN_CORRELATION)
    candidate_pairs = candidate_pairs[: Config.MAX_PAIRS_ANALYZE]
    print(f"  {len(candidate_pairs)} candidate pairs (corr > 0.5)")

    print("Step 3/6: Testing cointegration...")
    results = []
    for t1, t2, corr in tqdm(candidate_pairs):
        result = analyze_pair(t1, t2, log_prices, returns, corr)
        if result:
            results.append(result)

    print("Step 4/6: Generating signals...")
    for r in results:
        r["signal"] = generate_signal(r)
    results.sort(key=lambda x: -x["score"])

    print("Step 5/6: Building dashboard...")
    build_dashboard(results, corr_mat, output_html)

    print(f"Step 6/6: Done! → {output_html}")
    print(f"  Pairs analyzed: {len(results)}")
    print(f"  BUY signals: {sum(1 for r in results if r['signal'] == 'BUY')}")
    print(f"  SELL signals: {sum(1 for r in results if r['signal'] == 'SELL')}")


if __name__ == "__main__":
    run_pipeline(Config.CSV_PATH, Config.OUTPUT_HTML)
