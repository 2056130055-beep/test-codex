from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from pair_trading.analysis.backtest import backtest
from pair_trading.analysis.cointegration import engle_granger_test, is_recently_cointegrated
from pair_trading.analysis.correlation import (
    correlation_matrix,
    correlation_stability,
    filter_high_correlation_pairs,
    full_correlation,
    rolling_correlation,
)
from pair_trading.analysis.risk import (
    longest_extreme_streak,
    max_adverse_move,
    risk_score,
    zscore_extremes,
)
from pair_trading.analysis.spread import (
    beta_stability,
    calc_half_life,
    calc_ols_hedge_ratio,
    calc_spread,
    calc_zscore,
    half_life_trend,
    rolling_beta,
    rolling_half_life,
)
from pair_trading.config import Config
from pair_trading.dashboard.builder import build_dashboard
from pair_trading.dashboard.charts import downsample_series
from pair_trading.data_loader import get_price_matrix, get_returns_matrix, load_csv
from pair_trading.signals.generator import generate_signal


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


def analyze_pair(t1: str, t2: str, log_prices, returns, corr: float):
    p1, p2 = log_prices[t1], log_prices[t2]
    common = p1.dropna().index.intersection(p2.dropna().index)
    if len(common) < Config.MIN_COMMON_DAYS:
        return None
    p1c, p2c = p1.loc[common], p2.loc[common]
    r1, r2 = returns[t1].loc[common].dropna(), returns[t2].loc[common].dropna()

    corr_full = full_correlation(r1, r2)
    roll_corr_s = rolling_correlation(r1, r2, Config.ROLLING_CORR_WINDOW)
    corr_stab = correlation_stability(r1, r2, Config.ROLLING_CORR_WINDOW)

    alpha, beta = calc_ols_hedge_ratio(p1c, p2c)
    spread = calc_spread(p1c, p2c, beta=beta, alpha=alpha)
    eg = engle_granger_test(p1c, p2c)

    z = calc_zscore(spread, Config.ZSCORE_WINDOW)
    current_z = float(z.dropna().iloc[-1]) if len(z.dropna()) else 0.0
    hl = calc_half_life(spread)

    rb = rolling_beta(p1c, p2c, window=Config.HEDGE_RATIO_WINDOW, step=20)
    bstab = beta_stability(p1c, p2c, window=Config.HEDGE_RATIO_WINDOW)
    rhl = rolling_half_life(spread, window=Config.ROLLING_COINT_WINDOW, step=20)
    recent_hl = float(rhl["half_life"].dropna().iloc[-1]) if len(rhl) and len(rhl["half_life"].dropna()) else hl

    bt = backtest(z, Config.ENTRY_ZSCORE, Config.EXIT_ZSCORE, Config.STOP_LOSS_ZSCORE, Config.TRANSACTION_COST)

    extreme = zscore_extremes(z)
    streak = longest_extreme_streak(z, threshold=Config.ENTRY_ZSCORE)
    metrics = {
        "t1": t1,
        "t2": t2,
        "corr": float(corr_full),
        "roll_corr": float(roll_corr_s.mean()) if len(roll_corr_s) else float(corr_full),
        "corr_stability": float(corr_stab),
        "pval": float(eg["pval"]),
        "recent_coint_stable": is_recently_cointegrated(spread, window=Config.ROLLING_COINT_WINDOW, n_recent=3),
        "beta": float(beta),
        "alpha": float(alpha),
        "beta_stability": float(bstab),
        "half_life": float(hl),
        "recent_hl": float(recent_hl),
        "hl_trend": half_life_trend(rhl["half_life"].tolist() if len(rhl) else []),
        "zscore": current_z,
        "z_max": extreme["z_max"],
        "z_min": extreme["z_min"],
        "max_streak_days": streak,
        "max_adverse_z": 0.0,
        "sharpe": float(bt.sharpe_ratio),
        "win_rate": float(bt.win_rate),
        "n_trades": int(bt.n_trades),
        "max_dd": float(bt.max_drawdown),
        "equity_curve": [float(x) for x in bt.equity_curve[::5]],
        "eq_dates": [str(d.date()) for d in z.dropna().index[::5]][: len(bt.equity_curve[::5])],
        "p1_last": float(np.exp(p1c.iloc[-1])),
        "p2_last": float(np.exp(p2c.iloc[-1])),
        "n_obs": int(len(common)),
    }
    metrics["signal"] = generate_signal(metrics)
    metrics["max_adverse_z"] = max_adverse_move(z, metrics["signal"])
    metrics["risk"] = risk_score(
        {
            "sharpe": metrics["sharpe"],
            "win_rate": metrics["win_rate"],
            "max_dd": metrics["max_dd"],
            "streak": metrics["max_streak_days"],
            "adverse_move": metrics["max_adverse_z"],
        }
    )
    metrics["score"] = calculate_score(metrics)

    dates, zvals = downsample_series(z, step=3)
    metrics["dates"], metrics["zvals"] = dates, zvals
    metrics["rb_dates"] = rb["date"].tolist() if len(rb) else []
    metrics["rb_vals"] = rb["beta"].round(4).tolist() if len(rb) else []
    metrics["rhl_dates"] = rhl["date"].tolist() if len(rhl) else []
    metrics["rhl_vals"] = rhl["half_life"].round(2).tolist() if len(rhl) else []

    return metrics


def run_pipeline(csv_path: str, output_html: str):
    print("Step 1/6: Loading data...")
    df = load_csv(csv_path)
    log_prices = get_price_matrix(df, min_obs=Config.MIN_OBSERVATIONS)
    returns = get_returns_matrix(log_prices)

    print("Step 2/6: Computing correlations...")
    corr_mx = correlation_matrix(returns)
    candidate_pairs = filter_high_correlation_pairs(corr_mx, threshold=Config.MIN_CORRELATION)
    candidate_pairs = candidate_pairs[: Config.MAX_PAIRS_ANALYZE]
    print(f"  {len(candidate_pairs)} candidate pairs (corr > {Config.MIN_CORRELATION})")

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
    build_dashboard(results, corr_mx, output_html)

    print(f"Step 6/6: Done! → {output_html}")
    print(f"  Pairs analyzed: {len(results)}")
    print(f"  BUY signals: {sum(1 for r in results if r['signal'] == 'BUY')}")
    print(f"  SELL signals: {sum(1 for r in results if r['signal'] == 'SELL')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=Config.CSV_PATH)
    parser.add_argument("--output", default=Config.OUTPUT_HTML)
    args = parser.parse_args()

    if not Path(args.csv).exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")
    run_pipeline(args.csv, args.output)


if __name__ == "__main__":
    main()
