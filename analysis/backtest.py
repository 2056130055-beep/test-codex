from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    sharpe_ratio: float
    win_rate: float
    n_trades: int
    max_drawdown: float
    total_return: float
    equity_curve: list[float] = field(default_factory=list)
    trade_log: list[dict] = field(default_factory=list)


def backtest(
    zscore: pd.Series,
    entry: float = 2.0,
    exit: float = 0.5,
    stop: float = 3.5,
    fee: float = 0.003,
) -> BacktestResult:
    z = zscore.dropna()
    if z.empty:
        return BacktestResult(0.0, 0.0, 0, 0.0, 0.0, [1.0], [])

    pos = 0
    entry_z = 0.0
    pnl = 0.0
    equity = [1.0]
    daily_rets = []
    trades = []
    start_idx = 0

    for i in range(1, len(z)):
        prev_z = float(z.iloc[i - 1])
        curr_z = float(z.iloc[i])

        if pos == 0:
            if curr_z < -entry:
                pos = 1
                entry_z = curr_z
                start_idx = i
                pnl -= fee * 2
            elif curr_z > entry:
                pos = -1
                entry_z = curr_z
                start_idx = i
                pnl -= fee * 2
        else:
            step_pnl = pos * (curr_z - prev_z) * -0.01
            pnl += step_pnl
            daily_rets.append(step_pnl)
            should_exit = abs(curr_z) < exit
            stopped = abs(curr_z) > stop
            if should_exit or stopped:
                pnl -= fee * 2
                trades.append(
                    {
                        "entry_z": entry_z,
                        "exit_z": curr_z,
                        "pnl": pnl,
                        "duration_days": i - start_idx,
                    }
                )
                pos = 0

        equity.append(equity[-1] + pnl)
        pnl = 0.0

    eq = np.array(equity)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    wins = [1 for t in trades if t["pnl"] > 0]
    sharpe = 0.0
    if len(daily_rets) > 5 and np.std(daily_rets) > 1e-10:
        sharpe = float(np.sqrt(252) * np.mean(daily_rets) / np.std(daily_rets))

    return BacktestResult(
        sharpe_ratio=sharpe,
        win_rate=(sum(wins) / len(trades)) if trades else 0.0,
        n_trades=len(trades),
        max_drawdown=float(dd.min()) if len(dd) else 0.0,
        total_return=float(eq[-1] - 1.0),
        equity_curve=eq.tolist(),
        trade_log=trades,
    )
