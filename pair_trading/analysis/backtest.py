from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    n_trades: int = 0
    max_drawdown: float = 0.0
    total_return: float = 0.0
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
    if len(z) == 0:
        return BacktestResult(equity_curve=[1.0])

    position = 0
    entry_z = 0.0
    entry_idx = None
    equity = 1.0
    eq = []
    trades = []

    for i, (_, val) in enumerate(z.items()):
        pnl_today = 0.0
        if position == 0:
            if val < -entry:
                position = 1
                entry_z = float(val)
                entry_idx = i
                equity *= 1 - fee
            elif val > entry:
                position = -1
                entry_z = float(val)
                entry_idx = i
                equity *= 1 - fee
        else:
            pnl_today = position * (val - z.iloc[i - 1]) * -0.01 if i > 0 else 0.0
            equity *= 1 + pnl_today
            should_exit = abs(val) < exit or abs(val) > stop
            if should_exit:
                equity *= 1 - fee
                trade_pnl = (equity - 1.0) if len(trades) == 0 else (equity / (eq[-1] if eq else 1.0) - 1.0)
                trades.append(
                    {
                        "entry_z": entry_z,
                        "exit_z": float(val),
                        "pnl": float(trade_pnl),
                        "duration_days": i - (entry_idx or i),
                    }
                )
                position = 0
        eq.append(equity)

    if not eq:
        eq = [1.0]

    curve = np.array(eq)
    rets = np.diff(curve) / np.maximum(curve[:-1], 1e-9) if len(curve) > 1 else np.array([])
    sharpe = float(np.sqrt(252) * rets.mean() / rets.std()) if len(rets) > 2 and rets.std() > 0 else 0.0
    wins = sum(1 for t in trades if t["pnl"] > 0)
    dd = curve / np.maximum.accumulate(curve) - 1

    return BacktestResult(
        sharpe_ratio=sharpe,
        win_rate=(wins / len(trades)) if trades else 0.0,
        n_trades=len(trades),
        max_drawdown=float(dd.min()) if len(dd) else 0.0,
        total_return=float(curve[-1] - 1),
        equity_curve=curve.tolist(),
        trade_log=trades,
    )
