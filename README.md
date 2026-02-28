# Pair Trading Analysis Tool

Standalone HOSE pair-trading analyzer that loads OHLCV CSV data, computes correlation + cointegration metrics, generates signals, backtests z-score mean reversion rules, and builds an interactive HTML dashboard.

## Run

```bash
pip install -r requirements.txt
python main.py
```

Input CSV must include columns: `date,ticker,open,high,low,close,volume`.

Output file: `pair_trading_dashboard.html`.
