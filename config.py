class Config:
    # Data
    CSV_PATH = "hose_ohlcv_2021_2025.csv"
    MIN_OBSERVATIONS = 200
    MIN_COMMON_DAYS = 150

    # Correlation filter
    MIN_CORRELATION = 0.50
    ROLLING_CORR_WINDOW = 250
    CORR_STABILITY_THRESHOLD = 0.6

    # Cointegration
    ADF_SIGNIFICANCE = 0.05
    ROLLING_COINT_WINDOW = 120
    ROLLING_COINT_STEP = 20

    # Spread & Z-score
    ZSCORE_WINDOW = 60
    HEDGE_RATIO_WINDOW = 120

    # Half-life
    MAX_HALF_LIFE = 90
    IDEAL_HALF_LIFE_MAX = 45

    # Trading signals
    ENTRY_ZSCORE = 2.0
    EXIT_ZSCORE = 0.5
    STOP_LOSS_ZSCORE = 3.5

    # Backtest
    TRANSACTION_COST = 0.003
    INITIAL_CAPITAL = 1_000_000

    # Scoring weights
    SCORE_PVAL_MAX = 30
    SCORE_CORR_MAX = 25
    SCORE_HALFLIFE_MAX = 15
    SCORE_SHARPE_MAX = 10
    SCORE_STABILITY_MAX = 15
    SCORE_RECENT_COINT_MAX = 8
    SCORE_BETA_STABILITY_MAX = 5

    # Output
    OUTPUT_HTML = "pair_trading_dashboard.html"
    TOP_PAIRS_DISPLAY = 200
    MAX_PAIRS_ANALYZE = 5000
