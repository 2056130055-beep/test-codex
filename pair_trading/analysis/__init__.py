from .backtest import BacktestResult, backtest
from .cointegration import adf_test, engle_granger_test, is_recently_cointegrated, rolling_adf
from .correlation import (
    correlation_matrix,
    correlation_stability,
    filter_high_correlation_pairs,
    full_correlation,
    rolling_correlation,
)
from .risk import (
    estimated_max_loss,
    longest_extreme_streak,
    max_adverse_move,
    risk_score,
    zscore_extremes,
)
from .spread import (
    beta_stability,
    calc_half_life,
    calc_ols_hedge_ratio,
    calc_spread,
    calc_zscore,
    half_life_trend,
    rolling_beta,
    rolling_half_life,
)
