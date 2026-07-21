# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')

# %% [markdown]
# # Notebook 3: Trading Strategy Backtest & Performance Analysis
#
# **Strategy**: ML Signal-Based Long-Only Strategy
#   - Go LONG when model predicts UP (signal=1)
#   - Stay in CASH when model predicts DOWN (signal=0)
#   - No short positions (realistic for retail VN market)
#   - Transaction costs: 0.15% per trade (VN brokerage rate)
#
# **Benchmark**: Buy-and-Hold FPT.VN for the same test period
#
# **Performance Metrics**:
#   - Total Return (vs benchmark)
#   - Annualized Sharpe Ratio (rf = 4.5%, VN 1-year gov bond rate)
#   - Maximum Drawdown
#   - Win Rate per Trade
#   - Calmar Ratio (Return / MaxDD)

# %% 1. Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

import os
os.makedirs('images', exist_ok=True)

# %% 2. Load Predictions
test_df = pd.read_csv('data/test_predictions.csv', index_col=0, parse_dates=True)
print(f"[OK] Loaded {len(test_df)} test days")
print(f"     Period: {test_df.index[0].date()} to {test_df.index[-1].date()}")
print(f"     Signal distribution: {test_df['best_signal'].mean():.1%} LONG signals")

# Use XGBoost signal (best model from Notebook 2)
signal         = test_df['best_signal']
actual_returns = test_df['actual_return']
close_prices   = test_df['close']

# %% 3. Backtest Engine

TRANSACTION_COST = 0.0015   # 0.15% per trade (VN brokerage)
RISK_FREE_RATE   = 0.045    # 4.5% annual (VN 1-year government bond)
RF_DAILY         = RISK_FREE_RATE / 252

def run_backtest(signal, actual_returns, name='Strategy', tx_cost=TRANSACTION_COST):
    """
    Simulate a long-only ML signal strategy.
    
    Parameters
    ----------
    signal         : pd.Series of 0/1 -- 1=long, 0=cash
    actual_returns : pd.Series -- next-day realized returns
    name           : strategy label
    tx_cost        : one-way transaction cost (fraction)
    
    Returns
    -------
    dict of performance metrics, pd.Series of daily P&L
    """
    # Identify trade signals (position changes)
    position_changes = signal.diff().fillna(signal)

    # Transaction cost: charged when entering or exiting a position
    cost_series = position_changes.abs() * tx_cost

    # Strategy daily return = signal * asset return - transaction cost
    strategy_ret = signal * actual_returns - cost_series

    # Benchmark: buy-and-hold
    benchmark_ret = actual_returns

    # Cumulative wealth
    cum_strategy  = (1 + strategy_ret).cumprod()
    cum_benchmark = (1 + benchmark_ret).cumprod()

    # -- Performance metrics --
    n_days = len(strategy_ret)
    years  = n_days / 252

    total_return_strat = cum_strategy.iloc[-1] - 1
    total_return_bench = cum_benchmark.iloc[-1] - 1

    ann_return_strat   = (1 + total_return_strat) ** (1 / years) - 1
    ann_return_bench   = (1 + total_return_bench) ** (1 / years) - 1

    ann_vol = strategy_ret.std() * np.sqrt(252)

    # Sharpe ratio (annualized, excess over risk-free)
    excess_daily = strategy_ret - RF_DAILY
    sharpe = (excess_daily.mean() / (excess_daily.std() + 1e-10)) * np.sqrt(252)

    # Maximum drawdown
    rolling_max   = cum_strategy.cummax()
    drawdown      = (cum_strategy - rolling_max) / rolling_max
    max_drawdown  = drawdown.min()

    # Calmar ratio
    calmar = ann_return_strat / abs(max_drawdown + 1e-10)

    # Trade-level stats
    num_long_days = signal.sum()
    num_trades    = (position_changes > 0).sum()   # entries only
    win_days      = (strategy_ret[signal == 1] > 0).mean()

    return {
        'total_return':        total_return_strat,
        'benchmark_return':    total_return_bench,
        'ann_return':          ann_return_strat,
        'ann_vol':             ann_vol,
        'sharpe_ratio':        sharpe,
        'max_drawdown':        max_drawdown,
        'calmar_ratio':        calmar,
        'num_long_days':       int(num_long_days),
        'num_trades':          int(num_trades),
        'win_rate':            win_days,
        'cum_strategy':        cum_strategy,
        'cum_benchmark':       cum_benchmark,
        'drawdown':            drawdown,
        'strategy_ret':        strategy_ret,
    }

# Run backtest for ML strategy
metrics = run_backtest(signal, actual_returns, name='XGBoost Signal')

# Also compute naive baseline: always long (same as benchmark, but confirms our math)
naive_signal = pd.Series(1, index=signal.index)
naive_metrics = run_backtest(naive_signal, actual_returns, name='Always Long', tx_cost=0)

# %% 4. Print Results
print("\n" + "="*65)
print("   BACKTEST RESULTS (ML Signal Strategy vs Buy-and-Hold)")
print("="*65)
print(f"   Test period         : {test_df.index[0].date()} to {test_df.index[-1].date()}")
print(f"   Transaction costs   : {TRANSACTION_COST:.2%} per one-way trade")
print(f"   Risk-free rate      : {RISK_FREE_RATE:.1%} annual (VN gov bond)")
print("-"*65)
print(f"   {'Metric':<30} {'ML Strategy':>15} {'Buy-and-Hold':>15}")
print("-"*65)
print(f"   {'Total Return':<30} {metrics['total_return']:>14.2%} {metrics['benchmark_return']:>14.2%}")
print(f"   {'Annualized Return':<30} {metrics['ann_return']:>14.2%} {naive_metrics['ann_return']:>14.2%}")
print(f"   {'Annualized Volatility':<30} {metrics['ann_vol']:>14.2%} {naive_metrics['ann_vol']:>14.2%}")
print(f"   {'Sharpe Ratio':<30} {metrics['sharpe_ratio']:>14.3f} {naive_metrics['sharpe_ratio']:>14.3f}")
print(f"   {'Maximum Drawdown':<30} {metrics['max_drawdown']:>14.2%} {naive_metrics['max_drawdown']:>14.2%}")
print(f"   {'Calmar Ratio':<30} {metrics['calmar_ratio']:>14.3f} {naive_metrics['calmar_ratio']:>14.3f}")
print(f"   {'Win Rate (long days)':<30} {metrics['win_rate']:>14.2%} {'N/A':>15}")
print(f"   {'Number of Trades':<30} {metrics['num_trades']:>14d} {'N/A':>15}")
print(f"   {'Days in Market':<30} {metrics['num_long_days']:>14d} {len(signal):>15d}")
print(f"   {'Market Exposure':<30} {metrics['num_long_days']/len(signal):>14.1%} {'100.0%':>15}")
print("="*65)

# %% 5. LGD-equivalent Sensitivity: Threshold Analysis
#
# The decision threshold (default=0.50) determines how aggressively we trade.
# A higher threshold = fewer but higher-confidence trades.

print("\nThreshold Sensitivity Analysis:")
print("-"*55)
print(f"  {'Threshold':>10} {'Sharpe':>8} {'Total Ret':>10} {'Max DD':>10} {'# Trades':>10}")
print("-"*55)

threshold_results = []
for thresh in [0.40, 0.45, 0.50, 0.55, 0.60]:
    sig_t   = (test_df['best_prob'] > thresh).astype(int)
    m_t     = run_backtest(sig_t, actual_returns)
    marker  = " <-- baseline" if thresh == 0.50 else ""
    print(f"  {thresh:>10.2f} {m_t['sharpe_ratio']:>8.3f} {m_t['total_return']:>9.2%} "
          f"{m_t['max_drawdown']:>9.2%} {m_t['num_trades']:>10d}{marker}")
    threshold_results.append((thresh, m_t))
print("-"*55)

# %% 6. Visualization

# ---- 6.1 Equity Curve ----
fig, axes = plt.subplots(3, 1, figsize=(14, 11),
                          gridspec_kw={'height_ratios': [3, 1.5, 1.5]})

# Equity curve
ax = axes[0]
ax.plot(metrics['cum_strategy'].index, metrics['cum_strategy'].values * 100 - 100,
        color='#2ecc71', linewidth=2, label='ML Signal Strategy')
ax.plot(metrics['cum_benchmark'].index, metrics['cum_benchmark'].values * 100 - 100,
        color='#3498db', linewidth=2, linestyle='--', label='Buy-and-Hold')
ax.axhline(0, color='gray', linewidth=0.8, linestyle=':')
ax.fill_between(metrics['cum_strategy'].index,
                metrics['cum_strategy'].values * 100 - 100,
                metrics['cum_benchmark'].values * 100 - 100,
                where=(metrics['cum_strategy'].values >= metrics['cum_benchmark'].values),
                alpha=0.15, color='#2ecc71', label='Outperformance')
ax.fill_between(metrics['cum_strategy'].index,
                metrics['cum_strategy'].values * 100 - 100,
                metrics['cum_benchmark'].values * 100 - 100,
                where=(metrics['cum_strategy'].values < metrics['cum_benchmark'].values),
                alpha=0.15, color='#e74c3c')
ax.set_title('ML Signal Strategy vs Buy-and-Hold -- Cumulative Return (%)',
             fontweight='bold', fontsize=13)
ax.set_ylabel('Cumulative Return (%)')
ax.legend(fontsize=11)
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x:+.0f}%'))

# Drawdown
ax = axes[1]
ax.fill_between(metrics['drawdown'].index, metrics['drawdown'].values * 100, 0,
                color='#e74c3c', alpha=0.5)
ax.plot(metrics['drawdown'].index, metrics['drawdown'].values * 100,
        color='#c0392b', linewidth=1)
ax.axhline(metrics['max_drawdown'] * 100, color='#922b21', linestyle='--',
           linewidth=1.2, label=f"Max DD: {metrics['max_drawdown']:.2%}")
ax.set_title('Strategy Drawdown (%)', fontweight='bold', fontsize=12)
ax.set_ylabel('Drawdown (%)')
ax.legend(fontsize=10)
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x:.0f}%'))

# Position (in/out of market)
ax = axes[2]
ax.fill_between(signal.index, signal.values, step='post',
                color='#3498db', alpha=0.4, label='LONG (in market)')
ax.fill_between(signal.index, signal.values, 1, step='post',
                color='#bdc3c7', alpha=0.3, label='CASH (out of market)')
ax.set_title('Market Position (1=Long, 0=Cash)', fontweight='bold', fontsize=12)
ax.set_ylabel('Position')
ax.set_ylim(-0.1, 1.2)
ax.set_yticks([0, 1])
ax.set_yticklabels(['CASH', 'LONG'])
ax.legend(fontsize=10)

for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')

plt.tight_layout()
plt.savefig('images/09_equity_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart saved: images/09_equity_curve.png")

# ---- 6.2 Performance Summary Dashboard ----
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Bar chart: Total Returns
ax = axes[0]
labels = ['ML Strategy', 'Buy-and-Hold']
values = [metrics['total_return'] * 100, metrics['benchmark_return'] * 100]
colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in values]
bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor='white')
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + (1 if val >= 0 else -3),
            f'{val:+.1f}%', ha='center', fontweight='bold', fontsize=12)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_title('Total Return (%)', fontweight='bold', fontsize=12)
ax.set_ylabel('Return (%)')

# Bar chart: Sharpe Ratios
ax = axes[1]
sharpe_vals = [metrics['sharpe_ratio'], naive_metrics['sharpe_ratio']]
colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in sharpe_vals]
bars = ax.bar(labels, sharpe_vals, color=colors, width=0.5, edgecolor='white')
for bar, val in zip(bars, sharpe_vals):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.02,
            f'{val:.2f}', ha='center', fontweight='bold', fontsize=12)
ax.axhline(0, color='black', linewidth=0.8)
ax.axhline(1.0, color='#27ae60', linestyle='--', linewidth=1, alpha=0.7, label='Sharpe = 1.0 threshold')
ax.set_title('Annualized Sharpe Ratio', fontweight='bold', fontsize=12)
ax.set_ylabel('Sharpe Ratio')
ax.legend(fontsize=9)

# Threshold sensitivity: Sharpe vs Threshold
ax = axes[2]
threshs = [x[0] for x in threshold_results]
sharpes = [x[1]['sharpe_ratio'] for x in threshold_results]
colors_t = ['#e74c3c' if s < 0 else '#2ecc71' for s in sharpes]
ax.bar([str(t) for t in threshs], sharpes, color=colors_t, width=0.5, edgecolor='white')
for i, (t, s) in enumerate(zip(threshs, sharpes)):
    ax.text(i, s + 0.01, f'{s:.2f}', ha='center', fontweight='bold', fontsize=11)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_title('Sharpe Ratio by Signal Threshold', fontweight='bold', fontsize=12)
ax.set_xlabel('Probability Threshold')
ax.set_ylabel('Sharpe Ratio')

plt.suptitle('Strategy Performance Dashboard', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('images/10_performance_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart saved: images/10_performance_dashboard.png")

# %% 7. Final Summary
print("\n" + "="*55)
print("   FINAL PORTFOLIO SUMMARY")
print("="*55)
print(f"   Strategy          : ML Long-Only (XGBoost Signal)")
print(f"   Asset             : FPT.VN (Vietnam IT sector)")
print(f"   Test period       : {test_df.index[0].date()} to {test_df.index[-1].date()}")
print(f"   Total Return      : {metrics['total_return']:.2%} (vs {metrics['benchmark_return']:.2%} B&H)")
print(f"   Sharpe Ratio      : {metrics['sharpe_ratio']:.3f}")
print(f"   Max Drawdown      : {metrics['max_drawdown']:.2%}")
print(f"   Calmar Ratio      : {metrics['calmar_ratio']:.3f}")
print(f"   Win Rate          : {metrics['win_rate']:.2%}")
print(f"   Market Exposure   : {metrics['num_long_days']/len(signal):.1%}")
print("="*55)
print("\n=== NOTEBOOK 3 COMPLETE ===")
print("Key results to paste into README:")
print(f"  Total Return (Strategy)  : {metrics['total_return']:.2%}")
print(f"  Total Return (B&H)       : {metrics['benchmark_return']:.2%}")
print(f"  Sharpe Ratio             : {metrics['sharpe_ratio']:.3f}")
print(f"  Max Drawdown             : {metrics['max_drawdown']:.2%}")
print(f"  Win Rate                 : {metrics['win_rate']:.2%}")
