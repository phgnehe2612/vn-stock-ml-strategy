# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')

# %% [markdown]
# # Notebook 1: Data Collection, EDA & Feature Engineering
#
# **Asset**: FPT Corporation (FPT.VN) -- Vietnam's largest listed IT company
# **Period**: 2021-01-01 to 2024-12-31 (4 years of daily OHLCV data)
# **Goal**: Download market data, compute 15+ technical indicators as ML features,
#           and engineer a binary target variable (price direction: UP=1, DOWN=0)
#
# All features are computed without look-ahead bias (only past data used).
# The final feature matrix is saved to data/features.csv for Notebook 2.

# %% 1. Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
import seaborn as sns
import yfinance as yf
import warnings
import os
warnings.filterwarnings('ignore')

os.makedirs('images', exist_ok=True)
os.makedirs('data',   exist_ok=True)

plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['font.size']      = 11
plt.rcParams['axes.spines.top']   = False
plt.rcParams['axes.spines.right'] = False

# %% 2. Download Data
TICKER     = 'FPT.VN'
START_DATE = '2021-01-01'
END_DATE   = '2024-12-31'

print(f"Downloading {TICKER} ({START_DATE} to {END_DATE})...")
raw = yf.download(TICKER, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)

# Handle both single-level and multi-level column index (yfinance version differences)
if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = raw.columns.droplevel(1)

raw.index = pd.to_datetime(raw.index)
raw = raw[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
raw = raw.dropna()

print(f"[OK] Downloaded {len(raw)} trading days")
print(f"     Date range: {raw.index[0].date()} to {raw.index[-1].date()}")
print(f"     Price range: {raw['Close'].min():.0f} - {raw['Close'].max():.0f} VND")

# %% 3. Technical Indicator Functions (implemented from scratch -- no ta-lib dependency)

def compute_rsi(series, period=14):
    """Relative Strength Index -- momentum oscillator (0-100)."""
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    """Moving Average Convergence Divergence -- trend/momentum signal."""
    ema_fast    = series.ewm(span=fast,   adjust=False).mean()
    ema_slow    = series.ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram

def compute_bollinger(series, window=20, num_std=2):
    """Bollinger Bands -- volatility & mean-reversion signal."""
    sma      = series.rolling(window).mean()
    std      = series.rolling(window).std()
    upper    = sma + num_std * std
    lower    = sma - num_std * std
    bb_pct   = (series - lower) / (upper - lower + 1e-10)   # position within bands (0-1)
    bb_width = (upper - lower) / (sma + 1e-10)              # band width (normalized)
    return upper, lower, bb_pct, bb_width

def compute_atr(high, low, close, period=14):
    """Average True Range -- volatility measure."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low  - close.shift(1)).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()

# %% 4. Feature Engineering
df = raw.copy()

close  = df['Close']
high   = df['High']
low    = df['Low']
volume = df['Volume']

# -- Return-based features (lag returns) --
df['ret_1d']  = close.pct_change(1)
df['ret_3d']  = close.pct_change(3)
df['ret_5d']  = close.pct_change(5)
df['ret_10d'] = close.pct_change(10)
df['ret_20d'] = close.pct_change(20)

# -- Moving averages --
df['sma_5']  = close.rolling(5).mean()
df['sma_10'] = close.rolling(10).mean()
df['sma_20'] = close.rolling(20).mean()
df['sma_50'] = close.rolling(50).mean()

# -- Price relative to moving averages (normalized) --
df['price_sma5_ratio']  = close / df['sma_5']  - 1
df['price_sma20_ratio'] = close / df['sma_20'] - 1
df['sma5_sma20_ratio']  = df['sma_5'] / df['sma_20'] - 1   # golden/death cross signal

# -- Momentum: RSI --
df['rsi_14'] = compute_rsi(close, 14)
df['rsi_7']  = compute_rsi(close, 7)

# -- Trend: MACD --
df['macd'], df['macd_signal'], df['macd_hist'] = compute_macd(close)
df['macd_norm']  = df['macd']  / (close + 1e-10)    # normalize by price level
df['macd_cross'] = (df['macd'] > df['macd_signal']).astype(int)   # 1 = bullish cross

# -- Volatility: Bollinger Bands --
_, _, df['bb_pct'], df['bb_width'] = compute_bollinger(close, 20, 2)

# -- Volatility: ATR (normalized by price) --
df['atr_14']      = compute_atr(high, low, close, 14)
df['atr_14_norm'] = df['atr_14'] / close

# -- Volume: relative to 20-day average --
df['vol_sma20']    = volume.rolling(20).mean()
df['vol_ratio']    = volume / (df['vol_sma20'] + 1e-10)
df['vol_price_up'] = ((close > close.shift(1)) & (volume > df['vol_sma20'])).astype(int)

# -- Target variable: next-day direction (1=UP, 0=DOWN/FLAT) --
# Shift by -1: today's features predict tomorrow's direction
df['return_next_1d'] = close.pct_change(1).shift(-1)
df['target']         = (df['return_next_1d'] > 0).astype(int)

# Drop rows with NaN (warmup period for indicators)
df_clean = df.dropna().copy()

FEATURE_COLS = [
    'ret_1d', 'ret_3d', 'ret_5d', 'ret_10d', 'ret_20d',
    'price_sma5_ratio', 'price_sma20_ratio', 'sma5_sma20_ratio',
    'rsi_14', 'rsi_7',
    'macd_norm', 'macd_hist', 'macd_cross',
    'bb_pct', 'bb_width',
    'atr_14_norm',
    'vol_ratio', 'vol_price_up'
]

print(f"\n[OK] Feature engineering complete")
print(f"     Clean rows (after warmup removal): {len(df_clean)}")
print(f"     Features: {len(FEATURE_COLS)}")
print(f"     Target balance: {df_clean['target'].mean():.1%} UP days")
print(f"\nFeature list:")
for i, f in enumerate(FEATURE_COLS, 1):
    print(f"  {i:2d}. {f}")

# %% 5. EDA Charts
# ---- 5.1 Price & Volume overview ----
fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1, 1]})

ax = axes[0]
ax.plot(df_clean.index, df_clean['Close'], color='#2c3e50', linewidth=1.3, label='FPT.VN Close')
ax.plot(df_clean.index, df_clean['sma_20'], color='#e74c3c', linewidth=1, linestyle='--', alpha=0.8, label='SMA-20')
ax.plot(df_clean.index, df_clean['sma_50'], color='#3498db', linewidth=1, linestyle='--', alpha=0.8, label='SMA-50')
ax.set_title('FPT.VN Price History with Moving Averages (2021-2024)', fontweight='bold', fontsize=13)
ax.set_ylabel('Price (VND)', fontsize=11)
ax.legend(fontsize=10)
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x:,.0f}'))

ax = axes[1]
ax.plot(df_clean.index, df_clean['rsi_14'], color='#9b59b6', linewidth=1.2)
ax.axhline(70, color='#e74c3c', linestyle='--', linewidth=0.8, alpha=0.7)
ax.axhline(30, color='#2ecc71', linestyle='--', linewidth=0.8, alpha=0.7)
ax.fill_between(df_clean.index, 70, df_clean['rsi_14'].clip(70),
                alpha=0.2, color='#e74c3c')
ax.fill_between(df_clean.index, df_clean['rsi_14'].clip(upper=30), 30,
                alpha=0.2, color='#2ecc71')
ax.set_ylabel('RSI (14)', fontsize=11)
ax.set_ylim(0, 100)

ax = axes[2]
vol_colors = ['#2ecc71' if r > 0 else '#e74c3c' for r in df_clean['ret_1d']]
ax.bar(df_clean.index, df_clean['Volume'], color=vol_colors, alpha=0.6, width=1)
ax.plot(df_clean.index, df_clean['vol_sma20'], color='#2c3e50', linewidth=1, label='Vol SMA-20')
ax.set_ylabel('Volume', fontsize=11)
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x/1e6:.0f}M'))
ax.legend(fontsize=9)

for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')

plt.tight_layout()
plt.savefig('images/01_price_volume_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart saved: images/01_price_volume_overview.png")

# ---- 5.2 Feature Distributions by Target ----
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
selected_features = ['rsi_14', 'macd_norm', 'bb_pct', 'ret_5d', 'vol_ratio', 'atr_14_norm']
feature_labels    = ['RSI-14', 'MACD (normalized)', 'BB Position', '5-Day Return',
                     'Volume Ratio', 'ATR-14 (normalized)']

for ax, feat, label in zip(axes.flat, selected_features, feature_labels):
    for t, color, lbl in [(1, '#2ecc71', 'UP Day'), (0, '#e74c3c', 'DOWN Day')]:
        subset = df_clean.loc[df_clean['target'] == t, feat]
        ax.hist(subset, bins=35, alpha=0.55, color=color, label=lbl,
                edgecolor='white', density=True)
    ax.set_title(label, fontweight='bold', fontsize=11)
    ax.set_ylabel('Density', fontsize=9)
    ax.legend(fontsize=8)

plt.suptitle('Feature Distributions: UP Days vs DOWN Days', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('images/02_feature_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart saved: images/02_feature_distributions.png")

# ---- 5.3 Correlation heatmap ----
fig, ax = plt.subplots(figsize=(12, 9))
corr_matrix = df_clean[FEATURE_COLS + ['target']].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, annot=False, fmt='.2f', cmap='RdYlGn',
            center=0, ax=ax, linewidths=0.3, mask=mask,
            cbar_kws={'shrink': 0.8})
ax.set_title('Feature Correlation Matrix', fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig('images/03_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart saved: images/03_correlation_heatmap.png")

# ---- 5.4 Daily return distribution ----
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

ax = axes[0]
ax.hist(df_clean['ret_1d'] * 100, bins=60, color='#3498db', alpha=0.75, edgecolor='white')
ax.axvline(0, color='black', linewidth=1)
ax.axvline(df_clean['ret_1d'].mean()*100, color='#e74c3c', linestyle='--', linewidth=1.5,
           label=f"Mean: {df_clean['ret_1d'].mean()*100:.2f}%")
ax.set_title('Daily Return Distribution', fontweight='bold', fontsize=13)
ax.set_xlabel('Daily Return (%)')
ax.set_ylabel('Frequency')
ax.legend(fontsize=10)

ax = axes[1]
target_counts = df_clean['target'].value_counts()
colors = ['#2ecc71', '#e74c3c']
bars = ax.bar(['UP Day (target=1)', 'DOWN Day (target=0)'], target_counts.values,
              color=colors, width=0.5, edgecolor='white')
ax.set_title('Target Variable Balance', fontweight='bold', fontsize=13)
ax.set_ylabel('Count')
for bar, val in zip(bars, target_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            f'{val} ({val/len(df_clean):.1%})', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('images/04_return_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart saved: images/04_return_distribution.png")

# %% 6. Save Feature Matrix
save_cols = FEATURE_COLS + ['Close', 'return_next_1d', 'target']
df_clean[save_cols].to_csv('data/features.csv')
print(f"\n[OK] Feature matrix saved: data/features.csv ({len(df_clean)} rows x {len(FEATURE_COLS)} features)")

# Summary
print("\n" + "="*55)
print("   NOTEBOOK 1 SUMMARY")
print("="*55)
print(f"   Ticker        : {TICKER}")
print(f"   Period        : {df_clean.index[0].date()} to {df_clean.index[-1].date()}")
print(f"   Trading days  : {len(df_clean)}")
print(f"   Features      : {len(FEATURE_COLS)}")
print(f"   UP days       : {df_clean['target'].sum()} ({df_clean['target'].mean():.1%})")
print(f"   DOWN days     : {(1-df_clean['target']).sum()} ({1-df_clean['target'].mean():.1%})")
print(f"   Avg daily ret : {df_clean['ret_1d'].mean()*100:.3f}%")
print(f"   Volatility    : {df_clean['ret_1d'].std()*100:.2f}% per day")
print("="*55)
print("=== NOTEBOOK 1 COMPLETE ===")
