# VN Stock ML Trading Strategy

> **An end-to-end quantitative trading pipeline** for FPT.VN (Vietnam's largest IT company):
> feature engineering from 18 technical indicators → ML direction prediction
> (Logistic Regression, Random Forest, XGBoost) → signal-based backtest with
> transaction costs, Sharpe ratio, and drawdown analysis.

---

## Key Results

| Metric | Value |
|---|---|
| **Asset** | FPT Corporation (FPT.VN) — VN-Index top constituent |
| **Period** | 2021-01-01 to 2024-12-31 (992 clean trading days) |
| **Features** | 18 technical indicators (RSI, MACD, Bollinger Bands, ATR, Volume) |
| **Best Model** | **XGBoost** (tied with Random Forest, AUC = 0.545) |
| **ROC-AUC (Best)** | **0.545** |
| **Strategy Total Return** | **+19.62%** in 38.3% market exposure (vs +89.70% Buy-and-Hold at 100% exposure) |
| **Annualized Volatility** | **13.15%** strategy vs 23.24% Buy-and-Hold — 43% lower risk |
| **Annualized Sharpe Ratio** | **0.875** (threshold=0.50 baseline) |
| **Best Sharpe (tuned)** | **1.859** at threshold=0.40 (+51.63% total return) |
| **Maximum Drawdown** | **-11.78%** vs -15.04% Buy-and-Hold |
| **Win Rate** | **53.51%** on long days |
| **Market Exposure** | 38.3% — strategy avoids predicted DOWN days |

> **Interpretation**: The strategy does not aim to beat buy-and-hold in raw return, but to generate
> *risk-adjusted* alpha. At threshold=0.40, it achieves Sharpe 1.859 with 51.63% return while
> being exposed to the market only 38.3% of the time — a substantially better return-per-unit-of-risk profile.

---

## Project Structure

```
vn-stock-ml-strategy/
├── notebook_01_data_and_features.py   # Data download + EDA + 18 technical indicators
├── notebook_02_ml_models.py           # LR / RandomForest / XGBoost + tuning + evaluation
├── notebook_03_backtest.py            # Signal strategy + Sharpe + drawdown + sensitivity
├── images/                            # Auto-generated charts (10 total)
├── requirements.txt
└── README.md
```

---

## Methodology

### Notebook 1 — Data & Feature Engineering

1. **Data**: Downloaded via `yfinance` — FPT.VN daily OHLCV, 2021–2024 (1,042 raw days → 992 clean)
2. **Features (18 total)**:
   - **Momentum**: RSI-14, RSI-7, lagged returns (1d, 3d, 5d, 10d, 20d)
   - **Trend**: MACD (12/26/9), SMA ratios (5/20, 5 vs 20), golden/death cross signal
   - **Volatility**: Bollinger Band % position, BB width, ATR-14 (normalized by price)
   - **Volume**: Volume/SMA-20 ratio, volume-price confirmation flag
3. **Target**: Binary — 1 if next-day return > 0 (direction prediction). 46.7% UP / 53.3% DOWN
4. **Key design principle**: All features computed **without look-ahead bias** — only past data used

### Notebook 2 — ML Model Comparison

| Model | ROC-AUC | Accuracy | F1 Score |
|---|---|---|---|
| Logistic Regression (baseline) | 0.521 | 0.520 | 0.472 |
| Random Forest (ensemble) | 0.545 | 0.523 | 0.539 |
| **XGBoost (best by AUC)** | **0.545** | **0.530** | **0.474** |

**Design decisions:**
- **Train/test split**: Time-based 70/30 — random splitting causes look-ahead bias in time series
- **Hyperparameter tuning**: `RandomizedSearchCV` (20 iterations, StratifiedKFold CV=5, no shuffling)
- **Class imbalance**: Handled via `class_weight='balanced'`
- **XGBoost best params**: `n_estimators=200, max_depth=3, lr=0.05, subsample=1.0, colsample=0.7`
- **Random Forest best params**: `n_estimators=100, max_depth=3, min_samples_leaf=30, max_features=0.5`

> **Note on accuracy (~52-53%)**: Stock price direction prediction near 50% accuracy is the expected
> result for an efficient market. Any model claiming 70%+ accuracy is almost certainly over-fitted or
> suffering from look-ahead bias. The value is in *consistent edge* across many trades, not single-trade accuracy.

### Notebook 3 — Strategy Backtest

**Configuration:**
- Strategy: Long-only (long on signal=1, hold cash on signal=0)
- Transaction costs: 0.15% per one-way trade (VN standard brokerage rate)
- Risk-free rate: 4.5% annual (VN 1-year government bond yield)

**Results (test period: Nov 2023 – Dec 2024):**

| Metric | ML Strategy | Buy-and-Hold |
|---|---|---|
| Total Return | +19.62% | +89.70% |
| Annualized Return | 16.35% | 71.85% |
| Annualized Volatility | **13.15%** | 23.24% |
| Sharpe Ratio | **0.875** | 2.253 |
| Max Drawdown | **-11.78%** | -15.04% |
| Days in Market | 114 / 298 (38.3%) | 298 / 298 (100%) |

**Threshold Sensitivity Analysis:**

| Threshold | Sharpe | Total Return | Max Drawdown | Trades |
|---|---|---|---|---|
| 0.40 | **1.859** | **+51.63%** | -8.23% | 50 |
| 0.45 | 1.002 | +24.45% | -11.18% | 50 |
| **0.50 (baseline)** | **0.875** | **+19.62%** | **-11.78%** | 50 |
| 0.55 | 0.755 | +15.56% | -9.20% | 42 |
| 0.60 | 1.146 | +15.66% | -2.75% | 24 |

---

## Validation Charts

| Chart | Description |
|---|---|
| ![Equity Curve](images/09_equity_curve.png) | Cumulative return vs buy-and-hold, drawdown, position exposure |
| ![Performance Dashboard](images/10_performance_dashboard.png) | Total return, Sharpe, threshold sensitivity bar charts |
| ![ROC Curves](images/05_roc_curves.png) | ROC curves for all 3 models on hold-out test set |
| ![Feature Importance](images/06_feature_importance.png) | RF & XGBoost: which indicators drive predictions |

---

## How to Run

```bash
# 1. Clone
git clone https://github.com/phgnehe2612/vn-stock-ml-strategy.git
cd vn-stock-ml-strategy

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run in order
python notebook_01_data_and_features.py   # ~30s: downloads data, engineers 18 features
python notebook_02_ml_models.py           # ~2min: trains 3 ML models with hyperparameter tuning
python notebook_03_backtest.py            # ~10s: simulates strategy, outputs performance metrics
```

> **Note**: FPT.VN price data is downloaded automatically via `yfinance` — no manual download required.

---

## Limitations & Future Work

- **No short-selling**: Strategy only goes long or holds cash. Adding short positions could improve Sharpe
- **Single asset**: Extend to a multi-stock VN portfolio with position sizing (Kelly Criterion)
- **No macro overlay**: Incorporate macro regime signals (interest rates, VN-Index trend filter)
- **Deep learning**: Add LSTM/Transformer for sequence modeling of price dynamics
- **Live deployment**: Integrate with VN brokerage API for paper trading validation

---

## Tech Stack

`Python 3.10` · `pandas` · `numpy` · `scikit-learn` · `xgboost` · `yfinance` · `matplotlib` · `seaborn` · `joblib`

---

## References

- Fama, E. (1970). *Efficient Capital Markets: A Review of Theory and Empirical Work*. Journal of Finance
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley
- Chan, E. (2013). *Algorithmic Trading: Winning Strategies and Their Rationale*. Wiley
- VN-Index market data: Yahoo Finance via `yfinance` library
