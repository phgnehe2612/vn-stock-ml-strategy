# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg')

# %% [markdown]
# # Notebook 2: ML Model Training, Hyperparameter Tuning & Evaluation
#
# **Task**: Binary classification -- predict next-day price direction (UP=1, DOWN=0)
# **Models trained**:
#   1. Logistic Regression (interpretable baseline)
#   2. Random Forest (ensemble, feature importance)
#   3. XGBoost (gradient boosting, hyperparameter tuning via RandomizedSearchCV)
#
# **Critical design choice**: Time-based train/test split (70/30) is used instead
# of random split. Random splitting causes look-ahead bias in time series -- a model
# trained on future data will appear to work but fail in live trading.
#
# **Evaluation metrics**: Accuracy, F1, ROC-AUC (all computed on held-out test set only)

# %% 1. Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics         import (accuracy_score, f1_score, roc_auc_score,
                                     roc_curve, classification_report,
                                     confusion_matrix, ConfusionMatrixDisplay)
import xgboost as xgb
import joblib, os
os.makedirs('images', exist_ok=True)
os.makedirs('models', exist_ok=True)

# %% 2. Load Features
df = pd.read_csv('data/features.csv', index_col=0, parse_dates=True)
print(f"[OK] Loaded feature matrix: {df.shape}")

FEATURE_COLS = [
    'ret_1d', 'ret_3d', 'ret_5d', 'ret_10d', 'ret_20d',
    'price_sma5_ratio', 'price_sma20_ratio', 'sma5_sma20_ratio',
    'rsi_14', 'rsi_7',
    'macd_norm', 'macd_hist', 'macd_cross',
    'bb_pct', 'bb_width',
    'atr_14_norm',
    'vol_ratio', 'vol_price_up'
]

X = df[FEATURE_COLS].values
y = df['target'].values
dates = df.index

# %% 3. Time-Based Train / Test Split (CRITICAL for time series)
#
# Why NOT random split: if we randomly select dates, the model will train on
# future data and "see" test dates during training, inflating performance metrics.
# Time-based split preserves temporal ordering and reflects real trading conditions.

TRAIN_RATIO = 0.70
split_idx   = int(len(df) * TRAIN_RATIO)

X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
dates_test      = dates[split_idx:]

print(f"\nTrain set: {split_idx} days ({dates[0].date()} to {dates[split_idx-1].date()})")
print(f"Test set:  {len(df) - split_idx} days ({dates[split_idx].date()} to {dates[-1].date()})")
print(f"Train UP rate: {y_train.mean():.1%}   |   Test UP rate: {y_test.mean():.1%}")

# %% 4. Feature Scaling (required for Logistic Regression)
scaler      = StandardScaler()
X_train_sc  = scaler.fit_transform(X_train)
X_test_sc   = scaler.transform(X_test)
joblib.dump(scaler, 'models/scaler.pkl')

# %% 5. Model 1 -- Logistic Regression (Baseline)
print("\n[1/3] Training Logistic Regression...")
lr = LogisticRegression(C=0.05, max_iter=1000, random_state=42, class_weight='balanced')
lr.fit(X_train_sc, y_train)

lr_pred  = lr.predict(X_test_sc)
lr_prob  = lr.predict_proba(X_test_sc)[:, 1]
lr_acc   = accuracy_score(y_test, lr_pred)
lr_f1    = f1_score(y_test, lr_pred)
lr_auc   = roc_auc_score(y_test, lr_prob)
print(f"   Accuracy={lr_acc:.3f}  F1={lr_f1:.3f}  AUC={lr_auc:.3f}")

# %% 6. Model 2 -- Random Forest (Ensemble)
print("\n[2/3] Training Random Forest with hyperparameter tuning...")

rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth':    [3, 4, 5, 6],
    'min_samples_leaf': [20, 30, 50],
    'max_features': ['sqrt', 0.5],
}

# Time-series CV: walk-forward (no shuffling)
tscv = StratifiedKFold(n_splits=5, shuffle=False)

rf_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, class_weight='balanced'),
    rf_param_grid, n_iter=20, cv=tscv,
    scoring='roc_auc', n_jobs=-1, random_state=42
)
rf_search.fit(X_train, y_train)
rf = rf_search.best_estimator_

rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:, 1]
rf_acc  = accuracy_score(y_test, rf_pred)
rf_f1   = f1_score(y_test, rf_pred)
rf_auc  = roc_auc_score(y_test, rf_prob)
print(f"   Best params: {rf_search.best_params_}")
print(f"   Accuracy={rf_acc:.3f}  F1={rf_f1:.3f}  AUC={rf_auc:.3f}")
joblib.dump(rf, 'models/random_forest.pkl')

# %% 7. Model 3 -- XGBoost (Gradient Boosting)
print("\n[3/3] Training XGBoost with hyperparameter tuning...")

xgb_param_grid = {
    'n_estimators':   [100, 200, 300],
    'max_depth':      [3, 4, 5],
    'learning_rate':  [0.01, 0.05, 0.1],
    'subsample':      [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'min_child_weight': [10, 20, 30],
}

xgb_search = RandomizedSearchCV(
    xgb.XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0),
    xgb_param_grid, n_iter=20, cv=tscv,
    scoring='roc_auc', n_jobs=-1, random_state=42
)
xgb_search.fit(X_train, y_train)
xgb_model = xgb_search.best_estimator_

xgb_pred = xgb_model.predict(X_test)
xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
xgb_acc  = accuracy_score(y_test, xgb_pred)
xgb_f1   = f1_score(y_test, xgb_pred)
xgb_auc  = roc_auc_score(y_test, xgb_prob)
print(f"   Best params: {xgb_search.best_params_}")
print(f"   Accuracy={xgb_acc:.3f}  F1={xgb_f1:.3f}  AUC={xgb_auc:.3f}")
joblib.dump(xgb_model, 'models/xgboost.pkl')

# %% 8. Model Comparison Summary
print("\n" + "="*60)
print("   MODEL COMPARISON (Test Set)")
print("="*60)
print(f"   {'Model':<25} {'Accuracy':>9} {'F1':>9} {'ROC-AUC':>9}")
print("-"*60)
results = [
    ('Logistic Regression', lr_acc, lr_f1, lr_auc, lr_prob),
    ('Random Forest',        rf_acc, rf_f1, rf_auc, rf_prob),
    ('XGBoost',              xgb_acc, xgb_f1, xgb_auc, xgb_prob),
]
for name, acc, f1, auc, _ in results:
    marker = " <-- BEST" if auc == max(lr_auc, rf_auc, xgb_auc) else ""
    print(f"   {name:<25} {acc:>9.3f} {f1:>9.3f} {auc:>9.3f}{marker}")
print("="*60)

# Pick best model by AUC
best_name, best_acc, best_f1, best_auc, best_prob = max(results, key=lambda x: x[3])
print(f"\nBest model: {best_name} (AUC={best_auc:.3f})")

# Save best predictions for Notebook 3
test_df = pd.DataFrame({
    'date':          dates_test,
    'close':         df['Close'].values[split_idx:],
    'actual_return': df['return_next_1d'].values[split_idx:],
    'actual_target': y_test,
    'lr_signal':     lr_pred,
    'rf_signal':     rf_pred,
    'xgb_signal':    xgb_pred,
    'best_prob':     best_prob,
    'best_signal':   (best_prob > 0.50).astype(int),
}).set_index('date')
test_df.to_csv('data/test_predictions.csv')
print("[OK] Predictions saved: data/test_predictions.csv")

# %% 9. Visualization

# ---- 9.1 ROC Curves ----
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, (name, acc, f1, auc, prob) in zip(axes, results):
    fpr, tpr, _ = roc_curve(y_test, prob)
    ax.plot(fpr, tpr, color='#3498db', linewidth=2.5, label=f'AUC = {auc:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax.fill_between(fpr, tpr, alpha=0.1, color='#3498db')
    ax.set_title(f'ROC Curve\n{name}', fontweight='bold', fontsize=11)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(fontsize=11)
    ax.annotate(f'Acc={acc:.3f}\nF1={f1:.3f}', xy=(0.55, 0.15), fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.suptitle('Model Comparison: ROC Curves on Hold-Out Test Set', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('images/05_roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart saved: images/05_roc_curves.png")

# ---- 9.2 Feature Importance (Random Forest & XGBoost) ----
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

for ax, model, title in [(axes[0], rf, 'Random Forest'), (axes[1], xgb_model, 'XGBoost')]:
    fi = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True).tail(12)
    colors = ['#e74c3c' if v > fi.median() else '#3498db' for v in fi.values]
    ax.barh(fi.index, fi.values, color=colors, edgecolor='white', height=0.7)
    ax.set_title(f'Feature Importance\n{title}', fontweight='bold', fontsize=12)
    ax.set_xlabel('Importance Score')
    ax.axvline(fi.median(), color='gray', linestyle='--', linewidth=1, alpha=0.6)

plt.suptitle('Which Features Drive Predictions?', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('images/06_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart saved: images/06_feature_importance.png")

# ---- 9.3 Confusion Matrix (Best Model) ----
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, (best_prob > 0.50).astype(int))
disp = ConfusionMatrixDisplay(cm, display_labels=['DOWN (0)', 'UP (1)'])
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title(f'Confusion Matrix -- {best_name}', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig('images/07_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart saved: images/07_confusion_matrix.png")

# ---- 9.4 Prediction probability distribution ----
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(best_prob[y_test == 1], bins=30, alpha=0.6, color='#2ecc71',
        label='True UP days', density=True, edgecolor='white')
ax.hist(best_prob[y_test == 0], bins=30, alpha=0.6, color='#e74c3c',
        label='True DOWN days', density=True, edgecolor='white')
ax.axvline(0.50, color='black', linestyle='--', linewidth=1.5, label='Decision threshold = 0.50')
ax.set_xlabel('Predicted Probability (P(UP))', fontsize=11)
ax.set_ylabel('Density')
ax.set_title(f'Prediction Confidence Distribution -- {best_name}', fontweight='bold', fontsize=12)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('images/08_prediction_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart saved: images/08_prediction_distribution.png")

print("\n=== NOTEBOOK 2 COMPLETE ===")
print(f"Key results for README:")
print(f"  Best model      : {best_name}")
print(f"  Accuracy        : {best_acc:.3f}")
print(f"  F1 Score        : {best_f1:.3f}")
print(f"  ROC-AUC         : {best_auc:.3f}")
print(f"  (Logistic)  AUC : {lr_auc:.3f}")
print(f"  (RandomForest) AUC: {rf_auc:.3f}")
print(f"  (XGBoost)   AUC : {xgb_auc:.3f}")
