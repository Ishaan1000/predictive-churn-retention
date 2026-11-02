"""
models.py
=========
Trains and compares three classifiers on the SQL feature table.

  * Logistic Regression  (scaled numerics + one-hot categoricals, L2)
  * Random Forest
  * Gradient Boosting    (sklearn GradientBoostingClassifier)

Protocol: stratified 80/20 hold-out split. 5-fold CV on the training set
picks between the models; the hold-out is touched once, for the final
numbers. Every customer also gets an out-of-fold churn probability
(5-fold cross_val_predict on the full base), so tiering never scores a
customer with a model that saw their label.

Author: Ishaan Mankotia
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

SEED = 42
TARGET = "churned_next_month"
CATEGORICAL = ["region", "plan", "contract_type", "payment_method"]
DROP = ["customer_id", TARGET]


def split_xy(df):
    X = df.drop(columns=DROP)
    for c in ["paperless_billing", "has_streaming_addon", "has_device_emi"]:
        X[c] = X[c].astype(int)
    return X, df[TARGET].values


def numeric_cols(X):
    return [c for c in X.columns if c not in CATEGORICAL]


def build_models(X):
    num = numeric_cols(X)
    linear_prep = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL),
    ])
    # trees: ordinal-encode categoricals so SHAP gives one value per original feature
    tree_prep = ColumnTransformer([
        ("num", "passthrough", num),
        ("cat", OrdinalEncoder(), CATEGORICAL),
    ], verbose_feature_names_out=False)
    return {
        "Logistic Regression": Pipeline([("prep", linear_prep),
                                         ("clf", LogisticRegression(C=1.0, max_iter=2000))]),
        "Random Forest": Pipeline([("prep", tree_prep),
                                   ("clf", RandomForestClassifier(n_estimators=400, min_samples_leaf=5,
                                                                  max_features="sqrt", n_jobs=-1,
                                                                  random_state=SEED))]),
        "Gradient Boosting": Pipeline([("prep", tree_prep),
                                       ("clf", GradientBoostingClassifier(n_estimators=300, learning_rate=0.05,
                                                                          max_depth=3, subsample=0.8,
                                                                          random_state=SEED))]),
    }


def best_threshold(y, p):
    """Threshold that maximises F1 (chosen on CV predictions, not the hold-out)."""
    grid = np.linspace(0.1, 0.9, 81)
    return float(grid[np.argmax([f1_score(y, p >= t) for t in grid])])


def train_and_compare(df):
    X, y = split_xy(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    rows, fitted, rocs = [], {}, {}
    for name, model in build_models(X).items():
        cv_auc = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=-1)
        p_cv = cross_val_predict(model, X_tr, y_tr, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        thr = best_threshold(y_tr, p_cv)
        model.fit(X_tr, y_tr)
        p = model.predict_proba(X_te)[:, 1]
        yhat = p >= thr
        rows.append({
            "model": name,
            "cv_roc_auc_mean": cv_auc.mean(), "cv_roc_auc_std": cv_auc.std(),
            "test_roc_auc": roc_auc_score(y_te, p),
            "test_pr_auc": average_precision_score(y_te, p),
            "threshold": thr,
            "precision": precision_score(y_te, yhat), "recall": recall_score(y_te, yhat),
            "f1": f1_score(y_te, yhat), "brier": brier_score_loss(y_te, p),
        })
        fitted[name] = model
        rocs[name] = roc_curve(y_te, p)
        print(f"  {name:<20} CV AUC {cv_auc.mean():.3f} +/- {cv_auc.std():.3f} | test AUC {rows[-1]['test_roc_auc']:.3f}")

    results = pd.DataFrame(rows).round(4)
    best = results.sort_values("cv_roc_auc_mean", ascending=False).iloc[0]["model"]
    return {"results": results, "best": best, "models": fitted, "rocs": rocs,
            "X_train": X_tr, "X_test": X_te, "y_train": y_tr, "y_test": y_te, "X": X, "y": y}


def out_of_fold_scores(df, model_name):
    """5-fold OOF churn probability for every customer, from the chosen model spec."""
    X, y = split_xy(df)
    model = build_models(X)[model_name]
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED + 1)
    return cross_val_predict(model, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
