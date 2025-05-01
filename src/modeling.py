#!/usr/bin/env python3
# src/modeling.py

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


def load_data(x_path, y_path, xtest_path):
    """Load processed datasets from CSVs."""
    X_train = pd.read_csv(x_path, index_col=0)
    y_df    = pd.read_csv(y_path, index_col=0)
    y_train = y_df.iloc[:, 0]
    X_test  = pd.read_csv(xtest_path, index_col=0)
    return X_train, y_train, X_test

def scale_if_needed(model_name, X_train, X_test):
    """Return scaled or unscaled copies based on model type."""
    must_scale = {"Ridge", "Lasso", "SVR"}
    if model_name in must_scale:
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns, index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns, index=X_test.index
        )
        return X_train_scaled, X_test_scaled
    return X_train, X_test

def evaluate_baseline(models, X_train, y_train, cv):
    """Compute baseline CV RMSE for each model with default params."""
    results = []
    for name, model in models.items():
        X_tr, _ = scale_if_needed(name, X_train, X_train)
        scores = cross_val_score(
            model,
            X_tr,
            y_train,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1
        )
        print(f"[DEBUG] {name} fold scores (neg RMSE): {scores}")
        rmse_scores = -scores
        results.append({
            "model": name,
            "mean_rmse": rmse_scores.mean(),
            "std_rmse": rmse_scores.std()
        })
    return pd.DataFrame(results)

def tune_models(models, param_grids, X_train, y_train, cv, n_iter=20):
    """Perform RandomizedSearchCV for each model, return tuned RMSE and best params."""
    tuned = []
    for name, model in models.items():
        if name not in param_grids:
            continue
        X_tr, _ = scale_if_needed(name, X_train, X_train)
        search = RandomizedSearchCV(
            model,
            param_grids[name],
            n_iter=n_iter,
            scoring="neg_root_mean_squared_error",
            cv=cv,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        search.fit(X_tr, y_train)
        best_neg_rmse = search.best_score_
        std_neg_rmse  = search.cv_results_["std_test_score"][search.best_index_]
        tuned.append({
            "model": name,
            "mean_rmse": -best_neg_rmse,
            "std_rmse": std_neg_rmse,
            "best_params": search.best_params_
        })
    return pd.DataFrame(tuned)

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    X_train, y_train, X_test = load_data(
        args.X_train, args.y_train, args.X_test
    )

    print("▶ X_train shape:", X_train.shape)
    print("▶ y_train shape:", y_train.shape)
    print("▶ Any NaNs in X_train? ", X_train.isnull().any().any())
    print("▶ Any NaNs in y_train? ", y_train.isnull().any())
    print()

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "Ridge": Ridge(random_state=42),
        "Lasso": Lasso(random_state=42),
        "RandomForest": RandomForestRegressor(random_state=42),
        "SVR": SVR(),
        "XGBoost": XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            use_label_encoder=False,
            eval_metric="rmse"
        ),
        "LightGBM": LGBMRegressor(random_state=42),
        "CatBoost": CatBoostRegressor(random_state=42, verbose=0)
    }

    print("=== Baseline CV RMSE ===")
    baseline_df = evaluate_baseline(models, X_train, y_train, cv)
    print(baseline_df.to_string(index=False))
    baseline_df.to_csv(os.path.join(args.out_dir, "baseline_rmse.csv"), index=False)

    param_grids = {
        "Ridge":    {"alpha": np.logspace(-3, 3, 30)},
        "Lasso":    {"alpha": np.logspace(-4, 1, 30)},
        "RandomForest": {
            "n_estimators": [100, 200, 500],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 5]
        },
        "SVR": {
            "kernel": ["rbf", "linear"],
            "C": [0.1, 1, 10, 100],
            "epsilon": [0.01, 0.1, 1]
        },

        "XGBoost": {
            "n_estimators": [100, 300, 500],
            "max_depth": [3, 6, 9],
            "learning_rate": [0.01, 0.1, 0.2],
            "colsample_bytree": [0.5, 0.8, 1.0]
        },
        "LightGBM": {
            "n_estimators": [100, 300, 500],
            "num_leaves": [31, 50, 100],
            "learning_rate": [0.01, 0.1, 0.2]
        },
        "CatBoost": {
            "iterations": [200, 500],
            "depth": [4, 6, 8],
            "learning_rate": [0.01, 0.1]
        }
    }

    print("\n=== Hyperparameter Tuning (RandomizedSearchCV) ===")
    tuned_df = tune_models(models, param_grids, X_train, y_train, cv, n_iter=20)
    print(tuned_df.to_string(index=False))
    tuned_df.to_csv(os.path.join(args.out_dir, "tuned_rmse.csv"), index=False)

    print(f"\nAll done — results in `{args.out_dir}`.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train & evaluate regressors on House Prices (CV RMSE)"
    )
    parser.add_argument(
        "--X_train", required=True,
        help="Path to processed X_train CSV (header + index_col=0)"
    )
    parser.add_argument(
        "--y_train", required=True,
        help="Path to processed y_train CSV (single column + index_col=0)"
    )
    parser.add_argument(
        "--X_test", required=True,
        help="Path to processed X_test CSV (header + index_col=0)"
    )
    parser.add_argument(
        "--out_dir", default="results",
        help="Directory to save baseline_rmse.csv and tuned_rmse.csv"
    )
    args = parser.parse_args()
    main(args)
