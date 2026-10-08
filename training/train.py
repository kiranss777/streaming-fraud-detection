"""XGBoost fraud model - runs INSIDE the SageMaker training container.

SageMaker mounts the S3 inputs as local folders and tells us where via env vars:
  SM_CHANNEL_TRAIN / SM_CHANNEL_TEST   processed Parquet from the Glue job
  SM_MODEL_DIR                         whatever we save here -> models/.../model.tar.gz
  SM_OUTPUT_DATA_DIR                   reports saved here    -> models/.../output.tar.gz
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no screen in the container
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

FEATURES = [
    "category", "amt", "gender", "city_pop", "age", "hour", "day_of_week", "distance_km",
    # point-in-time card history (only earlier transactions)
    "secs_since_last_txn", "txn_count_1h", "txn_count_24h", "amt_sum_24h",
    "card_txn_count", "card_avg_amt", "card_std_amt", "amt_to_card_avg", "amt_zscore",
]
# Kept as single categorical features (not one-hot) so SHAP gives one readable value per feature.
CATEGORICAL = ["category", "gender"]
LABEL = "is_fraud"


def load(channel):
    return pd.read_parquet(os.environ[f"SM_CHANNEL_{channel.upper()}"])


def features(df, categories):
    X = df[FEATURES].copy()
    for c in CATEGORICAL:
        X[c] = pd.Categorical(X[c], categories=categories[c])
    return X


def evaluate(y, score, amt, threshold):
    pred = score >= threshold
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "threshold": round(float(threshold), 4),
        "roc_auc": round(roc_auc_score(y, score), 4),
        "pr_auc": round(average_precision_score(y, score), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(2 * precision * recall / (precision + recall) if precision + recall else 0.0, 4),
        "accuracy": round((tp + tn) / len(y), 4),
        "fraud_dollars_caught_pct": round(amt[pred & (y == 1)].sum() / amt[y == 1].sum(), 4),
        "confusion": {"caught": tp, "false_alarms": fp, "missed": fn, "correctly_ignored": tn},
    }


def main():
    p = argparse.ArgumentParser()  # SageMaker passes hyperparameters as CLI args
    p.add_argument("--max_depth", type=int, default=6)
    p.add_argument("--eta", type=float, default=0.1)
    p.add_argument("--num_round", type=int, default=1000)
    p.add_argument("--early_stopping_rounds", type=int, default=50)
    args = p.parse_args()

    model_dir = Path(os.environ["SM_MODEL_DIR"])
    out_dir = Path(os.environ["SM_OUTPUT_DATA_DIR"])

    train_all = load("train").sort_values("trans_ts")
    test = load("test")

    # Time-based split: validate on the most recent 20% of train, like predicting the future.
    cut = int(len(train_all) * 0.8)
    fit, valid = train_all.iloc[:cut], train_all.iloc[cut:]

    categories = {c: sorted(train_all[c].dropna().unique().tolist()) for c in CATEGORICAL}
    X_fit, X_valid, X_test = (features(d, categories) for d in (fit, valid, test))
    d_fit = xgb.DMatrix(X_fit, label=fit[LABEL], enable_categorical=True)
    d_valid = xgb.DMatrix(X_valid, label=valid[LABEL], enable_categorical=True)
    d_test = xgb.DMatrix(X_test, label=test[LABEL], enable_categorical=True)

    # Missing a fraud should cost as much as all the legit rows it hides among.
    neg, pos = (fit[LABEL] == 0).sum(), (fit[LABEL] == 1).sum()
    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "max_depth": args.max_depth,
        "eta": args.eta,
        "scale_pos_weight": neg / pos,
    }
    print(f"fit={len(fit):,} valid={len(valid):,} test={len(test):,} scale_pos_weight={neg / pos:.1f}")

    booster = xgb.train(
        params, d_fit, num_boost_round=args.num_round,
        evals=[(d_fit, "fit"), (d_valid, "valid")],
        early_stopping_rounds=args.early_stopping_rounds, verbose_eval=50,
    )
    iters = (0, booster.best_iteration + 1)

    # Pick the cutoff on validation (max F1), then judge it on test - never tune on test.
    valid_score = booster.predict(d_valid, iteration_range=iters)
    prec, rec, thr = precision_recall_curve(valid[LABEL], valid_score)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-9, None)
    threshold = float(thr[f1.argmax()])

    y, amt = test[LABEL].to_numpy(), test["amt"].to_numpy()
    test_score = booster.predict(d_test, iteration_range=iters)
    report = {
        "rows": {"fit": len(fit), "valid": len(valid), "test": len(test)},
        "best_iteration": booster.best_iteration,
        "test": evaluate(y, test_score, amt, threshold),
        "always_say_legit_accuracy": round(float((y == 0).mean()), 4),
        "threshold_tradeoff": [evaluate(y, test_score, amt, t) for t in (0.1, 0.3, 0.5, 0.7, 0.9, 0.97, 0.99)],
    }

    # ---------- SHAP ----------
    # XGBoost computes exact TreeSHAP itself (pred_contribs); last column is the baseline.
    rng = np.random.default_rng(0)
    frauds = np.flatnonzero(y == 1)
    legit = rng.choice(np.flatnonzero(y == 0), size=20_000, replace=False)
    sample = np.concatenate([frauds, legit])
    d_sample = xgb.DMatrix(X_test.iloc[sample], enable_categorical=True)
    contribs = booster.predict(d_sample, pred_contribs=True, iteration_range=iters)[:, :-1]

    importance = pd.Series(np.abs(contribs).mean(axis=0), index=FEATURES).sort_values(ascending=False)
    report["shap_mean_abs"] = importance.round(4).to_dict()

    # Beeswarm needs numbers to color by, so plot categories by their code.
    X_plot = X_test.iloc[sample].copy()
    for c in CATEGORICAL:
        X_plot[c] = X_plot[c].cat.codes
    shap.summary_plot(contribs, X_plot, show=False, max_display=len(FEATURES))
    plt.title("What drives the fraud score (SHAP, test set)")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary.png", dpi=150)
    plt.close()

    # Why were the highest-scoring caught frauds flagged? Top 3 reasons each.
    caught = [i for i in range(len(frauds)) if test_score[frauds[i]] >= threshold]
    top = sorted(caught, key=lambda i: -test_score[frauds[i]])[:5]
    report["example_explanations"] = [
        {
            "amt": float(amt[frauds[i]]),
            "category": str(test["category"].iloc[frauds[i]]),
            "score": round(float(test_score[frauds[i]]), 4),
            "top_reasons": [
                {"feature": FEATURES[j], "value": str(X_test.iloc[frauds[i]][FEATURES[j]]),
                 "shap": round(float(contribs[i, j]), 3)}
                for j in np.argsort(-contribs[i])[:3]
            ],
        }
        for i in top
    ]

    # ---------- Save ----------
    booster.save_model(model_dir / "xgboost-model.json")
    (model_dir / "metadata.json").write_text(json.dumps({
        "features": FEATURES,
        "categorical": CATEGORICAL,
        "categories": categories,  # streaming must encode categories identically
        "threshold": threshold,
        "best_iteration": booster.best_iteration,
        "xgboost_version": xgb.__version__,
    }, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["test"], indent=2))


if __name__ == "__main__":
    main()
