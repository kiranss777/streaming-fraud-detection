"""Scores swipes with the trained XGBoost model; explains flagged ones with SHAP.

The model folder is what SageMaker training saved (model.tar.gz, unpacked):
  xgboost-model.json  the trees
  metadata.json       feature order, category lists, cutoff, best iteration
"""
import json
import math
import time
from functools import lru_cache

import numpy as np
import pandas as pd
import xgboost as xgb

CATEGORY_NAMES = {
    "shopping_net": "online shopping", "shopping_pos": "in-store shopping",
    "misc_net": "online misc", "misc_pos": "in-store misc", "grocery_net": "online grocery",
    "grocery_pos": "in-store grocery", "gas_transport": "gas/transport", "food_dining": "dining",
    "entertainment": "entertainment", "health_fitness": "health/fitness", "home": "home",
    "kids_pets": "kids/pets", "personal_care": "personal care", "travel": "travel",
}
DAYS = {1: "Sunday", 2: "Monday", 3: "Tuesday", 4: "Wednesday", 5: "Thursday", 6: "Friday", 7: "Saturday"}


def clock(hour):
    h = int(hour)
    label = f"{h % 12 or 12} {'AM' if h < 12 else 'PM'}"
    return f"late night ({label})" if h >= 22 or h < 5 else label


def reason(feature, v):
    """One plain-English phrase for a feature that pushed the score toward fraud."""
    missing = v is None or (isinstance(v, float) and math.isnan(v))
    phrases = {
        "amt": lambda: f"${v:,.0f} purchase",
        "amt_sum_24h": lambda: f"${v:,.0f} spent in last 24h",
        "hour": lambda: clock(v),
        "category": lambda: f"{CATEGORY_NAMES.get(v, v)} purchase",
        "amt_to_card_avg": lambda: f"{v:.1f}x card's usual amount",
        "amt_zscore": lambda: f"amount {v:+.1f} std devs from card's norm",
        "txn_count_1h": lambda: f"{int(v)} card txns in past hour",
        "txn_count_24h": lambda: f"{int(v)} card txns in past 24h",
        "secs_since_last_txn": lambda: f"{v / 60:,.0f} min since card's last txn",
        "card_txn_count": lambda: "brand-new card" if v == 0 else f"card has {int(v):,} past txns",
        "card_avg_amt": lambda: f"card usually spends ${v:,.0f}",
        "card_std_amt": lambda: f"card's spend varies by ${v:,.0f}",
        "age": lambda: f"customer age {int(v)}",
        "city_pop": lambda: f"city population {int(v):,}",
        "distance_km": lambda: f"{v:,.0f} km from home",
        "gender": lambda: f"gender {v}",
        "day_of_week": lambda: DAYS.get(int(v), str(v)),
    }
    if missing:  # only card-history features can be empty: the card has (almost) no past swipes
        return "first-ever txn on card" if feature == "secs_since_last_txn" else "card has no spending history"
    return phrases.get(feature, lambda: f"{feature} = {v}")()


def shown(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "no history"
    if isinstance(v, (float, np.floating)):
        return f"{v:,.2f}"
    return str(v)


class Scorer:
    def __init__(self, model_dir):
        meta = json.load(open(f"{model_dir}/metadata.json"))
        self.features = meta["features"]
        self.categorical = meta["categorical"]
        self.categories = meta["categories"]
        self.threshold = meta["threshold"]
        self.iters = (0, meta["best_iteration"] + 1)
        self.booster = xgb.Booster()
        self.booster.load_model(f"{model_dir}/xgboost-model.json")

    def frame(self, df):
        X = df[self.features].copy()
        for c in self.categorical:  # same category list (and codes) as training
            X[c] = pd.Categorical(X[c], categories=self.categories[c])
        return X

    def score(self, df):
        """df + score, flagged; flagged rows also get reasons, shap_json, values_json.
        Sets self.timings (ms) for the stream's batch log."""
        t0 = time.perf_counter()
        # inplace_predict skips building a DMatrix: same scores, less overhead.
        score = self.booster.inplace_predict(self.frame(df), iteration_range=self.iters)
        t1 = time.perf_counter()
        flagged = score >= self.threshold
        out = df.assign(score=score.astype("float32"), flagged=flagged,
                        reasons=None, shap_json=None, values_json=None)
        if flagged.any():
            sub = df[flagged]
            # Exact TreeSHAP from XGBoost itself; last column is the baseline - dropped.
            contribs = self.booster.predict(xgb.DMatrix(self.frame(sub), enable_categorical=True),
                                            pred_contribs=True, iteration_range=self.iters)[:, :-1]
            reasons, shaps, values = [], [], []
            for vals, c in zip(sub[self.features].to_dict("records"), contribs):
                top = [j for j in np.argsort(-c)[:3] if c[j] > 0]
                reasons.append("; ".join(reason(self.features[j], vals[self.features[j]]) for j in top))
                shaps.append(json.dumps({f: round(float(s), 4) for f, s in zip(self.features, c)}))
                values.append(json.dumps({f: shown(vals[f]) for f in self.features}))
            out.loc[flagged, "reasons"] = reasons
            out.loc[flagged, "shap_json"] = shaps
            out.loc[flagged, "values_json"] = values
        self.timings = {"predict": 1000 * (t1 - t0), "shap": 1000 * (time.perf_counter() - t1)}
        return out


@lru_cache(maxsize=1)
def load_scorer(model_dir):
    """One Scorer per Python worker process, loaded on first use."""
    return Scorer(model_dir)
