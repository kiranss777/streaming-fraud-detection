"""Model features computed from live swipes - shared by the Spark stream (job.py) and the
train/serve parity test (tests/test_feature_parity.py).

This must reproduce glue/etl_features.py exactly: the model learned from Glue's features,
so any difference here silently changes what it sees in production ("train/serve skew").
"""
import json
import math
import pickle
from collections import deque
from functools import lru_cache

import numpy as np
import pandas as pd

HISTORY_FEATURES = [
    "secs_since_last_txn", "txn_count_1h", "txn_count_24h", "amt_sum_24h",
    "card_txn_count", "card_avg_amt", "card_std_amt", "amt_to_card_avg", "amt_zscore",
]
HOUR, DAY = 3_600, 86_400


def epoch_seconds(ts):
    """Timestamps -> whole epoch seconds, whatever unit pandas stored them in
    (pandas 2 parses to nanoseconds, pandas 3 to seconds)."""
    return pd.to_datetime(ts).astype("datetime64[s]").astype("int64")


# ---------- Row features: from the swipe itself ----------

def spark_age(ts, dob):
    """floor(months_between(ts, dob) / 12), following Spark's months_between rules
    (same day-of-month, or both month-ends -> whole months; else a 31-day-month fraction)."""
    months = (ts.dt.year - dob.dt.year) * 12 + (ts.dt.month - dob.dt.month)
    whole = (ts.dt.day == dob.dt.day) | (ts.dt.is_month_end & dob.dt.is_month_end)
    secs = (ts.dt.day - dob.dt.day) * DAY + ts.dt.hour * HOUR + ts.dt.minute * 60 + ts.dt.second
    between = np.round(months + np.where(whole, 0.0, secs / (31 * DAY)), 8)
    return np.floor(between / 12).astype("int64")


def haversine_km(lat1, lon1, lat2, lon2):
    dlat, dlon = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def row_features(df):
    ts, dob = df["trans_ts"], pd.to_datetime(df["dob"], format="%Y-%m-%d")
    return pd.DataFrame({
        "merchant": df["merchant"].str.replace(r"^fraud_", "", regex=True),  # generator artifact
        "age": spark_age(ts, dob),
        "hour": ts.dt.hour.astype("int64"),
        "day_of_week": ((ts.dt.dayofweek + 1) % 7 + 1).astype("int64"),  # Spark: 1 = Sunday
        "distance_km": haversine_km(df["lat"], df["long"], df["merch_lat"], df["merch_long"]),
    }, index=df.index)


# ---------- Card-history features: from that card's EARLIER swipes only ----------

class CardHistory:
    """One card's running state: totals over all earlier swipes + (time, amount) of the last 24h.
    count/sum/sum-of-squares give mean and std without keeping the whole history."""

    def __init__(self, count=0, amt_sum=0.0, amt_sumsq=0.0, last_ts=None, recent=()):
        self.count, self.amt_sum, self.amt_sumsq, self.last_ts = count, amt_sum, amt_sumsq, last_ts
        self.recent = deque(recent)

    def copy(self):
        return CardHistory(self.count, self.amt_sum, self.amt_sumsq, self.last_ts, self.recent)

    def to_list(self):
        return [self.count, self.amt_sum, self.amt_sumsq, self.last_ts, list(self.recent)]

    @classmethod
    def from_list(cls, v):
        count, amt_sum, amt_sumsq, last_ts, recent = v
        return cls(count, amt_sum, amt_sumsq, last_ts, (tuple(r) for r in recent))

    def next(self, ts, amt):
        """Features for a swipe at `ts` (epoch seconds), then record it.
        Windows match Glue's rangeBetween(-3600 / -86400, -1): earlier seconds only."""
        recent = self.recent
        while recent and recent[0][0] < ts - DAY:
            recent.popleft()
        n_1h = n_24h = 0
        sum_24h = 0.0
        for t, a in recent:
            if t <= ts - 1:
                n_24h += 1
                sum_24h += a
                if t >= ts - HOUR:
                    n_1h += 1

        n = self.count
        avg = self.amt_sum / n if n else math.nan
        std = math.sqrt(max(0.0, (self.amt_sumsq - self.amt_sum ** 2 / n) / (n - 1))) if n >= 2 else math.nan
        features = {
            "secs_since_last_txn": ts - self.last_ts if self.last_ts is not None else math.nan,
            "txn_count_1h": n_1h,
            "txn_count_24h": n_24h,
            "amt_sum_24h": sum_24h,
            "card_txn_count": n,
            "card_avg_amt": avg,
            "card_std_amt": std,
            "amt_to_card_avg": amt / avg if n else math.nan,
            "amt_zscore": (amt - avg) / std if std > 0 else math.nan,  # Spark: x / 0 -> null
        }

        self.count += 1
        self.amt_sum += amt
        self.amt_sumsq += amt * amt
        self.last_ts = ts
        recent.append((ts, amt))
        return features


def compute(swipes, histories, seed=None):
    """All model features for a batch of swipes from any number of cards.

    Swipes are processed in (time, trans_num) order - Glue's order. `histories` maps
    cc_num -> CardHistory and is updated in place; a card seen for the first time starts
    from its `seed` entry (copied, so the seed itself is never changed)."""
    df = swipes.assign(trans_ts=pd.to_datetime(swipes["trans_date_trans_time"], format="%Y-%m-%d %H:%M:%S"))
    df = df.assign(ts_sec=epoch_seconds(df["trans_ts"])).sort_values(["ts_sec", "trans_num"], kind="stable")
    rows = row_features(df)
    seed = seed or {}
    hist = []
    for cc, t, a in zip(df["cc_num"].tolist(), df["ts_sec"].tolist(), df["amt"].tolist()):
        h = histories.get(cc)
        if h is None:
            h = histories[cc] = seed[cc].copy() if cc in seed else CardHistory()
        hist.append(h.next(t, a))
    hist = pd.DataFrame(hist, index=df.index, columns=HISTORY_FEATURES)
    return pd.concat([df.drop(columns=["merchant"]), rows, hist], axis=1)


def dump_histories(histories):
    return json.dumps({cc: h.to_list() for cc, h in histories.items()})


def load_histories(text):
    return {cc: CardHistory.from_list(v) for cc, v in json.loads(text).items()}


# ---------- Starting point: each card's history at the end of the training period ----------

def build_seed(card_state, train_tail):
    """card_state: Glue's per-card totals at the end of train (processed/card_state).
    train_tail: cc_num, trans_ts, amt of training swipes from the last 24h of the period."""
    last_ts = epoch_seconds(card_state["last_txn_ts"])
    seed = {
        r.cc_num: CardHistory(int(r.txn_count), float(r.amt_sum), float(r.amt_sumsq), int(t))
        for r, t in zip(card_state.itertuples(index=False), last_ts)
    }
    tail = train_tail.assign(ts_sec=epoch_seconds(train_tail["trans_ts"]))
    for cc, t, a in tail.sort_values("ts_sec")[["cc_num", "ts_sec", "amt"]].itertuples(index=False):
        seed[cc].recent.append((t, float(a)))
    return seed


@lru_cache(maxsize=1)
def load_seed(path):
    with open(path, "rb") as f:
        return pickle.load(f)
