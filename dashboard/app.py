"""Live fraud dashboard (Streamlit), reading Cassandra and Kafka.

Tab 1 Live Alerts   - what got flagged, why (SHAP), and that card's recent history
Tab 2 Health & Model - throughput, end-to-end latency, Kafka backlog, live model grading

Every Cassandra read hits a bounded set of partitions (a few minutes, one day, one card) -
never a table scan. Refreshes every 5 seconds.
"""
import io
import json
import os
import random
import time
import uuid
from datetime import datetime, timedelta, timezone

import altair as alt
import boto3
import numpy as np
import pandas as pd
import streamlit as st
from cassandra.cluster import Cluster
from cassandra.query import ValueSequence
from confluent_kafka import Consumer, Producer, TopicPartition

DATA_HOST = os.environ["DATA_HOST_IP"]
TOPIC, PARTITIONS = "transactions", 8
LATENCY_EDGES_MS = [100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000, 10000]  # must match stream/job.py
SCORE_BUCKETS = 20
REFRESH = "5s"

LABELS = {
    "category": "Merchant category", "amt": "Amount ($)", "gender": "Gender",
    "city_pop": "City population", "age": "Customer age", "hour": "Hour of day",
    "day_of_week": "Day of week", "distance_km": "Distance to merchant (km)",
    "secs_since_last_txn": "Seconds since card's last txn", "txn_count_1h": "Card txns in last hour",
    "txn_count_24h": "Card txns in last 24h", "amt_sum_24h": "Spent in last 24h ($)",
    "card_txn_count": "Card's past txn count", "card_avg_amt": "Card's usual amount ($)",
    "card_std_amt": "Card's amount spread ($)", "amt_to_card_avg": "Amount vs card's usual (x)",
    "amt_zscore": "Amount z-score vs card",
}

st.set_page_config(page_title="Fraud Detection - Live", page_icon="🛡️", layout="wide")


# ---------- Connections (one per server process) ----------

@st.cache_resource
def cassandra():
    return Cluster([DATA_HOST]).connect("fraud")


@st.cache_resource
def kafka_consumer():  # only used to read partition end offsets (backlog), never consumes
    return Consumer({"bootstrap.servers": f"{DATA_HOST}:9092", "group.id": "dashboard-watermarks"})


@st.cache_resource
def kafka_producer():
    return Producer({"bootstrap.servers": f"{DATA_HOST}:9092"})


@st.cache_resource
def card_profiles():
    """One row per card (home location, dob, ...) for the demo burst - from the raw test file."""
    body = boto3.client("s3").get_object(Bucket=os.environ["DATA_BUCKET"], Key="raw/fraudTest.csv")["Body"].read()
    cols = ["cc_num", "gender", "city", "state", "lat", "long", "city_pop", "dob"]
    return pd.read_csv(io.BytesIO(body), usecols=cols, dtype={"cc_num": str}).drop_duplicates("cc_num")


def rows(query, params=()):
    return pd.DataFrame(list(cassandra().execute(query, params)))


# ---------- Reads (each a bounded partition read) ----------

def recent_alerts(minutes):
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    buckets = [now - timedelta(minutes=i) for i in range(minutes)]
    df = rows("SELECT * FROM alerts_by_minute WHERE minute IN %s", (ValueSequence(buckets),))  # renders (a, b, ...)
    return df.sort_values("scored_at", ascending=False) if not df.empty else df


def recent_metrics(minutes):
    now = datetime.now(timezone.utc)
    since = now - timedelta(minutes=minutes)
    days = sorted({since.date(), now.date()})
    df = pd.concat([rows("SELECT * FROM pipeline_metrics WHERE day = %s AND batch_end > %s", (d, since))
                    for d in days], ignore_index=True)
    if df.empty:
        return df
    df["batch_end"] = pd.to_datetime(df["batch_end"], utc=True)
    return df.sort_values("batch_end")


def card_history(cc_num, n=10):
    return rows("SELECT trans_ts, amt, merchant, category, score, flagged, is_fraud FROM transactions_by_card "
                "WHERE cc_num = %s LIMIT %s", (cc_num, n))


def kafka_backlog(metrics):
    """Messages sent to Kafka but not yet processed = end offset - last processed offset, per partition."""
    if metrics.empty:
        return None
    done = {}
    for offsets in metrics["max_offsets"].dropna():
        for p, o in offsets.items():
            done[p] = max(done.get(p, -1), o)
    total = 0
    for p in range(PARTITIONS):
        _, end = kafka_consumer().get_watermark_offsets(TopicPartition(TOPIC, p), timeout=2)
        total += max(0, end - (done.get(p, -1) + 1))
    return total



# ---------- Helpers ----------

def masked(cc):
    return f"•••• {str(cc)[-4:]}"


def hist_sum(series, length):
    return np.sum([np.asarray(h) for h in series if h is not None] or [np.zeros(length)], axis=0)


def percentile_upper(hist, q):
    """Upper edge of the latency bucket holding the q-th percentile - an honest 'at most'."""
    total = hist.sum()
    if total == 0:
        return None
    i = int(np.searchsorted(np.cumsum(hist), q / 100 * total))
    return LATENCY_EDGES_MS[i] if i < len(LATENCY_EDGES_MS) else float("inf")


def fmt_ms(v):
    return "—" if v is None else ("> 10 s" if v == float("inf") else f"≤ {v:,} ms")


def staleness_badge(metrics, backlog):
    if metrics.empty:
        return "⚪ No batches yet - start the producer"
    age = (datetime.now(timezone.utc) - metrics["batch_end"].max()).total_seconds()
    if age < 15:
        return f"🟢 Live - last batch scored {age:.0f}s ago"
    if backlog:
        return f"🔴 STALLED - {backlog:,} swipes waiting, last batch {age:.0f}s ago"
    return f"⚪ Idle - no new swipes for {age:.0f}s"


# ---------- Demo: inject a fraud burst ----------

def inject_burst(n=5):
    """A stolen-card pattern on a real card: big online/grocery spends, a couple of minutes apart,
    right after the card's latest swipe (so its history stays in time order)."""
    card = card_profiles().sample(1).iloc[0]
    last = card_history(card.cc_num, 1)
    start = pd.Timestamp(last.trans_ts.iloc[0]) if not last.empty else pd.Timestamp("2020-06-21")
    for i in range(1, n + 1):
        msg = {
            "trans_num": f"burst-{uuid.uuid4().hex}",
            "trans_date_trans_time": (start + pd.Timedelta(minutes=2 * i)).strftime("%Y-%m-%d %H:%M:%S"),
            "cc_num": card.cc_num, "merchant": "fraud_Demo Burst Store",
            "category": random.choice(["shopping_net", "misc_net", "grocery_pos"]),
            "amt": round(random.uniform(300, 1100), 2), "gender": card.gender, "city": card.city,
            "state": card.state, "lat": card.lat, "long": card.long, "city_pop": int(card.city_pop),
            "dob": card.dob, "merch_lat": card.lat + 0.4, "merch_long": card.long - 0.4, "is_fraud": 1,
        }
        kafka_producer().produce(TOPIC, key=card.cc_num, value=json.dumps(msg))
    kafka_producer().flush(5)
    return card.cc_num


# ---------- Tab 1: Live Alerts ----------

def shap_chart(alert):
    s = pd.DataFrame({"feature": list(alert["shap"].keys()), "shap": list(alert["shap"].values())})
    s["label"] = [f"{alert['feature_values'].get(f, '')} = {LABELS.get(f, f)}" for f in s.feature]
    s = s.reindex(s.shap.abs().sort_values(ascending=False).index).head(10)
    s["direction"] = np.where(s.shap > 0, "toward fraud", "toward legit")
    return alt.Chart(s).mark_bar().encode(
        x=alt.X("shap:Q", title="SHAP value (push on the fraud score)"),
        y=alt.Y("label:N", sort=None, title=None, axis=alt.Axis(labelLimit=340)),
        color=alt.Color("direction:N", scale=alt.Scale(domain=["toward fraud", "toward legit"],
                                                       range=["#ff0051", "#008bfb"]), legend=None),
        tooltip=["label", alt.Tooltip("shap:Q", format="+.2f")],
    ).properties(height=320)


def remember_pick():
    """Runs only when the user clicks a row: save WHICH alert (not its row position, which shifts
    as new alerts arrive every refresh) so the drill-down stays on it."""
    rows = st.session_state.alert_feed["selection"]["rows"]
    st.session_state.picked = st.session_state.feed_alerts.iloc[rows[0]] if rows else None


def live_alerts():
    window = 5
    alerts, metrics = recent_alerts(window), recent_metrics(window)
    backlog = kafka_backlog(metrics)
    st.markdown(f"**{staleness_badge(metrics, backlog)}**")

    last_min = metrics[metrics.batch_end > datetime.now(timezone.utc) - timedelta(minutes=1)] if not metrics.empty else metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Swipes scored / sec (last 1 min)", f"{last_min['rows'].sum() / 60:,.0f}" if not last_min.empty else "0")
    c2.metric(f"Alerts (last {window} min)", f"{len(alerts):,}")
    c3.metric(f"Fraud $ caught (last {window} min)", f"${metrics['fraud_amt_caught'].sum():,.0f}" if not metrics.empty else "$0",
              help="Dollar value of truly fraudulent swipes the model flagged.")
    c4.metric(f"Fraud $ missed (last {window} min)", f"${metrics['fraud_amt_missed'].sum():,.0f}" if not metrics.empty else "$0",
              help="Dollar value of truly fraudulent swipes the model let through.")

    if alerts.empty:
        st.info("No alerts in the last few minutes.")
        return
    feed = pd.DataFrame({
        "Time (UTC)": pd.to_datetime(alerts.scored_at).dt.strftime("%H:%M:%S"),
        "Card": alerts.cc_num.map(masked),
        "Merchant": alerts.merchant.str.replace(r"^fraud_", "", regex=True),
        "Amount": alerts.amt.map("${:,.2f}".format),
        "Score": alerts.score.round(3),
        "Truth": np.where(alerts.is_fraud, "✅ fraud", "⚠️ false alarm"),
        "Why flagged": alerts.reasons,
    })
    st.caption("Select a row to see why it was flagged and that card's recent activity. "
               "'Truth' comes from the dataset's labels - a real bank learns it weeks later via chargebacks.")
    st.session_state.feed_alerts = alerts  # the rows as displayed, for remember_pick
    st.dataframe(feed, hide_index=True, width="stretch", height=300,
                 on_select=remember_pick, selection_mode="single-row", key="alert_feed")
    alert = st.session_state.get("picked")
    if alert is not None:
        left, right = st.columns([3, 2])
        with left:
            st.subheader(f"Why {masked(alert.cc_num)} was flagged")
            st.caption(f"${alert.amt:,.2f} at {str(alert.merchant).removeprefix('fraud_')} · score {alert.score:.3f} · "
                       f"{'true fraud' if alert.is_fraud else 'false alarm'}")
            st.altair_chart(shap_chart(alert), width="stretch")
        with right:
            st.subheader("Card's last 10 swipes")
            h = card_history(alert.cc_num)
            if not h.empty:
                h["trans_ts"] = pd.to_datetime(h.trans_ts).dt.strftime("%Y-%m-%d %H:%M")
                h["merchant"] = h.merchant.str.replace(r"^fraud_", "", regex=True)
                st.dataframe(h, hide_index=True, width="stretch")


# ---------- Tab 2: Health & Model ----------

@st.fragment(run_every=REFRESH)
def health():
    window = st.radio("Window", [5, 15, 60], index=1, horizontal=True, format_func=lambda m: f"last {m} min")
    m = recent_metrics(window)
    if m.empty:
        st.info("No batches in this window yet.")
        return

    st.subheader("Throughput")
    per_10s = m.set_index("batch_end")
    st.line_chart(pd.DataFrame({
        "Arriving (producer → Kafka)": per_10s["produced_rate"].resample("10s").mean(),
        "Processed (Spark → model → Cassandra)": per_10s["rows"].resample("10s").sum() / 10,
    }), y_label="swipes / sec")
    st.caption("If arrivals stay above processed, the Kafka backlog grows. Measured capacity on this "
               "2 × 2-vCPU setup: ~9,600 swipes/sec.")

    lat = hist_sum(m.latency_hist_ms, len(LATENCY_EDGES_MS) + 1)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latency p50", fmt_ms(percentile_upper(lat, 50)))
    c2.metric("Latency p95", fmt_ms(percentile_upper(lat, 95)))
    c3.metric("Latency p99", fmt_ms(percentile_upper(lat, 99)))
    backlog = kafka_backlog(m)
    c4.metric("Kafka backlog", f"{backlog:,} swipes" if backlog is not None else "—")
    st.caption(f"Latency = Kafka send stamp → Cassandra write confirmed, per swipe. Micro-batch trigger: "
               f"{m.trigger.iloc[-1]}. Values are bucket upper bounds (exact counts per bucket, merged across batches).")

    st.subheader("Model, graded live against the true labels")
    per_min = m.assign(minute=m.batch_end.dt.floor("min")).groupby("minute")[["tp", "fp", "fn"]].sum()
    per_min["precision"] = per_min.tp / (per_min.tp + per_min.fp).replace(0, np.nan)
    per_min["recall"] = per_min.tp / (per_min.tp + per_min.fn).replace(0, np.nan)
    st.line_chart(per_min[["precision", "recall"]], y_label="per minute")
    st.caption("Real banks get fraud labels 30-90 days later (chargebacks). This synthetic dataset has them "
               "immediately, so the model can be graded live.")

    fraud_h = hist_sum(m.score_hist_fraud, SCORE_BUCKETS)
    legit_h = hist_sum(m.score_hist_legit, SCORE_BUCKETS)
    cutoff = st.slider("What-if cutoff (uncalibrated score)", 0.05, 0.95, 0.85, 0.05,
                       help="Scores come from a model trained with heavy fraud weighting, so they rank "
                            "well but aren't true probabilities.")
    k = int(round(cutoff * SCORE_BUCKETS))
    tp, fn = int(fraud_h[k:].sum()), int(fraud_h[:k].sum())
    fp, tn = int(legit_h[k:].sum()), int(legit_h[:k].sum())
    left, right = st.columns(2)
    with left:
        st.markdown(f"**At cutoff {cutoff:.2f}**")
        st.table(pd.DataFrame({"Flagged": [f"{tp:,} caught", f"{fp:,} false alarms"],
                               "Not flagged": [f"{fn:,} missed", f"{tn:,} correctly ignored"]},
                              index=["Actually fraud", "Actually legit"]))
        st.markdown(f"Recall **{tp / max(tp + fn, 1):.1%}** · Precision **{tp / max(tp + fp, 1):.1%}** · "
                    f"Model's own cutoff in this window: {int(m.tp.sum()):,} caught, {int(m.fp.sum()):,} false alarms")
    with right:
        dist = pd.DataFrame({
            "score": np.tile(np.arange(SCORE_BUCKETS) / SCORE_BUCKETS, 2),
            "share": np.concatenate([fraud_h / max(fraud_h.sum(), 1), legit_h / max(legit_h.sum(), 1)]),
            "label": ["fraud"] * SCORE_BUCKETS + ["legit"] * SCORE_BUCKETS,
        })
        st.altair_chart(alt.Chart(dist).mark_bar(opacity=0.7).encode(
            x=alt.X("score:Q", bin=alt.Bin(step=1 / SCORE_BUCKETS), title="score"),
            y=alt.Y("share:Q", stack=None, title="share of swipes"),
            color=alt.Color("label:N", scale=alt.Scale(range=["#ff0051", "#008bfb"])),
        ).properties(height=260, title="Score distribution (watch for drift)"), width="stretch")


# ---------- Layout ----------

head, pause, button = st.columns([4, 1, 1])
head.title("🛡️ Real-Time Fraud Detection")
paused = pause.toggle("⏸ Pause live updates", help="Freeze the alert feed so rows don't move while you click")
if button.button("💥 Inject fraud burst", help="Send 5 stolen-card-style swipes for a random real card"):
    card = inject_burst()
    button.success(f"Sent 5 swipes for {masked(card)} - watch the feed")

tab1, tab2 = st.tabs(["🚨 Live Alerts", "📈 Health & Model"])
with tab1:
    st.fragment(run_every=None if paused else REFRESH)(live_alerts)()
with tab2:
    health()
