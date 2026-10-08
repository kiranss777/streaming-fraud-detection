"""Spark Structured Streaming: Kafka swipes -> features -> XGBoost score + SHAP -> Cassandra.

Spark reads Kafka in micro-batches (TRIGGER) and tracks offsets in its checkpoint. Each batch
goes to `Processor` (foreachBatch), which:
  1. computes card-history features with vectorized pandas, from per-card state it keeps
     in memory and snapshots after every batch, tagged with Spark's batch id (crash-safe)
  2. scores with the model; SHAP + plain-English reasons for flagged swipes only
  3. writes every swipe to transactions_by_card and flagged ones to alerts_by_minute
     (Spark Cassandra connector)
  4. writes one pipeline_metrics row: rates, latency histogram (Kafka send stamp -> Cassandra
     write done), score histograms, confusion counts, $ caught/missed, Kafka offsets reached

Why not applyInPandasWithState: measured on this 2-vCPU host it spent ~2-4 s per batch shuttling
data between the JVM and Python workers, for ~0.1 s of actual Python work.
"""
import os
import pickle
import tarfile
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from cassandra.cluster import Cluster
from pyspark.sql import SparkSession, functions as F, types as T

import features
import scoring

BUCKET = os.environ["DATA_BUCKET"]
DATA_HOST = os.environ["DATA_HOST_IP"]
MODEL_KEY = os.environ.get("MODEL_S3_KEY", "models/current/model.tar.gz")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "unknown")
TRIGGER = os.environ.get("TRIGGER", "1 second")
MAX_PER_BATCH = int(os.environ.get("MAX_OFFSETS_PER_TRIGGER", "50000"))

WORK = Path("/work")
MODEL_DIR = str(WORK / "model")
CHECKPOINT = Path("/checkpoint")
STATE_DIR = CHECKPOINT / "card-state"  # card histories, one snapshot per completed batch
LATENCY_EDGES_MS = [100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000, 10000]  # 12 buckets, last = 10 s+
SCORE_BUCKETS = 20  # 0.00-0.05, 0.05-0.10, ...

MESSAGE = T.StructType([
    T.StructField("trans_num", T.StringType()), T.StructField("trans_date_trans_time", T.StringType()),
    T.StructField("cc_num", T.StringType()), T.StructField("merchant", T.StringType()),
    T.StructField("category", T.StringType()), T.StructField("amt", T.DoubleType()),
    T.StructField("gender", T.StringType()), T.StructField("city", T.StringType()),
    T.StructField("state", T.StringType()), T.StructField("lat", T.DoubleType()),
    T.StructField("long", T.DoubleType()), T.StructField("city_pop", T.LongType()),
    T.StructField("dob", T.StringType()), T.StructField("merch_lat", T.DoubleType()),
    T.StructField("merch_long", T.DoubleType()), T.StructField("is_fraud", T.LongType()),
])
TRANSACTION_ROWS = T.StructType([
    T.StructField("cc_num", T.StringType()), T.StructField("trans_ts", T.TimestampType()),
    T.StructField("trans_num", T.StringType()), T.StructField("amt", T.DoubleType()),
    T.StructField("merchant", T.StringType()), T.StructField("category", T.StringType()),
    T.StructField("score", T.FloatType()), T.StructField("flagged", T.BooleanType()),
    T.StructField("is_fraud", T.BooleanType()),
])
ALERT_ROWS = T.StructType(TRANSACTION_ROWS.fields + [
    T.StructField("reasons", T.StringType()), T.StructField("shap_json", T.StringType()),
    T.StructField("values_json", T.StringType()),
])


def prepare():
    """Fetch the model; build each card's history as of the end of the training period."""
    s3 = boto3.client("s3")
    WORK.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        s3.download_file(BUCKET, MODEL_KEY, f"{tmp}/model.tar.gz")
        with tarfile.open(f"{tmp}/model.tar.gz") as t:
            t.extractall(MODEL_DIR)

    def parquet(prefix, columns=None):
        with tempfile.TemporaryDirectory() as tmp:
            for obj in s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)["Contents"]:
                if obj["Key"].endswith(".parquet"):
                    s3.download_file(BUCKET, obj["Key"], f"{tmp}/{Path(obj['Key']).name}")
            return pd.read_parquet(tmp, columns=columns)

    train = parquet("processed/train/", ["cc_num", "trans_ts", "amt"])
    tail = train[train.trans_ts >= train.trans_ts.max() - pd.Timedelta(days=1)]
    return features.build_seed(parquet("processed/card_state/"), tail)


class CardState:
    """Per-card histories, snapshotted after each batch and tagged with Spark's batch id.

    If the job crashes mid-batch, Spark replays that batch (same id) on restart. We then start
    from the snapshot of the batch BEFORE it, so no swipe is counted twice. Cassandra writes are
    upserts by primary key, so replaying them is harmless too."""

    def __init__(self, seed):
        self.seed = seed
        self.histories, self.batch_id = None, None
        STATE_DIR.mkdir(parents=True, exist_ok=True)

    def before(self, batch_id):
        if self.histories is not None and self.batch_id == batch_id - 1:
            return self.histories  # normal case: state is current
        older = sorted(int(p.stem) for p in STATE_DIR.glob("*.pkl") if int(p.stem) < batch_id)
        self.histories = pickle.loads((STATE_DIR / f"{older[-1]}.pkl").read_bytes()) if older else {}
        print(f"card state restored from batch {older[-1] if older else 'none (fresh start)'}", flush=True)
        return self.histories

    def save(self, batch_id):
        (STATE_DIR / f"{batch_id}.pkl").write_bytes(pickle.dumps(self.histories))
        self.batch_id = batch_id
        for old in sorted(STATE_DIR.glob("*.pkl"), key=lambda p: int(p.stem))[:-3]:  # keep the last 3
            old.unlink()


class Processor:
    """foreachBatch handler: features -> score -> Cassandra -> metrics, for one micro-batch."""

    def __init__(self, spark, seed):
        self.spark = spark
        self.state = CardState(seed)
        self.scorer = scoring.Scorer(MODEL_DIR)
        self.session = Cluster([DATA_HOST]).connect("fraud")
        self.insert_metrics = self.session.prepare(
            "INSERT INTO pipeline_metrics (day, batch_end, batch_id, rows, produced_rate, processed_rate, "
            "written_rate, latency_hist_ms, score_hist_fraud, score_hist_legit, tp, fp, fn, tn, "
            "fraud_amt_caught, fraud_amt_missed, max_offsets, model_version, trigger) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)")

    def write(self, pdf, schema, table, extra=()):
        df = self.spark.createDataFrame(pdf[schema.fieldNames()], schema)
        if extra:  # alerts: JSON strings become Cassandra maps; 'flagged' is implied by the table
            df = df.select(*[c for c in df.columns if c not in ("shap_json", "values_json", "flagged")], *extra)
        (df.write.format("org.apache.spark.sql.cassandra")
         .options(keyspace="fraud", table=table).mode("append").save())

    def __call__(self, batch, batch_id):
        started = time.time()
        swipes = batch.toPandas()  # one Arrow transfer of this batch's Kafka messages
        if swipes.empty:
            return

        scored = self.scorer.score(features.compute(swipes, self.state.before(batch_id), self.state.seed))
        scored["is_fraud"] = scored["is_fraud"] == 1
        done_scoring = time.time()

        now = datetime.now(timezone.utc)
        self.write(scored, TRANSACTION_ROWS, "transactions_by_card")
        alerts = scored[scored.flagged]
        if not alerts.empty:
            self.write(alerts, ALERT_ROWS, "alerts_by_minute", extra=(
                F.lit(now.replace(second=0, microsecond=0)).alias("minute"), F.lit(now).alias("scored_at"),
                F.from_json("shap_json", T.MapType(T.StringType(), T.FloatType())).alias("shap"),
                F.from_json("values_json", T.MapType(T.StringType(), T.StringType())).alias("feature_values")))
        written = time.time()
        self.state.save(batch_id)

        # Latency = Cassandra write confirmed - Kafka send stamp (both on this host's clock).
        sent_ms = pd.to_datetime(scored.kafka_ts).astype("datetime64[ms]").astype("int64").to_numpy()
        latency_ms = written * 1000 - sent_ms
        span_s = (sent_ms.max() - sent_ms.min()) / 1000
        bucket = np.minimum((scored.score.to_numpy() * SCORE_BUCKETS).astype(int), SCORE_BUCKETS - 1)
        fraud, flagged, amt, n = (scored.is_fraud.to_numpy(), scored.flagged.to_numpy(),
                                  scored.amt.to_numpy(), len(scored))
        self.session.execute(self.insert_metrics, (
            now.date(), now, batch_id, n,
            n / span_s if span_s > 0 else None,
            n / (done_scoring - started), n / (written - done_scoring),
            np.bincount(np.searchsorted(LATENCY_EDGES_MS, latency_ms, side="right"),
                        minlength=len(LATENCY_EDGES_MS) + 1).tolist(),
            np.bincount(bucket[fraud], minlength=SCORE_BUCKETS).tolist(),
            np.bincount(bucket[~fraud], minlength=SCORE_BUCKETS).tolist(),
            int((flagged & fraud).sum()), int((flagged & ~fraud).sum()),
            int((~flagged & fraud).sum()), int((~flagged & ~fraud).sum()),
            float(amt[flagged & fraud].sum()), float(amt[~flagged & fraud].sum()),
            {int(p): int(o) for p, o in scored.groupby("partition")["offset"].max().items()},
            MODEL_VERSION, TRIGGER,
        ))
        p50, p95, p99 = np.percentile(latency_ms, [50, 95, 99])
        print(f"batch {batch_id}: {n:,} swipes, {int(flagged.sum())} flagged | latency p50 {p50:,.0f} / "
              f"p95 {p95:,.0f} / p99 {p99:,.0f} ms | features+score {1000 * (done_scoring - started):,.0f} ms, "
              f"cassandra {1000 * (written - done_scoring):,.0f} ms", flush=True)


def main():
    seed = prepare()
    spark = (
        SparkSession.builder.appName("fraud-stream")
        .master("local[2]")  # both vCPUs of the compute host
        .config("spark.driver.memory", "2g")
        .config("spark.jars.packages", os.environ["SPARK_PACKAGES"])
        .config("spark.sql.session.timeZone", "UTC")  # same as Glue
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")  # fast toPandas / createDataFrame
        .config("spark.cassandra.connection.host", DATA_HOST)
        .config("spark.cassandra.output.concurrent.writes", "32")  # in-flight writes per task (default 5)
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    processor = Processor(spark, seed)
    print(f"model {MODEL_VERSION} loaded (cutoff {processor.scorer.threshold:.3f}); "
          f"seeded history for {len(seed):,} cards", flush=True)

    swipes = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", f"{DATA_HOST}:9092")
        .option("subscribe", "transactions")
        .option("startingOffsets", "latest")
        .option("maxOffsetsPerTrigger", MAX_PER_BATCH)  # caps batch size so a backlog drains in steps
        .option("failOnDataLoss", "false")
        .load()
        .select(F.from_json(F.col("value").cast("string"), MESSAGE).alias("m"),
                F.col("timestamp").alias("kafka_ts"), "partition", "offset")
        .select("m.*", "kafka_ts", "partition", "offset")
    )
    (swipes.writeStream.foreachBatch(processor)
     .option("checkpointLocation", str(CHECKPOINT / "spark"))
     .trigger(processingTime=TRIGGER)
     .start()
     .awaitTermination())


if __name__ == "__main__":
    main()
