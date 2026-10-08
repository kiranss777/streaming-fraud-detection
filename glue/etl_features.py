"""Glue ETL: raw Kaggle CSVs -> model-ready features (Parquet).

Reads   s3://<bucket>/raw/fraudTrain.csv, raw/fraudTest.csv
Writes  s3://<bucket>/processed/train/       features + label
        s3://<bucket>/processed/test/        features + label
        s3://<bucket>/processed/card_state/  each card's running totals as of the end of train -
                                             the starting point for the live stream (via Cassandra)

Card features are point-in-time: each transaction only sees that card's EARLIER
transactions, exactly what the live stream will know when a swipe arrives.
"""
import sys

from awsglue.utils import getResolvedOptions
from pyspark.sql import SparkSession, Window, functions as F
from pyspark.sql.types import DoubleType, IntegerType, LongType, StringType, StructField, StructType

args = getResolvedOptions(sys.argv, ["DATA_BUCKET"])
BUCKET = f"s3://{args['DATA_BUCKET']}"

spark = SparkSession.builder.appName("fraud-etl-features").getOrCreate()

# Explicit schema: skips a second pass to infer types, and names the unnamed index column.
RAW_SCHEMA = StructType([
    StructField("row_id", LongType()),
    StructField("trans_date_trans_time", StringType()),
    StructField("cc_num", StringType()),  # an ID, not a number to do math on
    StructField("merchant", StringType()),
    StructField("category", StringType()),
    StructField("amt", DoubleType()),
    StructField("first", StringType()),
    StructField("last", StringType()),
    StructField("gender", StringType()),
    StructField("street", StringType()),
    StructField("city", StringType()),
    StructField("state", StringType()),
    StructField("zip", StringType()),
    StructField("lat", DoubleType()),
    StructField("long", DoubleType()),
    StructField("city_pop", LongType()),
    StructField("job", StringType()),
    StructField("dob", StringType()),
    StructField("trans_num", StringType()),
    StructField("unix_time", LongType()),
    StructField("merch_lat", DoubleType()),
    StructField("merch_long", DoubleType()),
    StructField("is_fraud", IntegerType()),
])


def read_raw(name):
    return (
        spark.read
        .option("header", True)
        .option("quote", '"')
        .option("escape", '"')  # fields like "fraud_Rippin, Kub and Mann" contain commas
        .schema(RAW_SCHEMA)
        .csv(f"{BUCKET}/raw/{name}")
    )


def haversine_km(lat1, lon1, lat2, lon2):
    dlat = F.radians(lat2 - lat1)
    dlon = F.radians(lon2 - lon1)
    a = F.sin(dlat / 2) ** 2 + F.cos(F.radians(lat1)) * F.cos(F.radians(lat2)) * F.sin(dlon / 2) ** 2
    return 2 * 6371.0 * F.asin(F.sqrt(a))


def row_features(df):
    """Features computable from a single transaction - the stream can compute these too."""
    ts = F.to_timestamp("trans_date_trans_time", "yyyy-MM-dd HH:mm:ss")
    return df.select(
        "trans_num",
        "cc_num",
        ts.alias("trans_ts"),
        F.regexp_replace("merchant", "^fraud_", "").alias("merchant"),  # generator artifact, not a label
        "category",
        "amt",
        "gender",
        "city",
        "state",
        "city_pop",
        F.floor(F.months_between(ts, F.to_date("dob")) / 12).cast("int").alias("age"),
        F.hour(ts).alias("hour"),
        F.dayofweek(ts).alias("day_of_week"),
        haversine_km(F.col("lat"), F.col("long"), F.col("merch_lat"), F.col("merch_long")).alias("distance_km"),
        "is_fraud",
    )


train = row_features(read_raw("fraudTrain.csv")).withColumn("split", F.lit("train"))
test = row_features(read_raw("fraudTest.csv")).withColumn("split", F.lit("test"))

# One continuous timeline per card: test (Jun-Dec 2020) directly follows train, so a
# test transaction's history includes the card's train-period transactions, as in real life.
# Only past amounts/times are used - never labels - so this leaks nothing.
txns = train.unionByName(test).withColumn("ts_sec", F.col("trans_ts").cast("long"))

by_card = Window.partitionBy("cc_num").orderBy("ts_sec", "trans_num")
all_before = by_card.rowsBetween(Window.unboundedPreceding, -1)  # every earlier txn, not this one
by_card_time = Window.partitionBy("cc_num").orderBy("ts_sec")
last_hour = by_card_time.rangeBetween(-3600, -1)                 # earlier txns in the past hour
last_day = by_card_time.rangeBetween(-86400, -1)                 # ...in the past 24 hours

features = (
    txns
    # Velocity: thieves use a stolen card fast, before it gets blocked.
    .withColumn("secs_since_last_txn", F.col("ts_sec") - F.lag("ts_sec").over(by_card))
    .withColumn("txn_count_1h", F.count("*").over(last_hour))
    .withColumn("txn_count_24h", F.count("*").over(last_day))
    .withColumn("amt_sum_24h", F.coalesce(F.sum("amt").over(last_day), F.lit(0.0)))
    # Spending habits so far. A card's first txn has no history (count 0, avg null) -
    # the model learns what "brand-new card" looks like.
    .withColumn("card_txn_count", F.count("*").over(all_before))
    .withColumn("card_avg_amt", F.avg("amt").over(all_before))
    .withColumn("card_std_amt", F.stddev("amt").over(all_before))
    .withColumn("amt_to_card_avg", F.col("amt") / F.col("card_avg_amt"))
    .withColumn("amt_zscore", (F.col("amt") - F.col("card_avg_amt")) / F.col("card_std_amt"))
)

for split in ("train", "test"):
    (features.filter(F.col("split") == split).drop("split", "ts_sec")
     .write.mode("overwrite").parquet(f"{BUCKET}/processed/{split}/"))

# Running totals per card at the end of train. The stream replays the test period as
# "live" traffic and continues these totals (count/sum/sum of squares give avg and std).
card_state = train.groupBy("cc_num").agg(
    F.count("*").alias("txn_count"),
    F.sum("amt").alias("amt_sum"),
    F.sum(F.col("amt") * F.col("amt")).alias("amt_sumsq"),
    F.max("trans_ts").alias("last_txn_ts"),
)
card_state.write.mode("overwrite").parquet(f"{BUCKET}/processed/card_state/")
