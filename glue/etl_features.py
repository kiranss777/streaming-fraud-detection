"""Glue ETL: raw Kaggle CSVs -> model-ready features (Parquet).

Reads   s3://<bucket>/raw/fraudTrain.csv, raw/fraudTest.csv
Writes  s3://<bucket>/processed/train/          features + label
        s3://<bucket>/processed/test/           features + label
        s3://<bucket>/processed/card_profiles/  per-card spending stats (train only),
                                                later loaded into Cassandra for streaming lookups
"""
import sys

from awsglue.utils import getResolvedOptions
from pyspark.sql import SparkSession, functions as F
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


train = row_features(read_raw("fraudTrain.csv"))
test = row_features(read_raw("fraudTest.csv"))

# Per-card spending profile, from train only so test rows never leak into it.
card_profiles = train.groupBy("cc_num").agg(
    F.avg("amt").alias("card_avg_amt"),
    F.stddev("amt").alias("card_std_amt"),
    F.count("*").alias("card_txn_count"),
)


def add_card_features(df):
    return (
        df.join(card_profiles, "cc_num", "left")  # unseen cards get nulls; XGBoost handles them
        .withColumn("amt_to_card_avg", F.col("amt") / F.col("card_avg_amt"))
        .withColumn("amt_zscore", (F.col("amt") - F.col("card_avg_amt")) / F.col("card_std_amt"))
    )


add_card_features(train).write.mode("overwrite").parquet(f"{BUCKET}/processed/train/")
add_card_features(test).write.mode("overwrite").parquet(f"{BUCKET}/processed/test/")
card_profiles.write.mode("overwrite").parquet(f"{BUCKET}/processed/card_profiles/")
