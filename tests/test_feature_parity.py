"""Train/serve parity: do the stream's features match what the model was trained on?

Runs every test-period swipe through stream/features.py (the exact code the Spark job uses),
starting from the same end-of-training seed the job uses, and compares all 17 model features
against Glue's processed/test output.

Usage (from the repo root, venv active):  python tests/test_feature_parity.py
"""
import io
import os
import sys
import tempfile
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stream"))
import features  # noqa: E402

MODEL_FEATURES = [
    "category", "amt", "gender", "city_pop", "age", "hour", "day_of_week", "distance_km",
    *features.HISTORY_FEATURES,
]
TOLERANCE = 1e-6  # relative, for floating-point features


def read_parquet_prefix(s3, bucket, prefix, columns=None):
    with tempfile.TemporaryDirectory() as tmp:
        for obj in s3.list_objects_v2(Bucket=bucket, Prefix=prefix)["Contents"]:
            if obj["Key"].endswith(".parquet"):
                s3.download_file(bucket, obj["Key"], str(Path(tmp) / Path(obj["Key"]).name))
        return pd.read_parquet(tmp, columns=columns)


def main():
    load_dotenv()
    bucket = os.environ["DATA_BUCKET"]
    s3 = boto3.client("s3")

    print("loading data from S3...")
    raw = pd.read_csv(io.BytesIO(s3.get_object(Bucket=bucket, Key="raw/fraudTest.csv")["Body"].read()),
                      dtype={"cc_num": str})
    glue = read_parquet_prefix(s3, bucket, "processed/test/").set_index("trans_num")
    card_state = read_parquet_prefix(s3, bucket, "processed/card_state/")
    train = read_parquet_prefix(s3, bucket, "processed/train/", columns=["cc_num", "trans_ts", "amt"])
    tail = train[train.trans_ts >= train.trans_ts.max() - pd.Timedelta(days=1)]

    print(f"computing stream features for {len(raw):,} swipes...")
    seed = features.build_seed(card_state, tail)
    stream = features.compute(raw, {}, seed).set_index("trans_num")

    glue = glue.loc[stream.index]
    failures = 0
    print(f"\n{'feature':22}{'mismatches':>12}")
    for f in MODEL_FEATURES:
        a, b = stream[f], glue[f]
        if not (pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b)):
            bad = (a.astype(str) != b.astype(str))
        else:
            a, b = a.astype(float), b.astype(float)
            both_nan = a.isna() & b.isna()
            close = np.isclose(a, b, rtol=TOLERANCE, atol=1e-9)
            bad = ~(both_nan | close)
        n_bad = int(bad.sum())
        failures += n_bad
        print(f"{f:22}{n_bad:>12,}")
        if n_bad:
            print(pd.DataFrame({"stream": a[bad], "glue": b[bad]}).head(3).to_string())

    print(f"\n{len(stream):,} swipes x {len(MODEL_FEATURES)} features: "
          f"{'PASS - identical' if failures == 0 else f'FAIL - {failures:,} mismatches'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
