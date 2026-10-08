"""Download the Kaggle fraud dataset and upload the raw CSVs to s3://$DATA_BUCKET/raw/.

The files only touch a temp folder, which is deleted afterwards.

Usage (from the repo root, venv active):  python src/upload_raw_data.py
"""
import os
import tempfile
from pathlib import Path

import boto3
from dotenv import load_dotenv

# The kaggle package authenticates at import time, so the token must be loaded first.
load_dotenv()
from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: E402

DATASET = "kartik2112/fraud-detection"
FILES = ["fraudTrain.csv", "fraudTest.csv"]


def main():
    bucket = os.environ["DATA_BUCKET"]
    s3 = boto3.client("s3")
    api = KaggleApi()
    api.authenticate()

    with tempfile.TemporaryDirectory() as tmp:
        print(f"Downloading {DATASET} from Kaggle...")
        api.dataset_download_files(DATASET, path=tmp, unzip=True, quiet=False)

        for name in FILES:
            local = Path(tmp) / name
            key = f"raw/{name}"
            print(f"Uploading {name} ({local.stat().st_size / 1e6:.0f} MB) -> s3://{bucket}/{key}")
            s3.upload_file(str(local), bucket, key)

    print("Done.")


if __name__ == "__main__":
    main()
