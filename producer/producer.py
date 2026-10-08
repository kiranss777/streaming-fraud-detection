"""Replays the test-period transactions (Jun-Dec 2020) into Kafka as if they were live swipes.

- Each message is one swipe (JSON), keyed by card number: Kafka sends all of a card's swipes
  to the same partition, in order - Spark's per-card history features depend on that.
- Kafka stamps each message with the wall-clock time it was sent (the record timestamp);
  Spark measures end-to-end latency from that stamp.

Usage (on the compute host, in /opt/fraud):
  docker compose run --rm producer --rate 100                # calm demo
  docker compose run --rm producer --rate 10000 --loops 3    # load test
"""
import argparse
import io
import itertools
import os
import time

import boto3
import pandas as pd
from confluent_kafka import Producer

TOPIC = "transactions"

# What a card network would send - no names or street addresses.
# is_fraud is NOT a model input: it rides along only so the dashboard can grade the model live.
FIELDS = [
    "trans_num", "trans_date_trans_time", "cc_num", "merchant", "category", "amt", "gender",
    "city", "state", "lat", "long", "city_pop", "dob", "merch_lat", "merch_long", "is_fraud",
]


def load(bucket):
    body = boto3.client("s3").get_object(Bucket=bucket, Key="raw/fraudTest.csv")["Body"].read()
    df = pd.read_csv(io.BytesIO(body), usecols=FIELDS, dtype={"cc_num": str})
    df["ts"] = pd.to_datetime(df["trans_date_trans_time"])
    return df.sort_values("ts", kind="stable").reset_index(drop=True)


def messages(df, loop):
    """(key, json) pairs. Loop k replays the period shifted k periods later with unique ids,
    so replays look like new swipes and each card's history keeps moving forward in time."""
    if loop:
        span = df.ts.max() - df.ts.min() + pd.Timedelta(seconds=1)
        df = df.assign(ts=df.ts + loop * span, trans_num=df.trans_num + f"-r{loop}")
    out = df.assign(trans_date_trans_time=df.ts.dt.strftime("%Y-%m-%d %H:%M:%S")).drop(columns="ts")
    return zip(out.cc_num.tolist(), out.to_json(orient="records", lines=True).splitlines())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rate", type=float, default=100, help="messages/sec; 0 = as fast as possible")
    p.add_argument("--limit", type=int, default=0, help="stop after N messages; 0 = no limit")
    p.add_argument("--loops", type=int, default=1, help="replay the test period this many times")
    args = p.parse_args()

    df = load(os.environ["DATA_BUCKET"])
    print(f"loaded {len(df):,} test transactions; target rate: {args.rate or 'max'} msg/s", flush=True)

    failed = 0

    def on_delivery(err, _msg):
        nonlocal failed
        if err:
            failed += 1

    producer = Producer({
        "bootstrap.servers": f"{os.environ['DATA_HOST_IP']}:9092",
        "linger.ms": 5,              # wait up to 5 ms to batch messages: far fewer network round trips
        "compression.type": "lz4",   # smaller batches on the wire, cheap on CPU
        "acks": "1",                 # broker confirms each batch (single broker, so 1 = all)
        "queue.buffering.max.messages": 500_000,
    })

    stream = itertools.chain.from_iterable(messages(df, k) for k in range(args.loops))
    if args.limit:
        stream = itertools.islice(stream, args.limit)

    sent = 0
    start = report_at = time.monotonic()
    report_sent = 0
    for key, value in stream:
        if args.rate:  # pace: never get ahead of rate x elapsed time
            ahead = sent / args.rate - (time.monotonic() - start)
            if ahead > 0:
                time.sleep(ahead)
        while True:
            try:
                producer.produce(TOPIC, key=key, value=value, on_delivery=on_delivery)
                break
            except BufferError:  # local send queue full - let it drain
                producer.poll(0.05)
        producer.poll(0)
        sent += 1

        now = time.monotonic()
        if now - report_at >= 5:
            print(f"sent {sent:,}  ({(sent - report_sent) / (now - report_at):,.0f} msg/s)", flush=True)
            report_at, report_sent = now, sent

    producer.flush()
    elapsed = time.monotonic() - start
    print(f"done: {sent:,} sent, {failed:,} failed, {elapsed:.1f}s, {sent / elapsed:,.0f} msg/s avg", flush=True)


if __name__ == "__main__":
    main()
