-- Cassandra is query-first: one table per question, so every read hits ONE partition
-- (one server-side bucket of rows) and never scans the table.
-- Single node, so replication_factor 1.

CREATE KEYSPACE IF NOT EXISTS fraud
  WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};

-- Q: "What has this card done recently?"
--    Dashboard drill-down (card's last 10 txns). Every scored transaction lands here,
--    so this table takes the full write load (the 10k writes/sec claim).
-- Partition = one card (~2k txns over 2 years, ~200 KB: small and bounded).
-- Writes spread across ~1,000 cards, so no single hot partition.
CREATE TABLE IF NOT EXISTS fraud.transactions_by_card (
    cc_num      text,
    trans_ts    timestamp,   -- event time (when the swipe happened)
    trans_num   text,
    amt         double,
    merchant    text,
    category    text,
    score       float,
    flagged     boolean,
    is_fraud    boolean,     -- true label (synthetic data has it; real banks learn it weeks later)
    PRIMARY KEY ((cc_num), trans_ts, trans_num)
) WITH CLUSTERING ORDER BY (trans_ts DESC, trans_num ASC);

-- Q: "What got flagged in the last N minutes?"
--    Dashboard alert feed. Partition = one minute of processing time, so "last 5 min" = 5
--    partition reads. ~50 alerts/sec at 10k TPS = ~3k rows per partition: fine.
CREATE TABLE IF NOT EXISTS fraud.alerts_by_minute (
    minute          timestamp,          -- scored_at truncated to the minute
    scored_at       timestamp,
    trans_num       text,
    cc_num          text,               -- dashboard shows only the last 4 digits
    trans_ts        timestamp,
    amt             double,
    merchant        text,
    category        text,
    score           float,
    is_fraud        boolean,            -- drives the TP/FP badge
    reasons         text,               -- plain-English top 3, e.g. "8x usual spend, 11 PM, ..."
    shap            map<text, float>,   -- feature -> SHAP value (the per-alert bar chart)
    feature_values  map<text, text>,    -- feature -> value shown beside each bar
    PRIMARY KEY ((minute), scored_at, trans_num)
) WITH CLUSTERING ORDER BY (scored_at DESC, trans_num ASC);

-- Q: "How is the pipeline doing over time?"
--    Dashboard health tab. One row per Spark micro-batch, partitioned by day.
--    Latency and scores are stored as fixed-bucket histograms (not per-batch percentiles),
--    so the dashboard can merge batches into exact p50/p95/p99 and rebuild the confusion
--    matrix for any cutoff on the slider.
CREATE TABLE IF NOT EXISTS fraud.pipeline_metrics (
    day               date,
    batch_end         timestamp,
    batch_id          bigint,
    rows              int,
    produced_rate     double,          -- msgs/sec the producer sent (from the producer stamps)
    processed_rate    double,          -- rows/sec Spark scored
    written_rate      double,          -- rows/sec confirmed written to Cassandra
    latency_hist_ms   list<int>,       -- counts per latency bucket, producer stamp -> Cassandra write
    score_hist_fraud  list<int>,       -- counts of true frauds per score bucket (0.00-0.05, 0.05-0.10, ...)
    score_hist_legit  list<int>,       -- counts of legit txns per score bucket
    tp int, fp int, fn int, tn int,    -- at the model's default cutoff
    fraud_amt_caught  double,          -- $ of true frauds flagged
    fraud_amt_missed  double,          -- $ of true frauds let through
    max_offsets       map<int, bigint>, -- Kafka partition -> last offset processed (dashboard: backlog)
    model_version     text,
    trigger           text,             -- micro-batch interval, shown beside the latency numbers
    PRIMARY KEY ((day), batch_end, batch_id)
) WITH CLUSTERING ORDER BY (batch_end DESC, batch_id ASC);
