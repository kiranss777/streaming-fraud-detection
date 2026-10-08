# Streaming Fraud Detection

Real-time credit card fraud detection pipeline.

- **Streaming:** Kafka → Spark Structured Streaming → Cassandra (Docker on a single EC2 host)
- **Batch ETL:** AWS Glue (S3 raw CSV → feature Parquet)
- **Model:** XGBoost trained on AWS SageMaker, scored in-stream by Spark
- **Dashboard:** Streamlit

Dataset: [Credit Card Transactions Fraud Detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection) (~1.85M synthetic transactions, Sparkov generator).

> Work in progress.
