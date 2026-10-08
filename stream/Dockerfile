# Debian 12 (bookworm): ships Java 17, which Spark 3.5 supports.
FROM python:3.11-slim-bookworm

# Spark runs on the JVM; procps provides `ps`, which Spark's launch scripts call.
RUN apt-get update && apt-get install -y --no-install-recommends openjdk-17-jre-headless procps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kafka + Cassandra connectors for Spark, downloaded at build time so startup is fast and offline.
ENV SPARK_PACKAGES=org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.3,com.datastax.spark:spark-cassandra-connector_2.12:3.5.1
RUN python -c "import os; from pyspark.sql import SparkSession; \
SparkSession.builder.config('spark.jars.packages', os.environ['SPARK_PACKAGES']).getOrCreate().stop()"

COPY *.py ./
ENV PYTHONPATH=/app
CMD ["python", "-u", "job.py"]
