import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, length, min as spark_min, max as spark_max, mean as spark_mean, stddev as spark_stddev

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

# Initialize Spark session
def initialize_spark():
    spark = SparkSession.builder \
        .appName("Yelp EDA and Data Cleaning") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
        .config("spark.hadoop.google.cloud.auth.service.account.enable", "true") \
        .config("spark.hadoop.google.cloud.auth.service.account.json.keyfile", "/home/bryan_choate/gcp-key.json") \
        .config("spark.jars", "/opt/spark/jars/gcs-connector-hadoop3-latest.jar") \
        .config("spark.hadoop.fs.gs.outputstream.buffer.size", "4194304") \
        .config("spark.hadoop.fs.gs.outputstream.upload.chunk.size", "2097152") \
        .config("spark.hadoop.fs.gs.http.max.retry", "10") \
        .config("spark.hadoop.fs.gs.http.connect-timeout", "60000") \
        .config("spark.hadoop.fs.gs.http.read-timeout", "60000") \
        .config("spark.executor.memory", "8g") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark

def load_data(spark, file_path):
    df = spark.read.json(file_path)
    logger.info(f"Loaded data from {file_path}")
    return df

def perform_eda(df, output_dir, file_name):
    # Number of observations and columns
    num_records = df.count()
    columns = df.columns
    logger.info(f"Number of records: {num_records}")
    logger.info(f"Columns: {columns}")

    # Count missing values per column
    null_counts = df.select([(sum(col(c).isNull().cast("int")).alias(c)) for c in df.columns])
    logger.info("Missing values per column:")
    null_counts.show()

    # Basic statistics for numeric columns
    numeric_cols = [c for c, dtype in df.dtypes if dtype in ['int', 'bigint', 'float', 'double']]
    if numeric_cols:
        stats = df.select([spark_min(c).alias(f"{c}_min") for c in numeric_cols] +
                          [spark_max(c).alias(f"{c}_max") for c in numeric_cols] +
                          [spark_mean(c).alias(f"{c}_mean") for c in numeric_cols] +
                          [spark_stddev(c).alias(f"{c}_stddev") for c in numeric_cols])
        logger.info("Descriptive statistics for numeric columns:")
        stats.show()

        # Generate histograms for numeric columns
        for col_name in numeric_cols:
            data = pd.DataFrame(df.select(col_name).dropna().collect(), columns=[col_name])
            if not data.empty:
                plt.figure(figsize=(10, 6))
                sns.histplot(data[col_name], kde=True)
                plt.title(f"Distribution of {col_name}")
                plt.xlabel(col_name)
                plt.ylabel("Frequency")
                plt.savefig(os.path.join(output_dir, f"{file_name}_{col_name}_distribution.png"))
                plt.clf()
                plt.close()  # Close the plot to free memory

    # Min and max date if date columns exist
    if 'date' in columns:
        date_stats = df.agg(spark_min("date").alias("min_date"), spark_max("date").alias("max_date"))
        logger.info("Date range:")
        date_stats.show()

    # Text data analysis
    if "text" in columns:
        df = df.withColumn("text_length", length("text"))
        text_length_stats = df.select(spark_min("text_length").alias("min_length"),
                                      spark_max("text_length").alias("max_length"),
                                      spark_mean("text_length").alias("mean_length"),
                                      spark_stddev("text_length").alias("stddev_length"))
        logger.info("Text length statistics:")
        text_length_stats.show()

        text_length_data = pd.DataFrame(df.select("text_length").dropna().collect(), columns=["text_length"])
        if not text_length_data.empty:
            plt.figure(figsize=(10, 6))
            sns.histplot(text_length_data["text_length"], bins=30, kde=True)
            plt.title("Distribution of Text Lengths")
            plt.xlabel("Text Length")
            plt.ylabel("Frequency")
            plt.savefig(os.path.join(output_dir, f"{file_name}_text_length_distribution.png"))
            plt.clf()
            plt.close()

    return df

def clean_data(df):
    # Log initial row count
    initial_row_count = df.count()
    logger.info(f"Initial row count: {initial_row_count}")

    # Drop columns with >20% missing values
    threshold = 0.2 * initial_row_count
    columns_to_drop = [c for c in df.columns if df.filter(df[c].isNull()).count() > threshold]
    logger.info(f"Dropping columns due to missing data: {columns_to_drop}")
    df = df.drop(*columns_to_drop)

    # Drop rows with missing values
    df = df.dropna()
    cleaned_row_count = df.count()
    logger.info(f"Row count after cleaning: {cleaned_row_count}")
    logger.info(f"Rows dropped: {initial_row_count - cleaned_row_count}")

    return df

def save_to_gcs(df, output_path):
    df.write.mode("overwrite").parquet(output_path)
    logger.info(f"Saved cleaned data to {output_path}")

# Main script
if __name__ == "__main__":
    spark = initialize_spark()
    output_dir = "/tmp"

    files = [
        "yelp_academic_dataset_business.json",
        "yelp_academic_dataset_checkin.json",
        "yelp_academic_dataset_review.json",
        "yelp_academic_dataset_tip.json",
        "yelp_academic_dataset_user.json"
    ]

    landing_path = "gs://cis-4130-bwc/landing/"
    cleaned_base_path = "gs://cis-4130-bwc/cleaned/"

    for file in files:
        input_path = os.path.join(landing_path, file)
        output_path = os.path.join(cleaned_base_path, file.replace(".json", ".parquet"))

        df = load_data(spark, input_path)
        df = perform_eda(df, output_dir, file.replace(".json", ""))
        cleaned_df = clean_data(df)
        save_to_gcs(cleaned_df, output_path)

    spark.stop()
