
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, length, min as spark_min, max as spark_max, mean as spark_mean, stddev as spark_stddev

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
    return spark

def load_data(spark, file_path):
    df = spark.read.json(file_path)
    return df

def perform_eda(df, output_dir, file_name):
    print(f"Performing EDA on {file_name}...")
    num_records = df.count()
    columns = df.columns
    print(f"Number of records: {num_records}")
    print(f"Columns: {columns}")

    null_counts = df.select([(sum(col(c).isNull().cast("int")).alias(c)) for c in df.columns])
    print("Missing values per column:")
    null_counts.show()

    numeric_cols = [c for c, dtype in df.dtypes if dtype in ['int', 'bigint', 'float', 'double']]
    print("\nNumeric columns:", numeric_cols, "\n")

    if numeric_cols:
        stats = df.select([spark_min(c).alias(f"{c}_min") for c in numeric_cols] +
                          [spark_max(c).alias(f"{c}_max") for c in numeric_cols] +
                          [spark_mean(c).alias(f"{c}_mean") for c in numeric_cols] +
                          [spark_stddev(c).alias(f"{c}_stddev") for c in numeric_cols])
        print("\nDescriptive statistics for numeric columns:\n")
        stats.show()

        for col_name in numeric_cols:
            print(f"\nGenerating distribution plot for {col_name}...\n")
            data = pd.DataFrame(df.select(col_name).dropna().collect(), columns=[col_name])
            if not data.empty:
                plt.figure(figsize=(10, 6))
                sns.histplot(data[col_name], kde=True)
                plt.title(f"Distribution of {col_name}")
                plt.xlabel(col_name)
                plt.ylabel("Frequency")
                plt.savefig(os.path.join(output_dir, f"{file_name}_{col_name}_distribution.png"))
                plt.close()

    print("\nEDA Checkpoint 1\n")

    if 'date' in columns:
        date_stats = df.agg(spark_min("date").alias("min_date"), spark_max("date").alias("max_date"))
        print("\nDate range:\n")
        date_stats.show()

    if "text" in columns:
        df = df.withColumn("text_length", length("text"))
        text_length_stats = df.select(spark_min("text_length").alias("min_length"),
                                      spark_max("text_length").alias("max_length"),
                                      spark_mean("text_length").alias("mean_length"),
                                      spark_stddev("text_length").alias("stddev_length"))
        print("\nText length statistics:\n")
        text_length_stats.show()

        text_length_data = pd.DataFrame(df.select("text_length").dropna().collect(), columns=["text_length"])
        if not text_length_data.empty:
            plt.figure(figsize=(10, 6))
            sns.histplot(text_length_data["text_length"], bins=30, kde=True)
            plt.title("Distribution of Text Lengths")
            plt.xlabel("Text Length")
            plt.ylabel("Frequency")
            print("\nSaving", os.path.join(output_dir, f"{file_name}_text_length_distribution.png\n"))
            plt.savefig(os.path.join(output_dir, f"{file_name}_text_length_distribution.png"))
            plt.close("all")

    print("\nEDA Checkpoint 2\n")

    return df

def clean_data(df):
    print("\nCleaning data...\n")
    for col_name in df.columns:
        print(f"\nCleaning column: {col_name}\n")
        df = df.withColumnRenamed(col_name, col_name.replace(" ", "_"))

    drop_columns = []
    if drop_columns:
        df = df.drop(*drop_columns)

    df = df.dropna()
    return df

def save_to_gcs(df, output_path):
    """
    Save the DataFrame to Google Cloud Storage in Parquet format.
    :param df: DataFrame to save
    :param output_path: Full GCS path where the file should be saved
    """
    print(f"\nSaving cleaned data to {output_path}...\n")
    if not output_path.startswith("gs://"):
        raise ValueError("The output_path must be a GCS path (e.g., gs://bucket-name/...")

    df.write.mode("overwrite").parquet(output_path)
    print(f"\nSaved cleaned data to {output_path}.\n")

if __name__ == "__main__":
    # Parse input argument
    if len(sys.argv) != 2:
        print("Usage: python data_cleaning.py <filename>")
        sys.exit(1)

    file_path = sys.argv[1]
    file_name = os.path.basename(file_path)

    # Define paths
    landing_path = "gs://cis-4130-bwc/landing/"
    cleaned_base_path = "gs://cis-4130-bwc/cleaned/"
    output_dir = "/tmp"

    # Initialize Spark
    spark = initialize_spark()

    try:
        # Sizes of large files may require sampling for EDA; cap at 500MB
        max_sample_size: int = 2 ** 29
        file_size: int = os.path.getsize(file_path)
        is_large_file: bool = file_size > max_sample_size
        sample_ratio: float = max_sample_size / file_size

        print(f"\n\nFile size: {file_size} bytes")
        print(f"Sample ratio: {sample_ratio}")
        print(f"Is large file: {is_large_file}\n\n")

        # Construct paths
        input_path = os.path.join(landing_path, file_name)
        output_path = os.path.join(cleaned_base_path, file_name.replace(".json", ".parquet"))
        print(f"\nCleaning {input_path} and saving to {output_path}...\n")

        # Load full dataset
        df = load_data(spark, input_path)

        # Perform EDA on the full dataset or a sample
        if is_large_file:
            print(f"\n{file_name} is a large file. Sampling {sample_ratio} data for EDA...\n")
            sampled_df = df.sample(fraction=sample_ratio)
            perform_eda(sampled_df, output_dir, file_name.replace(".json", ""))
        else:
            perform_eda(df, output_dir, file_name.replace(".json", ""))

        # Perform cleaning on the entire dataset
        cleaned_df = clean_data(df)

        # Save cleaned data with appropriate partitioning
        num_partitions = 200 if is_large_file else df.rdd.getNumPartitions()
        cleaned_df.repartition(num_partitions).write.mode("overwrite").parquet(output_path)
        print(f"\nSaved cleaned data to {output_path}.\n")

    finally:
        spark.stop()
