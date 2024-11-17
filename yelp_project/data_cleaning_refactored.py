
import os
import signal
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from functools import reduce

def handler(signum, frame):
    raise TimeoutError("Plotting timed out!")

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
    return spark

def load_data(spark, file_path):
    df = spark.read.json(file_path)
    return df

# Exploratory Data Analysis
def perform_eda(df, output_dir, file_name):
    print(f"Performing EDA on {file_name}...")
    num_records = df.count()
    columns = df.columns
    print(f"Number of records: {num_records}")
    print(f"Columns: {columns}")

    null_counts = df.select([(F.sum(F.col(c).isNull().cast("int")).alias(c)) for c in df.columns])
    print("Missing values per column:")
    null_counts.show()

    numeric_cols = [c for c, dtype in df.dtypes if dtype in ['int', 'bigint', 'float', 'double']]
    print("\nNumeric columns:", numeric_cols, "\n")

    for col_name in numeric_cols:
        print(f"Generating distribution plot for {col_name}...")
        try:
            # Inspect data
            print(f"Checking data for column: {col_name}")
            sample_data = df.select(col_name).dropna().limit(10).collect()
            print(f"Sample data for {col_name}: {sample_data}")

            # Limit data points for plotting
            data = pd.DataFrame(df.select(col_name).dropna().limit(100000).collect(), columns=[col_name])
            if not data.empty:
                # Add timeout for plotting
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(60)

                # Plot
                plt.figure(figsize=(10, 6))
                sns.histplot(data[col_name], kde=True)
                plt.title(f"Distribution of {col_name}")
                plt.xlabel(col_name)
                plt.ylabel("Frequency")
                plt.savefig(os.path.join(output_dir, f"{file_name}_{col_name}_distribution.png"))
                plt.close()

                print(f"Finished saving plot for {col_name}")
        except TimeoutError:
            print(f"Timeout while processing {col_name}. Skipping this column.")
        except Exception as e:
            print(f"Error processing {col_name}: {e}")
        finally:
            signal.alarm(0)  # Reset the timeout

    print("\nEDA Checkpoint 1\n")

    if 'date' in columns:
        date_stats = df.agg(F.min("date").alias("min_date"), F.max("date").alias("max_date"))
        print("\nDate range:\n")
        date_stats.show()

    if "text" in columns:
        df = df.withColumn("text_length", F.length("text"))
        text_length_stats = df.select(F.min("text_length").alias("min_length"),
                                    F.max("text_length").alias("max_length"),
                                    F.mean("text_length").alias("mean_length"),
                                    F.stddev("text_length").alias("stddev_length"))
        print("\nText length statistics:\n")
        text_length_stats.show()

        text_length_data = pd.DataFrame(df.select("text_length").dropna().collect(), columns=["text_length"])
        if not text_length_data.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(text_length_data["text_length"], bins=30, kde=True, ax=ax)
            ax.set_title("Distribution of Text Lengths")
            ax.set_xlabel("Text Length")
            ax.set_ylabel("Frequency")
            output_path = os.path.join(output_dir, f"{file_name}_text_length_distribution.png")
            print("Saving", output_path)
            plt.savefig(output_path)
            plt.close(fig)

    print("\nEDA Checkpoint 2\n")

    return df

def filter_invalid_data(df):
    # Create conditions for invalid data
    filters = []

    # Ensure `stars` column values are between 1 and 5 (inclusive)
    if "stars" in df.columns:
        filters.append((F.col("stars") >= 1) & (F.col("stars") <= 5))

    # Ensure latitude and longitude are within valid ranges
    if "latitude" in df.columns:
        filters.append((F.col("latitude") >= -90) & (F.col("latitude") <= 90))
    if "longitude" in df.columns:
        filters.append((F.col("longitude") >= -180) & (F.col("longitude") <= 180))

    # Ensure review_count, cool, funny, useful, etc are non-negative
    numeric_cols_to_validate = [
        "review_count", "cool", "funny", "useful", "compliment_cool", "compliment_cute",
        "compliment_funny", "compliment_hot", "compliment_list", "compliment_more",
        "compliment_note", "compliment_photos", "compliment_plain", "compliment_profile",
        "compliment_writer", "fans"
    ]
    
    for col_name in numeric_cols_to_validate:
        if col_name in df.columns:
            filters.append(F.col(col_name) >= 0)

    # Combine all filters using reduce
    if filters:
        combined_filter = reduce(lambda a, b: a & b, filters)
        df = df.filter(combined_filter)

    return df

def clean_data(df):
    # Original row and column counts
    original_rows = df.count()
    original_columns = len(df.columns)

    # Remove columns with significant missing values
    threshold = 0.05
    column_missing_counts = df.select([(F.sum(F.col(c).isNull().cast("int")) / original_rows).alias(c) for c in df.columns])
    to_drop = [col_name for col_name, value in column_missing_counts.collect()[0].asDict().items() if value > threshold]
    df = df.drop(*to_drop)

    # Apply validation filters to remove invalid rows
    df = filter_invalid_data(df)

    # Drop rows with any nulls
    df = df.dropna()

    # Final row and column counts
    final_rows = df.count()
    final_columns = len(df.columns)

    # Summary report
    rows_dropped = original_rows - final_rows
    columns_dropped = original_columns - final_columns

    print(f"Cleaning Summary:\n"
          f"Original Rows: {original_rows}, Final Rows: {final_rows}, Rows Dropped: {rows_dropped}\n"
          f"Original Columns: {original_columns}, Final Columns: {final_columns}, Columns Dropped: {columns_dropped}")

    return df


# Save directly to Google Cloud Storage
def save_to_gcs(df, output_path):
    print(f"Saving cleaned data to {output_path}...")
    if not output_path.startswith("gs://"):
        raise ValueError("The output_path must be a GCS path (e.g., gs://bucket-name/...")

    df.write.mode("overwrite").parquet(output_path)
    print(f"Saved cleaned data to {output_path}.")

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

        print(f"\nFile size: {file_size} bytes")
        print(f"Sample ratio: {sample_ratio}")
        print(f"Is large file: {is_large_file}\n")

        # Construct paths
        input_path = os.path.join(landing_path, file_name)
        output_path = os.path.join(cleaned_base_path, file_name.replace(".json", ".parquet"))
        print(f"Cleaning {input_path} and saving to {output_path}...")

        # Load full dataset
        df = load_data(spark, input_path)

        # Perform EDA on the full dataset or a sample
        if is_large_file:
            print(f"{file_name} is a large file. Sampling {sample_ratio} data for EDA...")
            sampled_df = df.sample(fraction=sample_ratio)
            perform_eda(sampled_df, output_dir, file_name.replace(".json", ""))
        else:
            perform_eda(df, output_dir, file_name.replace(".json", ""))

        # Perform cleaning on the entire dataset
        cleaned_df = clean_data(df)

        # Save cleaned data with appropriate partitioning
        num_partitions = 200 if is_large_file else df.rdd.getNumPartitions()
        cleaned_df.repartition(num_partitions).write.mode("overwrite").parquet(output_path)
        print(f"Saved cleaned data to {output_path}.")

    finally:
        spark.stop()
