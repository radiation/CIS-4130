from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
import json

# Initialize Spark
spark = SparkSession.builder.appName("Data Exploration").getOrCreate()

# Paths to Parquet files
parquet_paths = {
    "business": "gs://cis-4130-bwc/cleaned/yelp_academic_dataset_business.parquet",
    "review": "gs://cis-4130-bwc/cleaned/yelp_academic_dataset_review.parquet",
    "user": "gs://cis-4130-bwc/cleaned/yelp_academic_dataset_user.parquet",
    "checkin": "gs://cis-4130-bwc/cleaned/yelp_academic_dataset_checkin.parquet",
    "tip": "gs://cis-4130-bwc/cleaned/yelp_academic_dataset_tip.parquet"
}

# Load datasets
dataframes = {name: spark.read.parquet(path) for name, path in parquet_paths.items()}

# Metadata storage
metadata = {}

for name, df in dataframes.items():
    print(f"Processing dataset: {name}")

    # Schema
    print(f"Schema for {name} dataset:")
    df.printSchema()
    metadata[name] = {"columns": [{"name": col_name, "type": dtype} for col_name, dtype in df.dtypes]}

    # Summary statistics
    print(f"Summary statistics for {name} dataset:")
    summary_stats = df.describe()
    summary_stats.show()

    # Save summary statistics to metadata
    summary_data = summary_stats.collect()
    metadata[name]["summary"] = {row["summary"]: row.asDict() for row in summary_data}

    # Missing values
    print(f"Missing values in {name} dataset:")
    missing_counts = df.select([(sum(col(c).isNull().cast("int")).alias(c)) for c in df.columns])
    missing_data = missing_counts.collect()[0].asDict()
    print(missing_data)
    metadata[name]["missing_counts"] = missing_data

    # Unique value counts
    print(f"Distinct values for each column in {name} dataset:")
    unique_counts = {}
    for col_name in df.columns:
        distinct_count = df.select(col_name).distinct().count()
        unique_counts[col_name] = distinct_count
        print(f"{col_name}: {distinct_count} distinct values")
    metadata[name]["unique_counts"] = unique_counts

    # Correlations
    numeric_cols = [col_name for col_name, dtype in df.dtypes if dtype in ["int", "double"]]
    if numeric_cols:
        print(f"Correlation matrix for numeric columns in {name} dataset:")
        assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features")
        assembled_df = assembler.transform(df.select(numeric_cols))
        correlation_matrix = Correlation.corr(assembled_df, "features").head()[0]
        metadata[name]["correlations"] = str(correlation_matrix)  # Convert to string for saving
        print(correlation_matrix)

# Save metadata to a JSON file
with open("metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print("Metadata saved to metadata.json")

