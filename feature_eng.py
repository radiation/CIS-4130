from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import matplotlib.pyplot as plt
import pandas as pd

# Initialize Spark session
def initialize_spark(app_name="FeatureEngineering"):
    return SparkSession.builder \
        .appName(app_name) \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.cores", "2") \
        .config("spark.sql.shuffle.partitions", "200") \
        .getOrCreate()

# Load datasets
def load_datasets(spark, base_path):
    reviews = spark.read.parquet(f"{base_path}/yelp_academic_dataset_review.parquet")
    business = spark.read.parquet(f"{base_path}/yelp_academic_dataset_business.parquet")
    users = spark.read.parquet(f"{base_path}/yelp_academic_dataset_user.parquet")
    return reviews, business, users

# Log transformations
def log_transformation(action, details):
    print(f"[INFO] {action}: {details}")

# Feature engineering function
def feature_engineering(reviews, business, users):
    log_transformation("JOIN", "Joining reviews with business and user data.")
    enriched_data = reviews.alias("r").join(
        business.alias("b"),
        F.col("r.business_id") == F.col("b.business_id"),
        "inner"
    ).join(
        users.alias("u"),
        F.col("r.user_id") == F.col("u.user_id"),
        "inner"
    ).select(
        F.col("r.review_id"),
        F.col("r.stars").alias("review_stars"),
        F.col("r.text"),
        F.col("r.date"),
        F.col("r.cool").alias("review_cool"),
        F.col("r.funny").alias("review_funny"),
        F.col("r.useful").alias("review_useful"),
        F.col("b.business_id"),
        F.col("b.stars").alias("business_stars"),
        F.col("b.is_open"),
        F.col("b.latitude"),
        F.col("b.longitude"),
        F.col("b.review_count").alias("business_review_count"),
        F.col("u.user_id"),
        F.col("u.average_stars").alias("user_average_stars"),
        F.col("u.review_count").alias("user_review_count"),
        F.col("u.fans"),
        F.col("u.cool").alias("user_cool"),
        F.col("u.funny").alias("user_funny"),
        F.col("u.useful").alias("user_useful")
    )

    log_transformation("CREATE COLUMN", "Adding binary column for high-star reviews (4+).")
    enriched_data = enriched_data.withColumn(
        "is_high_star", F.when(F.col("review_stars") >= 4, 1).otherwise(0)
    )

    log_transformation("NORMALIZE", "Normalizing business and user review counts.")
    max_business_review_count = enriched_data.agg(F.max("business_review_count")).first()[0]
    max_user_review_count = enriched_data.agg(F.max("user_review_count")).first()[0]

    enriched_data = enriched_data.withColumn(
        "normalized_business_review_count",
        F.col("business_review_count") / F.lit(max_business_review_count)
    ).withColumn(
        "normalized_user_review_count",
        F.col("user_review_count") / F.lit(max_user_review_count)
    )

    log_transformation("CREATE COLUMN", "Adding rounded latitude and longitude for clustering.")
    enriched_data = enriched_data.withColumn(
        "rounded_latitude", F.round(F.col("latitude"), 2)
    ).withColumn(
        "rounded_longitude", F.round(F.col("longitude"), 2)
    )

    log_transformation("CREATE COLUMN", "Adding review text length.")
    enriched_data = enriched_data.withColumn(
        "review_text_length", F.length(F.col("text"))
    )

    log_transformation("BUCKETIZE", "Bucketing user review counts.")
    enriched_data = enriched_data.withColumn(
        "user_review_bucket",
        F.when(F.col("user_review_count") < 10, "low").when(
            F.col("user_review_count") < 50, "medium").otherwise("high")
    )

    return enriched_data

# Save transformed data
def save_to_parquet(df, output_path):
    log_transformation("SAVE", f"Saving transformed data to {output_path}.")
    df.write.mode("overwrite").parquet(output_path)

# Generate and save visualizations
def generate_visualizations(df, output_path):
    log_transformation("VISUALIZATION", "Generating visualizations for key features.")
    pandas_df = df.select("review_text_length", "review_stars", "business_review_count", "user_review_count").toPandas()

    plt.figure(figsize=(10, 6))
    pandas_df["review_text_length"].plot.hist(bins=50, alpha=0.7, title="Review Text Length Distribution")
    plt.savefig(f"{output_path}/review_text_length_distribution.png")

    plt.figure(figsize=(10, 6))
    pandas_df["review_stars"].value_counts().plot.bar(title="Review Stars Distribution")
    plt.savefig(f"{output_path}/review_stars_distribution.png")

    plt.figure(figsize=(10, 6))
    pandas_df["business_review_count"].plot.hist(bins=50, alpha=0.7, title="Business Review Count Distribution")
    plt.savefig(f"{output_path}/business_review_count_distribution.png")

    plt.figure(figsize=(10, 6))
    pandas_df["user_review_count"].plot.hist(bins=50, alpha=0.7, title="User Review Count Distribution")
    plt.savefig(f"{output_path}/user_review_count_distribution.png")

    log_transformation("VISUALIZATION", f"Visualizations saved to {output_path}.")

if __name__ == "__main__":
    base_path = "gs://cis-4130-bwc/cleaned"
    output_path = "gs://cis-4130-bwc/trusted"
    viz_output_path = "/tmp/visualizations"

    spark = initialize_spark()
    reviews, business, users = load_datasets(spark, base_path)

    try:
        log_transformation("START", "Starting feature engineering process.")
        transformed_data = feature_engineering(reviews, business, users)
        save_to_parquet(transformed_data, output_path)
        generate_visualizations(transformed_data, viz_output_path)
        log_transformation("COMPLETE", "Feature engineering process completed.")
    finally:
        spark.stop()
