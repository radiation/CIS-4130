from pyspark.sql import SparkSession
import pyspark.sql.functions as F

spark = SparkSession.builder \
    .appName("Feature Engineering") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.cores", "2") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()

# Initialize Spark session
def initialize_spark(app_name="FeatureEngineering"):
    return SparkSession.builder.appName(app_name).getOrCreate()

# Load datasets
def load_datasets(spark, base_path):
    reviews = spark.read.parquet(f"{base_path}/yelp_academic_dataset_review.parquet")
    business = spark.read.parquet(f"{base_path}/yelp_academic_dataset_business.parquet")
    users = spark.read.parquet(f"{base_path}/yelp_academic_dataset_user.parquet")
    checkins = spark.read.parquet(f"{base_path}/yelp_academic_dataset_checkin.parquet")
    tips = spark.read.parquet(f"{base_path}/yelp_academic_dataset_tip.parquet")
    return reviews, business, users, checkins, tips

# Feature engineering function
def feature_engineering(reviews, business, users):
    # Alias datasets for clearer joins
    reviews = reviews.alias("r")
    business = business.alias("b")
    users = users.alias("u")

    # Join datasets
    enriched_data = reviews.join(
        business,
        reviews["business_id"] == business["business_id"],
        "inner"
    ).select(
        F.col("r.*"),
        F.col("b.stars").alias("business_stars"),
        F.col("b.is_open"),
        F.col("b.latitude"),
        F.col("b.longitude"),
        F.col("b.review_count").alias("business_review_count")
    ).join(
        users,
        reviews["user_id"] == users["user_id"],
        "inner"
    ).select(
        F.col("r.*"),
        F.col("business_stars"),
        F.col("is_open"),
        F.col("latitude"),
        F.col("longitude"),
        F.col("business_review_count"),
        F.col("u.average_stars").alias("user_average_stars"),
        F.col("u.review_count").alias("user_review_count"),
        F.col("u.fans"),
        F.col("u.elite"),
        F.col("u.cool").alias("user_cool"),
        F.col("u.funny").alias("user_funny"),
        F.col("u.useful").alias("user_useful")
    )

    # Feature engineering
    enriched_data = enriched_data.withColumn(
        "is_high_star", F.when(F.col("stars") >= 4, 1).otherwise(0)  # Binary label for good reviews
    )

    # Calculate max values for normalization
    max_business_review_count = enriched_data.select(F.max("business_review_count").alias("max_business_review_count")).collect()[0]["max_business_review_count"]
    max_user_review_count = enriched_data.select(F.max("user_review_count").alias("max_user_review_count")).collect()[0]["max_user_review_count"]

    # Normalize review count
    enriched_data = enriched_data.withColumn(
        "normalized_business_review_count",
        F.col("business_review_count") / F.lit(max_business_review_count)
    )

    enriched_data = enriched_data.withColumn(
        "normalized_user_review_count",
        F.col("user_review_count") / F.lit(max_user_review_count)
    )

    # Add location clustering features
    enriched_data = enriched_data.withColumn(
        "rounded_latitude", F.round(F.col("latitude"), 2)
    ).withColumn(
        "rounded_longitude", F.round(F.col("longitude"), 2)
    )

    # Drop columns not needed for modeling
    enriched_data = enriched_data.drop("review_id", "text", "date", "elite", "user_id", "business_id")

    return enriched_data

# Save transformed data
def save_to_parquet(df, output_path):
    df.write.mode("overwrite").parquet(output_path)

if __name__ == "__main__":
    base_path = "gs://cis-4130-bwc/cleaned"
    output_path = "gs://cis-4130-bwc/trusted"

    spark = initialize_spark()
    reviews, business, users, checkins, tips = load_datasets(spark, base_path)

    try:
        # Perform feature engineering
        transformed_data = feature_engineering(reviews, business, users)

        # Save the transformed dataset
        save_to_parquet(transformed_data, output_path)

        print(f"Feature engineering complete. Transformed data saved to {output_path}.")
    finally:
        spark.stop()
