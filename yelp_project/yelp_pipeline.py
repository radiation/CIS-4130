from google.cloud import storage
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
import argparse

# Function to write content to GCS
def write_to_gcs(bucket_name, destination_path, content):
    # Initialize GCS client
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_path)

    # Write content to GCS
    blob.upload_from_string(content)
    print(f"Results saved to gs://{bucket_name}/{destination_path}")

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--base-path", required=True, help="Path to the base folder with cleaned data")
parser.add_argument("--output-path", required=True, help="Path to save transformed data")
parser.add_argument("--model-path", required=True, help="Path to save the trained model")
parser.add_argument("--results-path", required=True, help="Path to save evaluation results")
args = parser.parse_args()

# Initialize Spark session
spark = SparkSession.builder.appName("YelpPipeline").getOrCreate()

# Validate schema & load data
spark.read.parquet("gs://cis-4130-bwc/trusted/").printSchema()
data = spark.read.parquet(f"{args.base_path}/trusted")

# Define numeric columns for feature engineering
numeric_cols = [
    "normalized_business_review_count",
    "normalized_user_review_count",
]

# Define feature assembler
vector_assembler = VectorAssembler(inputCols=numeric_cols, outputCol="numeric_features")

# Add MinMaxScaler for numeric features
scaler = MinMaxScaler(inputCol="numeric_features", outputCol="scaled_features")

# Prepare label column
data = data.withColumn("label", when(col("is_high_star") == 1, 1).otherwise(0))

# Split data into training and testing sets
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# Define a Logistic Regression model
lr = LogisticRegression(featuresCol="scaled_features", labelCol="label")

# Create a pipeline with all transformations and the model
pipeline = Pipeline(stages=[vector_assembler, scaler, lr])

# Fit the pipeline to training data
pipeline_model = pipeline.fit(train_data)

# Save the trained model
pipeline_model.write().overwrite().save(args.model_path)

# Make predictions on the test data
predictions = pipeline_model.transform(test_data)

# Evaluate the model
evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
roc_auc = evaluator.evaluate(predictions)

# Prepare evaluation results content
evaluation_results_content = f"ROC AUC: {roc_auc}\n"

# Save evaluation results to GCS
bucket_name = args.results_path.split("/")[2]  # Extract the bucket name from results_path
destination_path = "/".join(args.results_path.split("/")[3:] + ["evaluation_results.txt"])  # Path inside the bucket

write_to_gcs(bucket_name, destination_path, evaluation_results_content)

# Save transformed data
predictions_path = f"{args.output_path}/predictions"  # Change this to a dedicated predictions folder
predictions.write.mode("overwrite").parquet(predictions_path)

# Stop Spark session
spark.stop()
