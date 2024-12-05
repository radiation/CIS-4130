from google.cloud import storage
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
import argparse

# Function to write content to GCS
def write_to_gcs(bucket_name, destination_path, content):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_path)
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

# Define a parameter grid for cross-validation
param_grid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.01, 0.1, 1.0]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .build()

# Define an evaluator
evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")

# Create a CrossValidator
cross_validator = CrossValidator(
    estimator=Pipeline(stages=[vector_assembler, scaler, lr]),
    estimatorParamMaps=param_grid,
    evaluator=evaluator,
    numFolds=5
)

# Fit the cross-validator to training data
cv_model = cross_validator.fit(train_data)

# Save the best model
cv_model.bestModel.write().overwrite().save(args.model_path)

# Make predictions on the test data
predictions = cv_model.bestModel.transform(test_data)

# Evaluate the model
roc_auc = evaluator.evaluate(predictions)

# Additional metrics
true_positive = predictions.filter((col("label") == 1) & (col("prediction") == 1)).count()
false_positive = predictions.filter((col("label") == 0) & (col("prediction") == 1)).count()
true_negative = predictions.filter((col("label") == 0) & (col("prediction") == 0)).count()
false_negative = predictions.filter((col("label") == 1) & (col("prediction") == 0)).count()

precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0
recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

# Prepare evaluation results content
evaluation_results_content = (
    f"ROC AUC: {roc_auc}\n"
    f"Precision: {precision}\n"
    f"Recall: {recall}\n"
    f"F1 Score: {f1_score}\n"
)

# Save evaluation results to GCS
bucket_name = args.results_path.split("/")[2]
destination_path = "/".join(args.results_path.split("/")[3:] + ["evaluation_results.txt"])

write_to_gcs(bucket_name, destination_path, evaluation_results_content)

# Save transformed data
predictions_path = f"{args.output_path}"
predictions.write.mode("overwrite").parquet(predictions_path)

# Stop Spark session
spark.stop()
