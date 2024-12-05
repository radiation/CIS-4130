import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Data Visualization") \
    .getOrCreate()

# Define paths
BASE_PATH = "gs://cis-4130-bwc"
MODEL_PATH = f"{BASE_PATH}/models"
PREDICTIONS_PATH = f"{BASE_PATH}/predictions"

# Load the entire pipeline model
pipeline_model = PipelineModel.load(MODEL_PATH)

# Extract the logistic regression model from the pipeline
logistic_model = next(stage for stage in pipeline_model.stages if stage.__class__.__name__ == "LogisticRegressionModel")

# Load prediction data
predictions = spark.read.parquet(PREDICTIONS_PATH)

# Extract feature importance (coefficients)
assembler = next(stage for stage in pipeline_model.stages if stage.__class__.__name__ == "VectorAssembler")
features = assembler.getInputCols()
coefficients = logistic_model.coefficients.toArray()
feature_importance = list(zip(features, coefficients))

# Convert feature importance to a Pandas DataFrame for visualization
feature_importance_df = pd.DataFrame(feature_importance, columns=["Feature", "Coefficient"])
feature_importance_df["AbsCoefficient"] = feature_importance_df["Coefficient"].abs()
feature_importance_df = feature_importance_df.sort_values(by="AbsCoefficient", ascending=False)

# Visualization 1: Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance_df, x="AbsCoefficient", y="Feature")
plt.title("Feature Importance")
plt.xlabel("Coefficient Magnitude (Absolute)")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("feature_importance.png")

# Visualization 2: Distribution of Predictions
predictions_pd = predictions.select("probability", "label", "prediction").toPandas()
predictions_pd["probability"] = predictions_pd["probability"].apply(lambda x: x[1])  # Extract probability for class 1
plt.figure(figsize=(10, 6))
sns.histplot(data=predictions_pd, x="probability", bins=20, hue="label", kde=True)
plt.title("Distribution of Predicted Probabilities")
plt.xlabel("Probability")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("predicted_probabilities.png")

# Visualization 3: Confusion Matrix
conf_matrix = confusion_matrix(predictions_pd["label"], predictions_pd["prediction"])
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["Not High Star", "High Star"])
disp.plot(cmap="viridis")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")

# Visualization 4: ROC Curve
fpr, tpr, thresholds = roc_curve(predictions_pd["label"], predictions_pd["probability"])
roc_auc = roc_auc_score(predictions_pd["label"], predictions_pd["probability"])
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png")

# Upload visualizations to GCS
os.system(f"gsutil cp feature_importance.png {BASE_PATH}/visualizations/")
os.system(f"gsutil cp predicted_probabilities.png {BASE_PATH}/visualizations/")
os.system(f"gsutil cp confusion_matrix.png {BASE_PATH}/visualizations/")
os.system(f"gsutil cp roc_curve.png {BASE_PATH}/visualizations/")

print("Visualizations saved and uploaded to GCS!")
