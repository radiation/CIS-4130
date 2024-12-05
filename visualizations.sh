#!/bin/bash

# Define variables
SCRIPT_NAME="visualizations.py"
GCS_BUCKET="gs://cis-4130-bwc"
REGION="us-central1"
CLUSTER_NAME="cluster-f6d4"
SCRIPT_PATH="${GCS_BUCKET}/scripts/${SCRIPT_NAME}"
VISUALIZATIONS_PATH="${GCS_BUCKET}/visualizations"
MODEL_PATH="${GCS_BUCKET}/models"
PREDICTIONS_PATH="${GCS_BUCKET}/predictions"

# Step 1: Copy the script to GCS
echo "Uploading ${SCRIPT_NAME} to GCS..."
gsutil cp ${SCRIPT_NAME} ${SCRIPT_PATH}

# Step 2: Submit the Dataproc job
echo "Submitting the Dataproc job..."
gcloud dataproc jobs submit pyspark \
    ${SCRIPT_PATH} \
    --region ${REGION} \
    --cluster ${CLUSTER_NAME} \
    --properties spark.executor.memory=4g,spark.driver.memory=4g,spark.sql.shuffle.partitions=200 \
    -- \
    --predictions-path ${PREDICTIONS_PATH} \
    --model-path ${MODEL_PATH} \
    --visualizations-path ${VISUALIZATIONS_PATH}

echo "Job submitted successfully."
