gsutil cp feature_eng.py gs://cis-4130-bwc/scripts/

gcloud dataproc jobs submit pyspark gs://cis-4130-bwc/scripts/feature_eng.py \
    --cluster=cluster-f6d4 \
    --region=us-central1 \
    --properties="spark.executor.memory=4g,spark.driver.memory=4g,spark.sql.shuffle.partitions=200" \
    -- \
    --base-path=gs://cis-4130-bwc/cleaned \
    --output-path=gs://cis-4130-bwc/trusted