gsutil cp yelp_pipeline.py gs://cis-4130-bwc/scripts/

gcloud dataproc jobs submit pyspark \
    gs://cis-4130-bwc/scripts/yelp_pipeline.py \
    --region us-central1 \
    --cluster cluster-f6d4 \
    --properties spark.executor.memory=4g,spark.driver.memory=4g,spark.sql.shuffle.partitions=200 \
    -- \
    --base-path gs://cis-4130-bwc \
    --output-path gs://cis-4130-bwc/predictions \
    --model-path gs://cis-4130-bwc/models \
    --results-path gs://cis-4130-bwc/results
