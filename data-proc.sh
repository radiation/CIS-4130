gcloud dataproc clusters create my-dataproc-cluster \
    --region=us-central1 \
    --num-workers=4 \
    --worker-machine-type=n1-standard-4 \
    --worker-boot-disk-size=100GB \
    --properties="spark:spark.executor.memory=4g,spark:spark.driver.memory=4g,spark:spark.sql.shuffle.partitions=200"
