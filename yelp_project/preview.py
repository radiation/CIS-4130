from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("VerifyOutput").getOrCreate()

df = spark.read.parquet("gs://cis-4130-bwc/trusted/")
df.show(5)
df.printSchema()
