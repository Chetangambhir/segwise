from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count, lit
from pyspark.sql.types import IntegerType, DoubleType
import itertools
import os

os.environ['PYSPARK_SUBMIT_ARGS'] = '--master local[3] pyspark-shell'

spark = SparkSession.builder.appName("PlayStoreAppAnalysis").getOrCreate()

df = spark.read.csv("D:/SegwiseAssignment/google-play-dataset-by-tapivedotcom.csv", header=True, inferSchema=True)

columns_to_analyze = [
    "free", "genre", "minInstalls", "rating", "ratingsTotal", 
    "releasedYear", "contentRating", "adSupported", "inAppProductPrice"
]

df = df.select(*columns_to_analyze)

df = df.withColumn("free", col("free").cast(IntegerType())) \
       .withColumn("minInstalls", col("minInstalls").cast(IntegerType())) \
       .withColumn("rating", col("rating").cast(DoubleType())) \
       .withColumn("ratingsTotal", col("ratingsTotal").cast(IntegerType())) \
       .withColumn("releasedYear", col("releasedYear").cast(IntegerType())) \
       .withColumn("adSupported", col("adSupported").cast(IntegerType())) \
       .withColumn("inAppProductPrice", when(col("inAppProductPrice").isNull(), 0).otherwise(col("inAppProductPrice").cast(DoubleType())))

bins = {
    "minInstalls": [0, 1000, 10000, 100000, 1000000, 10000000],
    "rating": [0, 1, 2, 3, 4, 5],
    "ratingsTotal": [0, 100, 1000, 10000, 100000, 1000000],
    "inAppProductPrice": [0, 1, 5, 10, 20, 50, 100]
}

for field, bin_ranges in bins.items():
    for i in range(len(bin_ranges) - 1):
        df = df.withColumn(
            field,
            when(
                (col(field) >= bin_ranges[i]) & (col(field) < bin_ranges[i + 1]),
                lit(f"{bin_ranges[i]}-{bin_ranges[i + 1]}")
            ).otherwise(col(field))
        )

combinations = []
for i in range(1, len(columns_to_analyze) + 1):
    combinations.extend(itertools.combinations(columns_to_analyze, i))

output_df = None

for combo in combinations:
    agg_df = df.groupBy(*combo).agg(count(lit(1)).alias("count"))
    agg_df = agg_df.withColumn("combination", lit("; ".join(combo)))
    agg_df = agg_df.select("combination", "count")

    total_count = agg_df.groupBy().sum("count").collect()[0][0]
    agg_df = agg_df.filter(col("count") / total_count >= 0.02)

    if output_df is None:
        output_df = agg_df
    else:
        output_df = output_df.union(agg_df)

output_df.coalesce(1).write.csv("output_file_path.csv", mode="overwrite", header=True)
