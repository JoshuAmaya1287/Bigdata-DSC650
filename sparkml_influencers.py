
import happybase
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def main():
    spark = (
        SparkSession.builder
        .appName("Influencer_to_HBase")
        .enableHiveSupport()
        .getOrCreate()
    )

    feature_cols = ["posts", "followers", "avg_likes", "eng_rate_60_day"]
    label_col = "influence_score"

    influencers_df = spark.sql("SELECT * FROM influencers_2026")

    # Cast string columns from OpenCSVSerde to numeric types and drop nulls
    casted_df = influencers_df
    for col_name in feature_cols + [label_col]:
        casted_df = casted_df.withColumn(col_name, F.col(col_name).cast("double"))

    clean_df = casted_df.dropna(subset=feature_cols + [label_col])

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    assembled_df = assembler.transform(clean_df).select("features", label_col)

    train_df, test_df = assembled_df.randomSplit([0.8, 0.2], seed=42)

    lr = LinearRegression(featuresCol="features", labelCol=label_col)
    model = lr.fit(train_df)
    predictions = model.transform(test_df)

    rmse_evaluator = RegressionEvaluator(
        labelCol=label_col, predictionCol="prediction", metricName="rmse"
    )
    r2_evaluator = RegressionEvaluator(
        labelCol=label_col, predictionCol="prediction", metricName="r2"
    )

    rmse = rmse_evaluator.evaluate(predictions)
    r2 = r2_evaluator.evaluate(predictions)

    print(f"RMSE: {rmse}")
    print(f"R2: {r2}")

    # Write metrics to HBase table 'influencers_table_2026' under column family 'cf'
    connection = happybase.Connection("localhost")
    connection.open()
    try:
        table = connection.table("influencers_table_2026")
        table.put(
            b"influencer_metric_1",
            {
                b"cf:rmse": str(rmse).encode("utf-8"),
                b"cf:r2": str(r2).encode("utf-8"),
            },
        )
    finally:
        connection.close()

    spark.stop()


if __name__ == "__main__":
    main()
