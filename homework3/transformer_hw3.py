import sys

from pyspark import keyword_only
from pyspark.ml import Pipeline, Transformer
from pyspark.ml.param.shared import HasInputCols, HasOutputCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable

# from pyspark.sql.functions import col, concat, lit, split, when
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as func
from pyspark.sql.functions import col, datediff, lit, to_date


class Transform(
    Transformer,
    HasInputCols,
    HasOutputCol,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    @keyword_only
    def __init__(self, inputCols=None, outputCol=None):
        super(Transform, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)
        return

    @keyword_only
    def setParams(self, inputCols=None, outputCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        input_cols = self.getInputCols()
        output_col = self.getOutputCol()

        print(input_cols, output_col)

        window = (
            Window.partitionBy("batter").orderBy("days_diff").rangeBetween(-100, -1)
        )

        df_rolling_avg = dataset.withColumn(
            "rolling_avg", func.sum("Hit").over(window) / func.sum("atBat").over(window)
        ).orderBy("batter", "batter_game_id")
        df_rolling_avg = df_rolling_avg.select(
            "batter", col("batter_game_id").alias("game_id"), "rolling_avg"
        )

        return df_rolling_avg.show()


def main():
    appName = "Homework3 Spark"
    master = "local"

    spark = (
        SparkSession.builder.appName(appName)
        .master(master)
        .config("spark.jars", "mariadb-java-client-3.0.8.jar")
        .enableHiveSupport()
        .getOrCreate()
    )

    sql_game = "select local_date, game_id from baseball.game"
    sql_batter = "select batter, atBat, Hit, game_id from baseball.batter_counts"

    database = "baseball"
    user = "root"
    df_pass = "root"
    server = "localhost"
    port = 3306
    jdbc_url = f"jdbc:mysql://{server}:{port}/{database}?permitMysqlScheme"
    jdbc_driver = "org.mariadb.jdbc.Driver"

    df = (
        spark.read.format("jdbc")
        .option("url", jdbc_url)
        .option("query", sql_game)
        .option("user", user)
        .option("password", df_pass)
        .option("driver", jdbc_driver)
        .load()
    )

    df2 = (
        spark.read.format("jdbc")
        .option("url", jdbc_url)
        .option("query", sql_batter)
        .option("user", user)
        .option("password", df_pass)
        .option("driver", jdbc_driver)
        .load()
    )

    temp = df.agg({"local_date": "min"}).collect()[0]

    df = df.withColumn("min_local_date", lit(temp["min(local_date)"]))
    df = df.withColumn(
        "days_diff",
        (datediff(to_date(df["local_date"]), to_date(df["min_local_date"]))),
    )

    df_batter = df2.select(
        col("game_id").alias("batter_game_id"), "batter", "atBat", "Hit"
    )

    df_game = df.select("game_id", "days_diff")
    df_batter_game = df_batter.join(
        df_game, df_game.game_id == df_batter.batter_game_id, "left"
    )

    transform = Transform(inputCols=["atBat", "Hit"], outputCol="Rolling_Average")

    pipeline = Pipeline(stages=[transform])

    model = pipeline.fit(df_batter_game)
    model.transform(df_batter_game)


if __name__ == "__main__":
    sys.exit(main())
