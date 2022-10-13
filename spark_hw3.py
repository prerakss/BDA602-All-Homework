from pyspark.sql import SparkSession
from pyspark.sql.functions import col, datediff, lit, to_date

appName = "PySpark Example - MariaDB Example"
master = "local"
# Create Spark session
spark = (
    SparkSession.builder.appName(appName)
    .master(master)
    .config("spark.jars", "mariadb-java-client-3.0.8.jar")
    .enableHiveSupport()
    .getOrCreate()
)


sql = "select batter, atBat, Hit, game_id from baseball.batter_counts"
sql2 = "select local_date, game_id from baseball.game"

database = "baseball"
user = "root"
server = "localhost"
port = 3306
jdbc_url = f"jdbc:mysql://{server}:{port}/{database}?permitMysqlScheme"
jdbc_driver = "org.mariadb.jdbc.Driver"

df = (
    spark.read.format("jdbc")
    .option("url", jdbc_url)
    .option("query", sql)
    .option("user", user)
    .option("password", "root")
    .option("driver", jdbc_driver)
    .load()
)

df2 = (
    spark.read.format("jdbc")
    .option("url", jdbc_url)
    .option("query", sql2)
    .option("user", user)
    .option("password", "root")
    .option("driver", jdbc_driver)
    .load()
)


temp = df.agg({"local_date": "min"}).collect()[0]

df = df.withColumn("min_local_date", lit(temp["min(local_date)"]))
df = df.withColumn(
    "days_diff", (datediff(to_date(df["local_date"]), to_date(df["min_local_date"])))
)

df_batter = df2.select(col("game_id").alias("batter_game_id"), "batter", "atBat", "Hit")


df_game = df.select("game_id", "days_diff")
df_batter_game = df_batter.join(
    df_game, df_game.game_id == df_batter.batter_game_id, "left"
)

df_batter_game_grouped = df_batter_game.groupBy(
    "batter", "batter_game_id", "days_diff"
).sum("atBat", "Hit")
df_batter_game_grouped = df_batter_game_grouped.select(
    "batter",
    "batter_game_id",
    "days_diff",
    col("sum(atBat)").alias("sum_atBat"),
    col("sum(Hit)").alias("sum_Hit"),
)
df_batter_game_grouped_2 = df_batter_game_grouped


df_batter_game_grouped.createOrReplaceTempView("temp_table")
