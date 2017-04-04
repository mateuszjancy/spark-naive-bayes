package naivebayes

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

object WatherScript extends App {

  val conf = new SparkConf().setAppName("Simple Application")
  val sc = new SparkContext(conf)

  val spark = SparkSession
    .builder()
    .appName("Spark SQL basic example")
    .config("spark.some.config.option", "some-value")
    .getOrCreate()

  // For implicit conversions like converting RDDs to DataFrames
  import spark.implicits._

  val watherRaw: RDD[String] = sc.textFile("/Users/mateusz/Workspace/mllib/spark-naive-bayes/src/main/resources/wather-nums.csv")

  val dataRaw = watherRaw.map(_.split(";")).map { csv =>
    val label = csv.last.toDouble
    val point = csv.init.map(_.toDouble)
    (label, point)
  }

  val data: Dataset[LabeledPoint] = dataRaw
    .map { case (label, point) =>
      LabeledPoint(label, Vectors.dense(point))
    }.toDS()

  val Array(training: Dataset[LabeledPoint], test: Dataset[LabeledPoint]) = data.randomSplit(Array(0.7, 0.3), seed = 1234L)

  val model = new NaiveBayes()
    .setModelType("multinomial")
    .fit(training)

  val predictions = model.transform(test)
  predictions.show()

  val evaluator = new MulticlassClassificationEvaluator()
    .setLabelCol("label")
    .setPredictionCol("prediction")
    .setMetricName("accuracy")
  val accuracy = evaluator.evaluate(predictions)

  println("Test set accuracy = " + accuracy)
}
