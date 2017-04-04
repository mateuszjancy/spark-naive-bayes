name := "spark-naive-bayes"

version := "1.0"

scalaVersion := "2.11.8"

libraryDependencies ++= {
  val sparkV = "2.1.0"
  Seq(
    "org.apache.spark" %% "spark-core" % sparkV,
    "org.apache.spark" %% "spark-sql" % sparkV,
    "org.apache.spark" %% "spark-mllib" % sparkV
  )
}