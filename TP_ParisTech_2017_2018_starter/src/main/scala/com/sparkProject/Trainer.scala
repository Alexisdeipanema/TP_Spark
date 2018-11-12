package com.sparkProject


import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator, RegressionEvaluator}


object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()


    /*******************************************************************************
      *
      *       TP 3
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    val data = spark
      .read
      .load("/home/alexis/TP_Spark/TP_ParisTech_2017_2018_starter/data_cleaned/prepared_trainingset/*.parquet")

    val Array(training_set, test) = data
      .randomSplit(Array[Double](0.9, 0.1), 18)

    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    val remover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("stop_words_free")

    val tf_maker = new CountVectorizer()
      .setInputCol("stop_words_free")
      .setOutputCol("tf_made")

    val idf_maker = new IDF()
      .setInputCol("tf_made")
      .setOutputCol("tf-idf_made")


    val country_conversion = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")

    val currency_conversion = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")

    val encoding_country = new OneHotEncoder()
      .setInputCol("country_indexed")
      .setOutputCol("country_onehot")

    val encoding_currency = new OneHotEncoder()
      .setInputCol("currency_indexed")
      .setOutputCol("currency_onehot")

    val assembler = new VectorAssembler()
      .setInputCols(Array("tf_made", "days_campaign", "hours_prepa", "goal", "country_onehot", "currency_onehot"))
      .setOutputCol("features")

    val regression_logistique = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(300)

    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover, tf_maker,idf_maker, country_conversion, currency_conversion,encoding_country , encoding_currency,assembler, regression_logistique))

    val Parametres_regression_logistique:Array[Double] = Array(10E-8,10E-6,10E-4,10E-2)

    val minDFs:Array[Double] = Array(55,75,95)

    val Grille_parametres= new ParamGridBuilder()
      .addGrid(regression_logistique.regParam, Parametres_regression_logistique)
      .addGrid(tf_maker.minDF,minDFs)
      .build()

    val evaluator=new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")



    val split_training = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(Grille_parametres)
      .setTrainRatio(0.7)

    val modele = split_training.fit(training_set)

    val prediction_data=modele.transform(test)

    val evaluation = evaluator.evaluate(prediction_data)
    println("F1 " + evaluation)

    modele.write.overwrite().save("/home/alexis/TP_Spark/TP_ParisTech_2017_2018_starter/pipeline")

  }
}
