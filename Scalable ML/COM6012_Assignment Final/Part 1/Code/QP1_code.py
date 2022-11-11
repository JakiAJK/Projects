from pyspark.sql import SparkSession
import numpy as np
from pyspark.sql.types import StringType
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

from pyspark.ml.evaluation import BinaryClassificationEvaluator,MulticlassClassificationEvaluator
import matplotlib.pyplot as plt

from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, MultilayerPerceptronClassifier
import json
from pyspark.sql.functions import when
import time

spark = SparkSession.builder \
        .appName("Assignment_1") \
        .config("spark.local.dir","/fastdata/acp21jka/") \
        .getOrCreate()
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)
sc = spark.sparkContext
sc.setLogLevel("ERROR") # Only log ERRORs

#Load dataset and preprocessing
rawtrain = spark.read.csv("../Data/XOR_Arbiter_PUFs/5xor_128bit/train_5xor_128dim.csv")
rawtest = spark.read.csv("../Data/XOR_Arbiter_PUFs/5xor_128bit/test_5xor_128dim.csv")
rawtrain.cache()
ncolumns = len(rawtrain.columns)


# rename last column
rawtrain = rawtrain.withColumnRenamed('_c128', 'labels')
rawtest = rawtest.withColumnRenamed('_c128', 'labels')
colmn_names = [x.name for x in rawtrain.schema.fields if x.dataType == (StringType())]
# convert to double type
for c in colmn_names:
    rawtrain = rawtrain.withColumn(c, col(c).cast("double"))
    rawtest = rawtest.withColumn(c, col(c).cast("double"))

# replace values of -1 with 0 because classifiers dont work with negative labels
rawtrain = rawtrain.withColumn("labels", when(rawtrain.labels == -1, 0) \
      .otherwise(rawtrain.labels))
rawtest = rawtest.withColumn("labels", when(rawtest.labels == -1, 0) \
      .otherwise(rawtest.labels))

Data,_ = rawtrain.randomSplit([0.01, 0.99], 210116270) # select 1% of traindata
small_trData, small_testData = Data.randomSplit([0.8, 0.2], 210116270) # train and test from smaller data

print(f"There are {rawtrain.cache().count()} rows in the full training set, and {rawtest.cache().count()} in the full test set")

print(f"There are {small_trData.cache().count()} rows in the smaller training set, and {small_testData.cache().count()} in the smaller test set")

# We use BinaryClassificationEvaluator to compute Area Under Curves
evaluator = BinaryClassificationEvaluator\
      (labelCol="labels", rawPredictionCol="prediction")
# We use MulticlassClassificationEvaluator to compute Accuracy
evaluator_M = MulticlassClassificationEvaluator\
      (labelCol="labels", predictionCol="prediction", metricName="accuracy")

vecAssembler = VectorAssembler(inputCols = colmn_names[0:ncolumns-1], outputCol = 'features')

########################################################## 1 #################################################################
print("\n############################## RF ################################")
rf = RandomForestClassifier(labelCol="labels", featuresCol="features", seed=210116270)
#pipeline for model
stages_rf = [vecAssembler, rf]
pipeline_rf = Pipeline(stages=stages_rf)

#cross-val grid creation
paramGrid1 = ParamGridBuilder() \
    .addGrid(rf.maxDepth, [1, 5, 10]) \
    .addGrid(rf.maxBins, [10, 20, 50]) \
    .addGrid(rf.subsamplingRate, [0.1, 0.5, 0.9]) \
    .build()


crossval1 = CrossValidator(estimator=pipeline_rf,
                          estimatorParamMaps=paramGrid1,
                          evaluator=evaluator,
                          numFolds=5)
# fit and predict for best cross-val model
cvModel_rf = crossval1.fit(small_trData)
prediction = cvModel_rf.transform(small_testData)
accuracy = evaluator_M.evaluate(prediction)

best_RF_model= cvModel_rf.bestModel #best performing model
print("Accuracy for best rf model after hyperparameter tuning on small train data = %g " % accuracy)
paramDict1 = {param[0].name: param[1] for param in best_RF_model.stages[-1].extractParamMap().items()} # Dict having best param values
print("Best RF model params-")
print(json.dumps(paramDict1, indent = 4))


# Computation on full dataset
print("############################# Full Datasets on Random Forest #################################")
s1=time.time()
rf_final= RandomForestClassifier(labelCol="labels", featuresCol="features", seed=210116270, maxDepth=paramDict1['maxDepth'],
                                maxBins=paramDict1['maxBins'], subsamplingRate=paramDict1['subsamplingRate']) # retrieve values from cross-val above
stages_rf_final = [vecAssembler, rf_final]
pipeline_rf_final = Pipeline(stages=stages_rf_final)

RF_model = pipeline_rf_final.fit(rawtrain)
e1=time.time()
print("Train time for RF model", e1-s1)
preds1 = RF_model.transform(rawtest)
accuracy1 = evaluator_M.evaluate(preds1)

print("Test time for RF model", time.time()-e1)

print("Accuracy for best random forest model on whole dataset = %g " % accuracy1)
# Area under curve using Binary Evaluator
Area_under_ROC = evaluator.evaluate(preds1, {evaluator.metricName: "areaUnderROC"})
Area_under_PRC = evaluator.evaluate(preds1, {evaluator.metricName: "areaUnderPR"})
print("Area under Receiver Operating Characteristic Curve: {:.3f}".format(Area_under_ROC))
print("Area under Precision Recall Curve: {:.3f}".format(Area_under_PRC))
print("\n############################## LR ################################")
########################################################## 2 #################################################################

lr = LogisticRegression(featuresCol='features', labelCol='labels', family='binomial')

stages_lr = [vecAssembler, lr]
pipeline_lr = Pipeline(stages=stages_lr)


paramGrid2 = ParamGridBuilder() \
    .addGrid(lr.elasticNetParam, [0.0, 0.3, 0.5, 0.7, 1.0]) \
    .addGrid(lr.regParam, [0.001, 0.01, 0.1]) \
    .addGrid(lr.maxIter, [25, 50, 100]) \
    .build()



crossval2 = CrossValidator(estimator=pipeline_lr,
                          estimatorParamMaps=paramGrid2,
                          evaluator=evaluator,
                          numFolds=3)

cvModel_lr = crossval2.fit(small_trData)
prediction = cvModel_lr.transform(small_testData)
accuracy = evaluator_M.evaluate(prediction)
best_LR_model= cvModel_lr.bestModel

print("Accuracy for best lr model after hyperparameter tuning on small train data = %g " % accuracy)
paramDict2 = {param[0].name: param[1] for param in best_LR_model.stages[-1].extractParamMap().items()}
print("Best LR model params-")
print(json.dumps(paramDict2, indent = 4))


print("############################# Full Datasets on Logistic Regression #################################")
s2=time.time()
lr_final= LogisticRegression(featuresCol='features', labelCol='labels', family='binomial', elasticNetParam=paramDict2['elasticNetParam'],
                                regParam=paramDict2['regParam'], maxIter=paramDict2['maxIter'])
stages_lr_final = [vecAssembler, lr_final]
pipeline_lr_final = Pipeline(stages=stages_lr_final)

lr_model = pipeline_lr_final.fit(rawtrain)
e2=time.time()
print("Train time for LR model", e2-s2)
preds2 = lr_model.transform(rawtest)
accuracy2 = evaluator_M.evaluate(preds2)

print("Test time for LR model", time.time()-e2)

print("Accuracy for best logistic regression model on whole dataset = %g " % accuracy2)
Area_under_ROC = evaluator.evaluate(preds2, {evaluator.metricName: "areaUnderROC"})
Area_under_PRC = evaluator.evaluate(preds2, {evaluator.metricName: "areaUnderPR"})
print("Area under Receiver Operating Characteristic Curve: {:.3f}".format(Area_under_ROC))
print("Area under Precision Recall Curve: {:.3f}".format(Area_under_PRC))
print("\n############################## MPC ################################")


########################################################## 3 #################################################################



# The first element HAS to be equal to the number of input features
# Last element equals number of distinct labels
layers = [len(small_trData.columns)-1, 20, 2] 
mpc = MultilayerPerceptronClassifier(labelCol="labels", featuresCol="features", layers=layers, seed=210116270)

# Create the pipeline
stages3 = [vecAssembler, mpc]
pipeline_mpc = Pipeline(stages=stages3)

paramGrid3 = ParamGridBuilder() \
    .addGrid(mpc.stepSize, [0.1, 0.01, 0.001]) \
    .addGrid(mpc.blockSize, [32,64,128]) \
    .addGrid(mpc.maxIter, [10, 25, 100]) \
    .build()


# Make Crossvalidator object
crossval3 = CrossValidator(estimator=pipeline_mpc,
                          estimatorParamMaps=paramGrid3,
                          evaluator=evaluator,
                          numFolds=3)

cvModel_mpc = crossval3.fit(small_trData)
prediction = cvModel_mpc.transform(small_testData)
accuracy = evaluator_M.evaluate(prediction)
best_MPC_model= cvModel_mpc.bestModel

print("Accuracy for best mpc model after hyperparameter tuning on small train data = %g " % accuracy)
paramDict3 = {param[0].name: param[1] for param in best_MPC_model.stages[-1].extractParamMap().items()}
print("Best MPC model params-")
print(json.dumps(paramDict3, indent = 4))


print("############################# Full Datasets on Multilayer Perceptron Classifier #################################")
s3=time.time()
mpc_final= MultilayerPerceptronClassifier(labelCol="labels", featuresCol="features", layers=layers, seed=210116270, stepSize=paramDict3['stepSize'],
                                blockSize=paramDict3['blockSize'], maxIter=paramDict3['maxIter'])
stages_mpc_final = [vecAssembler, mpc_final]
pipeline_mpc_final = Pipeline(stages=stages_mpc_final)

mpc_model = pipeline_mpc_final.fit(rawtrain)
e3=time.time()
print("Train time for MPC model", e3-s3)
preds3 = mpc_model.transform(rawtest)
accuracy3 = evaluator_M.evaluate(preds3)

print("Test time for MPC model", time.time()-e3)

print("Accuracy for best Multilayer Perceptron Classifier model on whole dataset = %g " % accuracy3)
Area_under_ROC = evaluator.evaluate(preds3, {evaluator.metricName: "areaUnderROC"})
Area_under_PRC = evaluator.evaluate(preds3, {evaluator.metricName: "areaUnderPR"})
print("Area under Receiver Operating Characteristic Curve: {:.3f}".format(Area_under_ROC))
print("Area under Precision Recall Curve: {:.3f}".format(Area_under_PRC))
print("##############################################################")

### for interesting observations
preds_1 = RF_model.transform(rawtrain)
accuracy_1 = evaluator_M.evaluate(preds_1)

print("Accuracy for best Random Forest model on trainset = %g " % accuracy_1)

preds_2 = lr_model.transform(rawtrain)
accuracy_2 = evaluator_M.evaluate(preds_2)

print("Accuracy for best Logistic Regression model on trainset = %g " % accuracy_2)

preds_3 = mpc_model.transform(rawtrain)
accuracy_3 = evaluator_M.evaluate(preds_3)

print("Accuracy for best Multilayer Perceptron Classifier model on trainset = %g " % accuracy_3)