from pyspark.sql import SparkSession
import numpy as np
from pyspark.sql.types import StringType
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.ml.evaluation import BinaryClassificationEvaluator
import matplotlib.pyplot as plt

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import MultilayerPerceptronClassifier
import json
from pyspark.sql.functions import when
import time

spark = SparkSession.builder \
        .appName("Assignment 1") \
        .config("spark.local.dir","/fastdata/acp21jka/") \
        .getOrCreate()
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)
sc = spark.sparkContext
sc.setLogLevel("WARN")

#Load dataset and preprocessing
#rawdata = spark.read.csv("../Data/XOR_Arbiter_PUFs/5xor_128bit/train_5xor_128dim.csv")
#rawdata = spark.read.csv("/home/acp21jka/com6012/ScalableML/Data/XOR_Arbiter_PUFs/5xor_128bit/train_5xor_128dim.csv")
rawtrain = spark.read.csv("/data/acp21jka/ScalableML/Data/XOR_Arbiter_PUFs/5xor_128bit/train_5xor_128dim.csv")
rawtest = spark.read.csv("/data/acp21jka/ScalableML/Data/XOR_Arbiter_PUFs/5xor_128bit/test_5xor_128dim.csv")
rawtrain.cache()
ncolumns = len(rawtrain.columns)



rawtrain = rawtrain.withColumnRenamed('_c128', 'labels')
rawtest = rawtest.withColumnRenamed('_c128', 'labels')
colmn_names = [x.name for x in rawtrain.schema.fields if x.dataType == (StringType())]

for c in colmn_names:
    rawtrain = rawtrain.withColumn(c, col(c).cast("double"))
    rawtest = rawtest.withColumn(c, col(c).cast("double"))

Data,_ = rawtrain.randomSplit([0.01, 0.99], 42)
small_trData, small_testData = Data.randomSplit([0.8, 0.2], 42)

print(f"There are {small_trData.cache().count()} rows in the training set, and {small_testData.cache().count()} in the test set")
vecAssembler = VectorAssembler(inputCols = colmn_names[0:ncolumns-1], outputCol = 'features')

rf = RandomForestClassifier(labelCol="labels", featuresCol="features", seed=210116270)
stages_rf = [vecAssembler, rf]
pipeline_rf = Pipeline(stages=stages_rf)

# Create Paramater grid for crossvalidation. Each paramter is added with .addGrid()
paramGrid1 = ParamGridBuilder() \
    .addGrid(rf.maxDepth, [1, 5, 10]) \
    .addGrid(rf.maxBins, [2, 10, 20]) \
    .addGrid(rf.subsamplingRate, [0.1, 0.5, 0.9]) \
    .build()

evaluator = BinaryClassificationEvaluator\
      (labelCol="labels", rawPredictionCol="prediction")

evaluator_M = MulticlassClassificationEvaluator\
      (labelCol="labels", predictionCol="prediction", metricName="accuracy")

# Make Crossvalidator object, we use the same evaluator 
crossval1 = CrossValidator(estimator=pipeline_rf,
                          estimatorParamMaps=paramGrid1,
                          evaluator=evaluator,
                          numFolds=5)

# .fit() will run crossvalidation on all the folds and return the model with the best paramaters found
cvModel_rf = crossval1.fit(small_trData)
prediction = cvModel_rf.transform(small_testData)
accuracy = evaluator_M.evaluate(prediction)
print("Predictions")

best_RF_model= cvModel_rf.bestModel
print("Accuracy for best rf model = %g " % accuracy)
# .bestModel() returns the model object in the crossvalidator. This object is a pipeline
# .stages[-1] returns the last stage in the pipeline, which for our case is our classifier
# .extractParamMap() returns a map with the parameters, which we turn into a dictionary 
paramDict1 = {param[0].name: param[1] for param in best_RF_model.stages[-1].extractParamMap().items()}
# Here, we're converting the dictionary to a JSON object to make it easy to print. You can print it however you'd like
print(json.dumps(paramDict1, indent = 4))

#best_max_depth= bestModel._java_obj.getMaxDepth()
#best_max_bins= bestModel._java_obj.getMaxBins()
#best_subsampling= bestModel._java_obj.getMaxDepth()


print("##############################################################")
s1=time.time()
rf_final= RandomForestClassifier(labelCol="labels", featuresCol="features", seed=210116270, maxDepth=paramDict1['maxDepth'],
                                maxBins=paramDict1['maxBins'], subsamplingRate=paramDict1['subsamplingRate'])
stages_rf_final = [vecAssembler, rf_final]
pipeline_rf_final = Pipeline(stages=stages_rf_final)

RF_model = pipeline_rf_final.fit(rawtrain)
print("Train time for RF model", time.time()-s1)
preds = RF_model.transform(rawtest)
accuracy1 = evaluator_M.evaluate(preds)



print("Accuracy for best random forest model on whole dataset = %g " % accuracy1)
Area_under_ROC = evaluator.evaluate(preds, {evaluator.metricName: "areaUnderROC"})
Area_under_PRC = evaluator.evaluate(preds, {evaluator.metricName: "areaUnderPR"})
print("Area under Receiver Operating Characteristic Curve: {:.3f}".format(Area_under_ROC))
print("Area under Precision Recall Curve: {:.3f}".format(Area_under_PRC))
print("##############################################################")

########################################################## 2 #################################################################

lr = LogisticRegression(featuresCol='features', labelCol='labels', family='binomial')

stages_lr = [vecAssembler, lr]
pipeline_lr = Pipeline(stages=stages_lr)


#Create Paramater grid for crossvalidation. Each paramter is added with .addGrid()
#FUN FACT: replacing 0.0 and 1.0 with 0 and 1 will return a Java cast conversion error
#This is because elasticNetParam requires a certain type (float in this case)
paramGrid2 = ParamGridBuilder() \
    .addGrid(lr.elasticNetParam, [0.0, 0.2, 0.5, 0.7, 1.0]) \
    .addGrid(lr.regParam, [0.001, 0.01, 0.1]) \
    .addGrid(lr.maxIter, [25, 50, 100]) \
    .build()



# Make Crossvalidator object
crossval2 = CrossValidator(estimator=pipeline_lr,
                          estimatorParamMaps=paramGrid2,
                          evaluator=evaluator,
                          numFolds=3)

# .fit() will run crossvalidation on all the folds and return the model with the best paramaters found
cvModel_lr = crossval2.fit(small_trData)
prediction = cvModel_lr.transform(small_testData)
accuracy = evaluator_M.evaluate(prediction)
best_LR_model= cvModel_lr.bestModel

print("Accuracy for best lr model = %g " % accuracy)
# .bestModel() returns the model object in the crossvalidator. This object is a pipeline
# .stages[-1] returns the last stage in the pipeline, which for our case is our classifier
# .extractParamMap() returns a map with the parameters, which we turn into a dictionary 
paramDict2 = {param[0].name: param[1] for param in best_LR_model.stages[-1].extractParamMap().items()}
# Here, we're converting the dictionary to a JSON object to make it easy to print. You can print it however you'd like
print(json.dumps(paramDict2, indent = 4))


print("##############################################################")
s2=time.time()
lr_final= LogisticRegression(featuresCol='features', labelCol='labels', family='binomial', elasticNetParam=paramDict2['elasticNetParam'],
                                regParam=paramDict2['regParam'], maxIter=paramDict2['maxIter'])
stages_lr_final = [vecAssembler, lr_final]
pipeline_lr_final = Pipeline(stages=stages_lr_final)

lr_model = pipeline_lr_final.fit(rawtrain)
print("Train time for LR model", time.time()-s2)
preds = lr_model.transform(rawtest)
accuracy2 = evaluator_M.evaluate(preds)


print("Accuracy for best logistic regression model on whole dataset = %g " % accuracy2)
Area_under_ROC = evaluator.evaluate(preds, {evaluator.metricName: "areaUnderROC"})
Area_under_PRC = evaluator.evaluate(preds, {evaluator.metricName: "areaUnderPR"})
print("Area under Receiver Operating Characteristic Curve: {:.3f}".format(Area_under_ROC))
print("Area under Precision Recall Curve: {:.3f}".format(Area_under_PRC))
print("##############################################################")
print("##############################################################")


########################################################## 3 #################################################################



# The first element HAS to be equal to the number of input features
layers = [len(small_trData.columns)-1, 20, 2] 
mpc = MultilayerPerceptronClassifier(labelCol="labels", featuresCol="features", layers=layers, seed=210116270)

# Create the pipeline
stages3 = [vecAssembler, mpc]
pipeline_mpc = Pipeline(stages=stages3)

paramGrid3 = ParamGridBuilder() \
    .addGrid(mpc.stepSize, [0.1, 0.01, 0.001]) \
    .addGrid(mpc.blockSize, [64,128,256]) \
    .addGrid(mpc.maxIter, [10, 25, 100]) \
    .build()


# Make Crossvalidator object
crossval3 = CrossValidator(estimator=pipeline_mpc,
                          estimatorParamMaps=paramGrid3,
                          evaluator=evaluator,
                          numFolds=3)

# .fit() will run crossvalidation on all the folds and return the model with the best paramaters found
cvModel_mpc = crossval3.fit(small_trData)
prediction = cvModel_mpc.transform(small_testData)
accuracy = evaluator_M.evaluate(prediction)
best_MPC_model= cvModel_mpc.bestModel

print("Accuracy for best mpc model = %g " % accuracy)
# .bestModel() returns the model object in the crossvalidator. This object is a pipeline
# .stages[-1] returns the last stage in the pipeline, which for our case is our classifier
# .extractParamMap() returns a map with the parameters, which we turn into a dictionary 
paramDict3 = {param[0].name: param[1] for param in best_MPC_model.stages[-1].extractParamMap().items()}
# Here, we're converting the dictionary to a JSON object to make it easy to print. You can print it however you'd like
print(json.dumps(paramDict3, indent = 4))


print("##############################################################")
s3=time.time()
mpc_final= MultilayerPerceptronClassifier(labelCol="labels", featuresCol="features", layers=layers, seed=210116270, stepSize=paramDict3['stepSize'],
                                blockSize=paramDict3['blockSize'], maxIter=paramDict3['maxIter'])
stages_mpc_final = [vecAssembler, mpc_final]
pipeline_mpc_final = Pipeline(stages=stages_mpc_final)

mpc_model = pipeline_mpc_final.fit(rawtrain)
print("Train time for MPC model", time.time()-s3)
preds = mpc_model.transform(rawtest)
accuracy3 = evaluator_M.evaluate(preds)


print("Accuracy for best Multilayer Perceptron Classifier model on whole dataset = %g " % accuracy3)
Area_under_ROC = evaluator.evaluate(preds, {evaluator.metricName: "areaUnderROC"})
Area_under_PRC = evaluator.evaluate(preds, {evaluator.metricName: "areaUnderPR"})
print("Area under Receiver Operating Characteristic Curve: {:.3f}".format(Area_under_ROC))
print("Area under Precision Recall Curve: {:.3f}".format(Area_under_PRC))
print("##############################################################")
print("##############################################################")

