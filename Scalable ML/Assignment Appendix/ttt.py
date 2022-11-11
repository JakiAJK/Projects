from pyspark.sql import SparkSession
import numpy as np
from pyspark.sql.types import StringType
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import MultilayerPerceptronClassifier
import json
from pyspark.sql.functions import when

spark = SparkSession.builder \
        .master("local[3]") \
        .appName("Assignment 1") \
        .config("spark.local.dir","/fastdata/acp21jka/") \
        .getOrCreate()
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)
sc = spark.sparkContext
sc.setLogLevel("WARN")

#Load dataset and preprocessing
#rawdata = spark.read.csv("../Data/XOR_Arbiter_PUFs/5xor_128bit/train_5xor_128dim.csv")
#rawdata = spark.read.csv("/home/acp21jka/com6012/ScalableML/Data/XOR_Arbiter_PUFs/5xor_128bit/train_5xor_128dim.csv")
rawdata = spark.read.csv("/data/acp21jka/ScalableML/Data/XOR_Arbiter_PUFs/5xor_128bit/train_5xor_128dim.csv")
rawtest = spark.read.csv("/data/acp21jka/ScalableML/Data/XOR_Arbiter_PUFs/5xor_128bit/test_5xor_128dim.csv")
rawdata.cache()
ncolumns = len(rawdata.columns)



rawdata = rawdata.withColumnRenamed('_c128', 'labels')
rawtest = rawtest.withColumnRenamed('_c128', 'labels')
colmn_names = [x.name for x in rawdata.schema.fields if x.dataType == (StringType())]

for c in colmn_names:
    rawdata = rawdata.withColumn(c, col(c).cast("double"))
    rawtest = rawtest.withColumn(c, col(c).cast("double"))

rawdata = rawdata.withColumn("labels", when(rawdata.labels == -1, 0) \
      .otherwise(rawdata.labels))
rawtest = rawtest.withColumn("labels", when(rawtest.labels == -1, 0) \
      .otherwise(rawtest.labels))

Data,_ = rawdata.randomSplit([0.01, 0.99], 42)
trainingData, testData = Data.randomSplit([0.8, 0.2], 42)

print(f"There are {trainingData.cache().count()} rows in the training set, and {testData.cache().count()} in the test set")
vecAssembler = VectorAssembler(inputCols = colmn_names[0:ncolumns-1], outputCol = 'features')










spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
# Convert the Spark DataFrame to a Pandas DataFrame using Arrow
trainingDataPandas = trainingData.select("*").toPandas()
nfeatures = ncolumns-1
Xtrain = trainingDataPandas.iloc[:, 0:nfeatures]
ytrain = trainingDataPandas.iloc[:, -1]
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(20, activation='relu', input_shape=(np.shape(Xtrain)[1],)))
model.add(layers.Dense(5, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(Xtrain, ytrain, epochs=100, batch_size=100, validation_split=0.2, verbose=False)

import matplotlib.pyplot as plt

history_dict = history.history
acc_values= history_dict['accuracy']
val_acc_values= history_dict['val_accuracy']
epochs = range(1, len(acc_values)+1)

plt.plot(epochs, acc_values, 'bo', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("./Output/keras_nn_train_validation_history.png")

import tempfile
import tensorflow
import pandas as pd

class ModelWrapperPickable:

    def __init__(self, model):
        self.model = model

    def __getstate__(self):
        model_str = ''
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            tensorflow.keras.models.save_model(self.model, fd.name, overwrite=True)
            model_str = fd.read()
        d = { 'model_str': model_str }
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            self.model = tensorflow.keras.models.load_model(fd.name)
            
model_wrapper = ModelWrapperPickable(model)
Xtest = testData.select(colmn_names[0:ncolumns-1])
from pyspark.sql.types import StructField, StructType, DoubleType
pred_field = [StructField("prediction", DoubleType(), True)] 
new_schema = StructType(Xtest.schema.fields + pred_field)

def predict(iterator):
    for features in iterator:
        yield pd.concat([features, pd.Series(model_wrapper.model.predict(features).flatten(), name="prediction")], axis=1)
        
prediction_keras_df = Xtest.mapInPandas(predict, new_schema)
ypred_keras = prediction_keras_df.select('prediction').toPandas().values
ypred_keras[ypred_keras <0.5] = 0
ypred_keras[ypred_keras >0.5] = 1
testDataPandas = testData.select("*").toPandas()
ytest = testDataPandas.iloc[:, -1].values
from sklearn.metrics import accuracy_score
print("Accuracy of keras = %g " % accuracy_score(ypred_keras, ytest))

