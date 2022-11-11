from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import matplotlib.pyplot as plt
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.functions import *

spark = SparkSession.builder \
        .master("local[2]") \
        .appName("Part 2 Assignment 2") \
        .config("spark.local.dir","/fastdata/acp21jka") \
        .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

# load in ratings data
ratings = spark.read.load('/data/acp21jka/ScalableML/Data/ml-25m/ratings.csv', format = 'csv', inferSchema = "true", header = "true").cache()
ratings.show(20,False)
myseed=210116270

print("The number of unique movies are: {:d}".format(ratings.select("userId").distinct().count()))
print("The number of unique movies are: {:d}".format(ratings.select("movieId").distinct().count()))



# split
(training, test) = ratings.randomSplit([0.8, 0.2], myseed)
training = training.cache()
test = test.cache()


from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")

    #pipeline = Pipeline(stages=stages)

    # We use a ParamGridBuilder to construct a grid of parameters to search over.
    # A CrossValidator requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
    # We use a ParamGridBuilder to construct a grid of parameters to search over.
    # With 2 values for als.rank, 2 values for als.regParam, and 1 value for als.maxIter,
    # this grid will have 2 x 2 x 1 = 4 parameter settings for CrossValidator to choose from.
param_grid = ParamGridBuilder()\
    .addGrid(als.rank, [10]) \
    .build()
    
cross_val = CrossValidator(estimator=als, 
                               estimatorParamMaps=param_grid,
                               evaluator=RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction"),
                               numFolds=5,
                               collectSubModels=True # this flag allows us to store ALL the models trained during k-fold cross validation
                               )

cv_model = cross_val.fit(training)


for k, models in enumerate(cv_model.subModels):
    print("*************** Fold #{:d} ***************\n".format(k+1))
    for i, m in enumerate(models):
        print("--- Model #{:d} out of {:d} ---".format(i+1, len(models)))
        print("\tModel summary: {}\n".format(m))
    print("***************************************\n")



\







from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

# load in ratings data
ratings = spark.read.load('/data/acp21jka/ScalableML/Data/ml-latest-small/ratings.csv', format = 'csv', inferSchema = "true", header = "true")
ratings.show(20,False)
myseed=6012

# split
(training, test) = ratings.randomSplit([0.8, 0.2], myseed)
training = training
#test = test.cache()

# define model
als = ALS(userCol = "userId", itemCol = "movieId", seed = myseed, coldStartStrategy = "drop")
# define evaluator

# run model with default rank (which is 10)
model = als.fit(training)

item_factors= model.itemFactors
print(item_factors.count())
item_factors.show(10,False)

from pyspark.ml.clustering import KMeans
from pyspark.ml.clustering import KMeansModel
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.linalg import Vectors
kmeans = KMeans().setK(10).setSeed(myseed).setFeaturesCol("features").setPredictionCol("cluster")
model_KMeans = kmeans.fit(item_factors)
clusters = model_KMeans.transform(item_factors)
clusters.show(10,False)
top_clusters=clusters.groupBy('cluster').count().sort('count', ascending=False)
top_clusters.show()
top_2= top_clusters.limit(2)
top_2.show()

genome_scores = spark.read.load("/data/acp21jka/ScalableML/Data/ml-25m/genome-scores.csv", format = 'csv', inferSchema = "true", header = "true")
genome_tags = spark.read.load("/data/acp21jka/ScalableML/Data/ml-25m/genome-tags.csv", format = 'csv', inferSchema = "true", header = "true")

first_cluster=top_clusters.collect()[0]['cluster']
second_cluster=top_clusters.collect()[1]['cluster']
movies_ID_cluster=clusters.filter( (clusters.cluster == first_cluster) | (clusters.cluster == second_cluster))
movies_ID_cluster.show(10,False)
movies_ID_cluster = movies_ID_cluster.withColumnRenamed("id", "movieId")



genome_scores_filtered=genome_scores.join(movies_ID_cluster,['movieId'], 'leftsemi')
genome_scores_filtered.show()
        
        
temp_list=genome_scores_filtered.groupBy("tagId").sum("relevance").collect()
df = spark.sparkContext.parallelize(temp_list).toDF()
df=df.withColumnRenamed("sum(relevance)",'total')
df=df.orderBy(df.total.desc())
df.show(10)

top_tag_id=df.collect()[0]['tagId']
bottom_tag_id=df.collect()[-1]['tagId']
        
top_tag_name=genome_tags.filter(genome_tags["tagId"]== top_tag_id).collect()[0][1]       
bottom_tag_name=genome_tags.filter(genome_tags["tagId"]== bottom_tag_id).collect()[0][1]




neww=tags.join(movies_ID_cluster,['movieId'], 'leftsemi')
tags_all=neww.groupBy('tag').count().sort('count', ascending=False)
tags_all.show(20)
top_tag=tags_all.collect()[0]['tag']
bottom_tag=tags_all.collect()[-1]['tag']
        
print("Tags are-",top_tag,bottom_tag)























































# define model
als = ALS(userCol = "userId", itemCol = "movieId", seed = myseed, coldStartStrategy = "drop")
# define evaluator
evaluator = RegressionEvaluator(metricName = "rmse", labelCol = "rating", predictionCol = "prediction")

# run model with default rank (which is 10)
model = als.fit(training)


def run_model(_train, _test, _als, _evaluator):
    
    model = _als.fit(_train)
    predictions = model.transform(_test)
    rmse = _evaluator.evaluate(predictions)
    print(f"rank {_als.getRank()} Root-mean-square error = {rmse}")
    return rmse

# run model for 5 times    
ranks = [5,10,15,20,25]
results = []
for _rank in ranks:
    als.setRank(_rank)
    results.append(run_model(training, test, als, evaluator))

print(results)

# select a user
users = ratings.select(als.getUserCol()).distinct().sample(withReplacement = False, fraction = 0.1, seed = myseed).limit(1)
users.show()
# get recomendations from model
userSubsetRecs = model.recommendForUserSubset(users, 5)
userSubsetRecs.show(1, False)
# get movie_id
movies = userSubsetRecs.collect()[0].recommendations
movies = [row.movieId for row in movies]
print(movies)

# loading movies.csv
movie_data = spark.read.load('/data/acp21jka/ScalableML/Data/ml-25m/movies.csv', format = 'csv', inferSchema = "true", header = "true").cache()
movie_data.show(20, False)

# find movie according to movie_id
for movie_id in movies:
    _data = movie_data.filter(movie_data.movieId == f"{movie_id}").collect()
    _data = _data[0]    
    print(f"movie id: {movie_id} \t title: {_data.title} \t genres: {_data.genres}")


