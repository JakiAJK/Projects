from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import matplotlib.pyplot as plt

spark = SparkSession.builder \
        .master("local[3]") \
        .appName("Part 2 Assignment 2") \
        .config("spark.local.dir","/fastdata/acp21jka") \
        .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from functools import reduce
from pyspark.sql import DataFrame

from pyspark.ml.clustering import KMeans
from pyspark.ml.clustering import KMeansModel
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.linalg import Vectors

# load in ratings data
ratings = spark.read.load('/data/acp21jka/ScalableML/Data/ml-25m/ratings.csv', format = 'csv', inferSchema = "true", header = "true").cache()
ratings.show(20,False)
genome_scores = spark.read.load("/data/acp21jka/ScalableML/Data/ml-25m/genome-scores.csv", format = 'csv', inferSchema = "true", header = "true").cache()
genome_tags = spark.read.load("/data/acp21jka/ScalableML/Data/ml-25m/genome-tags.csv", format = 'csv', inferSchema = "true", header = "true").cache()
myseed=210116270

print("The number of unique users are: {:d}".format(ratings.select("userId").distinct().count()))
print("The number of unique movies are: {:d}".format(ratings.select("movieId").distinct().count()))

# split
#(training, test) = ratings.randomSplit([0.8, 0.2], myseed)
#training = training.cache()
#test = test.cache()

tot_users=ratings.select("userId").distinct().count()
top_10_per=int(tot_users/10)

splitDF = ratings.randomSplit( [1.0,1.0,1.0,1.0,1.0] , myseed) 
(df1,df2,df3,df4,df5) = splitDF[0],splitDF[1],splitDF[2],splitDF[3],splitDF[4]

df_list=[df1,df2,df3,df4,df5]

als_setting1 = ALS(userCol = "userId", itemCol = "movieId", seed = myseed, coldStartStrategy = "drop")
als_setting2 = ALS(userCol = "userId", itemCol = "movieId", seed = myseed, coldStartStrategy = "drop", rank=5, maxIter=15, regParam=0.01)

def compute_RMSEs(list_dfs, als_setting):
    rmse_hot=[]
    rmse_cool=[]
    df_list=list_dfs

    for i in range(len(df_list)):
        test=df_list[i]
        other_dfs=df_list[:i]+ df_list[i+1:]
        training = reduce(DataFrame.unionAll, other_dfs)

        #training.groupBy('userId').count().show(10,False)

        hot_users=training.groupBy('userId').count().sort('count', ascending=False).limit(top_10_per)
        cool_users=training.groupBy('userId').count().sort('count', ascending=True).limit(top_10_per)


        #hot_users.show(30,False)
        #cool_users.show(30,False)

        als = als_setting
        # define evaluator
        evaluator = RegressionEvaluator(metricName = "rmse", labelCol = "rating", predictionCol = "prediction")

        # run model with default rank (which is 10)
        model = als.fit(training)

        predictions = model.transform(test)

        #predictions.show(10,False)

        hot= predictions.join(hot_users, ['userId'], 'leftsemi')
        cool= predictions.join(cool_users, ['userId'], 'leftsemi')


        #hot.show(10,False)
        rmse_1 = evaluator.evaluate(hot)
        #print("Root-mean-square error hot users = " + str(rmse_1))
        rmse_hot.append(rmse_1)
    
        #cool.show(10,False)
        rmse_2 = evaluator.evaluate(cool)
        #print("Root-mean-square error cool users = " + str(rmse_2))
        rmse_cool.append(rmse_2)
        
        
        
        
        item_factors= model.itemFactors
        #print(item_factors.count())
        #item_factors.show(10,False)
        
        #kmeans = KMeans(k=10, seed=myseed)  # Two clusters with seed = 1
        kmeans = KMeans().setK(10).setSeed(myseed).setFeaturesCol("features").setPredictionCol("cluster")
        model_KMeans = kmeans.fit(item_factors)
        clusters = model_KMeans.transform(item_factors)
        #clusters.show(10,False)
        top_clusters=clusters.groupBy('cluster').count().sort('count', ascending=False)
        #top_clusters.show()
        top_clusters.show(2,False)
        
        first_cluster=top_clusters.collect()[0]['cluster']
        second_cluster=top_clusters.collect()[1]['cluster']
        
        top_name1,bot_name1= find_tags(first_cluster, clusters)
        top_name2,bot_name2= find_tags(second_cluster, clusters)
        
        print("Tags cluster 1-", top_name1, bot_name1)
        print("Tags cluster 2-", top_name2, bot_name2)
        
    return rmse_hot,rmse_cool

def find_tags(cluster_num, all_cluster_data): # Pass a cluster to return top and bottom tag
    clusters=all_cluster_data
    movies_ID_cluster = clusters.filter((clusters.cluster == cluster_num))
    movies_ID_cluster = movies_ID_cluster.withColumnRenamed("id", "movieId")
    
    print("The number of unique movies are: {:d}".format(movies_ID_cluster.select("movieId").distinct().count()))
    genome_scores_filtered=genome_scores.join(movies_ID_cluster,['movieId'], 'leftsemi')
    
    
    tags_relevance = spark.sparkContext.parallelize(genome_scores_filtered.groupBy("tagId").sum("relevance").collect()).toDF()
    
    tags_relevance=tags_relevance.withColumnRenamed("sum(relevance)",'total')
    tags_relevance_sorted=tags_relevance.orderBy(tags_relevance.total.desc())
    
    tags_relevance_sorted.show(2,False)

    top_tag_id=tags_relevance_sorted.collect()[0]['tagId']
    bottom_tag_id=tags_relevance_sorted.collect()[-1]['tagId']
        
    top_tag_name=genome_tags.filter(genome_tags["tagId"]== top_tag_id).collect()[0][1]       
    bottom_tag_name=genome_tags.filter(genome_tags["tagId"]== bottom_tag_id).collect()[0][1]
    
    return top_tag_name, bottom_tag_name
    
    
    
    
    
    
    
    


RMSE_hot1,RMSE_cool1 = compute_RMSEs(df_list, als_setting1)
RMSE_hot2,RMSE_cool2 = compute_RMSEs(df_list, als_setting2)

print(RMSE_hot1)
print(RMSE_hot2)
print(RMSE_cool1)
print(RMSE_cool2)

#import numpy as np
#print('ALS 1')
#spark.createDataFrame((np.array([RMSE_hot1, RMSE_cool1])).T, schema=['hot', 'cool']).show()

#print('ALS 2')
#spark.createDataFrame((np.array([RMSE_hot2, RMSE_cool2])).T, schema=['hot', 'cool']).show()
