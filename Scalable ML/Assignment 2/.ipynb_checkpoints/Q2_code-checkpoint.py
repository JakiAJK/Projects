from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import matplotlib.pyplot as plt
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from functools import reduce
from pyspark.sql import DataFrame
from pyspark.ml.clustering import KMeans
from pyspark.ml.clustering import KMeansModel
import pandas as pd

spark = SparkSession.builder \
        .master("local[3]") \
        .appName("Part 2 Assignment 2") \
        .config("spark.local.dir","/fastdata/acp21jka") \
        .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("ERROR") # Only log ERRORs


# load in ratings data
ratings = spark.read.load('/data/acp21jka/ScalableML/Data/ml-25m/ratings.csv', format = 'csv', inferSchema = "true", header = "true").cache()
print("Ratings.csv")
ratings.show(10,False)

tags = spark.read.load("/data/acp21jka/ScalableML/Data/ml-25m/tags.csv", format = 'csv', inferSchema = "true", header = "true").cache()
myseed=210116270

print("The number of unique users are: {:d}".format(ratings.select("userId").distinct().count()))
print("The number of unique movies are: {:d}".format(ratings.select("movieId").distinct().count()))



splitDF = ratings.randomSplit( [1.0,1.0,1.0,1.0,1.0] , myseed) # split the data for cross validation
(df1,df2,df3,df4,df5) = splitDF[0],splitDF[1],splitDF[2],splitDF[3],splitDF[4]

df_list=[df1,df2,df3,df4,df5]

#different ALS setting
als_setting1 = ALS(userCol = "userId", itemCol = "movieId", seed = myseed, coldStartStrategy = "drop")
als_setting2 = ALS(userCol = "userId", itemCol = "movieId", seed = myseed, coldStartStrategy = "drop", rank=50)

def compute_RMSEs(list_dfs, als_setting, get_tags=False):
    rmse_hot,rmse_cool=[],[]
    df_list=list_dfs

    for i in range(len(df_list)):
        
        test= df_list[i]
        other_dfs= df_list[:i]+ df_list[i+1:] # training set is based on the other 4 dfs
        training= reduce(DataFrame.unionAll, other_dfs) # join the above 4 dfs to form 1 df
        tot_users=training.select("userId").distinct().count() # total users in the new train set
        top_10_per=int(tot_users/10)
        
        # choose the top 10% and bottom 10% users in the training set by counting the number of times a user has reviewed
        hot_users= training.groupBy('userId').count().sort('count', ascending=False).limit(top_10_per)
        cool_users= training.groupBy('userId').count().sort('count', ascending=True).limit(top_10_per)


        als = als_setting
        evaluator = RegressionEvaluator(metricName = "rmse", labelCol = "rating", predictionCol = "prediction") # evaluator

        model = als.fit(training)

        predictions = model.transform(test)

        # filter out hot and cool users based on ID in the prediction df
        hot= predictions.join(hot_users, ['userId'], 'leftsemi')
        cool= predictions.join(cool_users, ['userId'], 'leftsemi')

        print("------------------------------split {}--------------------------".format(i+1))
        # finding the RMSEs
        rmse_1 = evaluator.evaluate(hot)
        #print("Root-mean-square error hot users = " + str(rmse_1))
        rmse_hot.append(rmse_1)
        print("Hot users RMSE for split {} is {}".format(i+1,rmse_1))

        rmse_2 = evaluator.evaluate(cool)
        #print("Root-mean-square error cool users = " + str(rmse_2))
        rmse_cool.append(rmse_2)
        print("Cool users RMSE for split {} is {}".format(i+1,rmse_2))        
        
        
        if get_tags:
            # get item factors
            item_factors= model.itemFactors
            kmeans = KMeans().setK(10).setSeed(myseed).setFeaturesCol("features").setPredictionCol("cluster")
        
            model_KMeans = kmeans.fit(item_factors)
            clusters = model_KMeans.transform(item_factors)
            top_clusters= clusters.groupBy('cluster').count().sort('count', ascending=False) # sort by count of clusters
            
            first_cluster=top_clusters.collect()[0]['cluster'] # cluster number occuring most number of times, largest cluster
            second_cluster=top_clusters.collect()[1]['cluster'] # 2nd largest cluster
        
            top_name1,bot_name1= find_tags(first_cluster, clusters) # retrieve tag names from biggest cluster
            top_name2,bot_name2= find_tags(second_cluster, clusters) # retrieve tag names from 2nd biggest cluster
            
            top_names.append(top_name1)
            top_names.append(top_name2)
            bot_names.append(bot_name1)
            bot_names.append(bot_name2)
            
        
    return rmse_hot,rmse_cool

def find_tags(cluster_num, all_cluster_data): # Pass a cluster to return top and bottom tag
    clusters=all_cluster_data
    movies_ID_cluster = clusters.filter((clusters.cluster == cluster_num)) # only filter data corresponding to the cluster number
    # movies_ID_cluster contains movie IDs which belong to the selected cluster only
    movies_ID_cluster = movies_ID_cluster.withColumnRenamed("id", "movieId") # rename column such that it matches tags.csv column name
    
    tags_filtered=tags.join(movies_ID_cluster,['movieId'], 'leftsemi') # filter out tags with movie IDs from the selected cluster
    #tags_filtered.show(10,False)
    tags_count=tags_filtered.groupBy('tag').count().sort('count', ascending=False)  # counting the number of tags for all movies in selected cluster
    #tags_count.show(10,False)
    top_tag_name=tags_count.collect()[0]['tag'] # highest count
    bottom_tag_name=tags_count.collect()[-1]['tag'] # lowest count
    
    return top_tag_name, bottom_tag_name
    


top_names,bot_names=[],[]
print("#############################################ALS setting 1####################################################\n")
RMSE_hot1,RMSE_cool1 = compute_RMSEs(df_list, als_setting1, get_tags=True) #find RMSEs and top tags from Lab 7 ALS setting
print("#############################################ALS setting 2####################################################\n")
RMSE_hot2,RMSE_cool2 = compute_RMSEs(df_list, als_setting2)


rmse_pd_df = pd.DataFrame({"ALS setting": ["ALS1", "ALS1", "ALS1", "ALS1", "ALS1",
                         "ALS2", "ALS2", "ALS2", "ALS2", "ALS2"],
                           "Split_num":[1,2,3,4,5,1,2,3,4,5],
                           "HOT Users RMSE": RMSE_hot1+RMSE_hot2,
                           "COOL Users RMSE": RMSE_cool1+RMSE_cool2})
table_rmse= spark.createDataFrame(rmse_pd_df)
print("RMSE for Hot and Cool Users per split per ALS setting")
table_rmse.show(10,False)

tags_pd_df = pd.DataFrame({"Cluster": ["Largest", "2nd largest", "Largest", "2nd largest", "Largest",
                         "2nd largest", "Largest", "2nd largest", "Largest", "2nd largest"],
                           "Split_num":[1,1,2,2,3,3,4,4,5,5],
                           "Top Tag": top_names,
                           "Bottom Tag": bot_names})
table_tags= spark.createDataFrame(tags_pd_df)
print("Top and Botto tags for 2 clusters")
table_tags.show(10,False)
