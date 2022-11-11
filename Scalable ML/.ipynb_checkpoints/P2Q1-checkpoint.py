from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import dayofweek, weekofyear
from datetime import datetime
from pyspark.sql.functions import col,udf
from pyspark.sql.types import TimestampType

spark = SparkSession.builder \
        .master("local[2]") \
        .appName("Part 1 Assignment 2") \
        .config("spark.local.dir","/fastdata/acp21jka") \
        .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")



logFile = spark.read.text("/data/acp21jka/ScalableML/Data/NASA_access_log_Jul95.gz").cache()  # add it to cache, so it can be used in the following steps efficiently 

# split into 5 columns using regex and split
data = logFile.withColumn('host', F.regexp_extract('value', '^(.*) - -.*', 1)) \
                 .withColumn('timestamp', F.regexp_extract('value', '\[(\S+)',1)) \
                .withColumn('request', F.regexp_extract('value', '.*\"(.*)\".*',1)) \
                .withColumn('HTTP reply code', F.split('value', ' ').getItem(F.size(F.split('value', ' ')) -2).cast("int")) \
                .withColumn('bytes in the reply', F.split('value', ' ').getItem(F.size(F.split('value', ' ')) - 1).cast("int")).drop("value").cache()

data.show(10,False)
print("Number of rows originally ",data.count())
print("Invalid data-")
data.where(~F.col('timestamp').contains('Jul')).show()

# We select all data except the ones where data is invalid or empty
data=data.where(F.col('timestamp').contains('Jul')).cache()
data.show(10,False)
print("Number of rows after removing invalid data ",data.count())


# Convert the time data such that it can be parsed using the pyspark Timestamp functions

myfunc =  udf(lambda x: datetime.strptime(x, '%d/%b/%Y:%H:%M:%S'), TimestampType())

time_data = data.withColumn('casted_timestamp', myfunc(col('timestamp'))) \
            .withColumn("DayOfweek", dayofweek(col("casted_timestamp"))) \
            .withColumn("Week", weekofyear(col("casted_timestamp")))

time_data.show(5,False)

#data2 = data1.withColumn("DayOfweek", dayofweek(col("time_"))) \
             #.withColumn("Week", weekofyear(col("time_")))
#data2.show(5)


final=time_data.groupBy('Week','DayOfweek').count()

day_dict={1:'Sunday',2:'Monday',3:'Tuesday',4:'Wednesday',5:'Thursday',6:'Friday',7:'Saturday'}

day_map_func = udf(lambda num : day_dict.get(num,num))
final = final.withColumn("Day", day_map_func(col("DayOfweek")))


final_sorted=final.orderBy(["DayOfweek","count"], ascending=True)
final_sorted.show(20,False)

#Sun=final.filter(final.Day == "Sunday")
#Mon=final.filter(final.Day == "Monday")
#Tue=final.filter(final.Day == "Tuesday")
#Wed=final.filter(final.Day == "Wednesday")
#Thu=final.filter(final.Day == "Thursday")
#Fri=final.filter(final.Day == "Friday")
#Sat=final.filter(final.Day == "Saturday")

def get_values(day_num, data):
    day_data= data.filter(data.DayOfweek == day_num)
    return day_data.collect()[0]['count'] , day_data.collect()[-1]['count']

low_Sun, high_Sun= get_values(1,final_sorted)
low_Mon, high_Mon= get_values(2,final_sorted)
low_Tue, high_Tue= get_values(3,final_sorted)
low_Wed, high_Wed= get_values(4,final_sorted)
low_Thu, high_Thu= get_values(5,final_sorted)
low_Fri, high_Fri= get_values(6,final_sorted)
low_Sat, high_Sat= get_values(7,final_sorted)


print("Sun", low_Sun, high_Sun)
print("Mon", low_Mon, high_Mon)
print("Tue", low_Tue, high_Tue)
print("Wed", low_Wed, high_Wed)
print("Thu", low_Thu, high_Thu)
print("Fri", low_Fri, high_Fri)
print("Sat", low_Sat, high_Sat)

#>>> print("Sun", low_Sun, high_Sun)
#Sun 35272 60265
#>>> print("Mon", low_Mon, high_Mon)
#Mon 64259 89584
#>>> print("Tue", low_Tue, high_Tue)
#Tue 62699 80407
#>>> print("Wed", low_Wed, high_Wed)
#Wed 58849 94575
#>>> print("Thu", low_Thu, high_Thu)
#Thu 61680 134203
#>>> print("Fri", low_Fri, high_Fri)
#Fri 27121 87233
#>>> print("Sat", low_Sat, high_Sat)
#Sat 35267 64714


print("=========================================================== MPG data part ============================================================")
#.mpg


nn=data.filter(data.request.contains('.mpg'))
nn.show(10,False)

most=nn.groupBy('request').count().sort('count', ascending=False).limit(12)
least=nn.groupBy('request').count().sort('count', ascending=True).limit(12)


least=least.withColumn('request', F.regexp_replace('request', 'HTTP/1.0', ''))
most=most.withColumn('request', F.regexp_replace('request', 'HTTP/1.0', ''))
pattern='([^\/]+$)'
most=most.withColumn('request', F.regexp_extract('request', pattern ,1)).withColumnRenamed("request", "video name").withColumnRenamed("count", "total number of requests")
most.show(12,False)
least=least.withColumn('request', F.regexp_extract('request', pattern ,1)).withColumnRenamed("request", "video name").withColumnRenamed("count", "total number of requests")
least.show(12,False)

all_data=most.union(least)

#.mpg