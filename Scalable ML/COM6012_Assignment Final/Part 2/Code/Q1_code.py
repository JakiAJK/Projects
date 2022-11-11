from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import dayofweek, weekofyear
from datetime import datetime
from pyspark.sql.functions import udf, col
from pyspark.sql.types import TimestampType
import numpy as np
import pandas as pd
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt

spark = SparkSession.builder \
        .master("local[2]") \
        .appName("Part 1 Assignment 2") \
        .config("spark.local.dir","/fastdata/acp21jka") \
        .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("ERROR") # Only log ERRORs



logFile = spark.read.text("../Data/NASA_access_log_Jul95.gz").cache()  # add it to cache


# split into 5 columns using regex and split
data = logFile.withColumn('host', F.regexp_extract('value', '^(.*) - -.*', 1)) \
                 .withColumn('timestamp', F.regexp_extract('value', '\[(\S+)',1)) \
                .withColumn('request', F.regexp_extract('value', '.*\"(.*)\".*',1)) \
                .withColumn('HTTP reply code', F.split('value', ' ').getItem(F.size(F.split('value', ' ')) -2).cast("int")) \
                .withColumn('bytes in the reply', F.split('value', ' ').getItem(F.size(F.split('value', ' ')) - 1).cast("int")).drop("value").cache()

print("Data after splitting using regex")
data.show(10,False)
print("Number of rows originally ",data.count())
print("Invalid data-")
data.where(~F.col('timestamp').contains('Jul')).show()

# We select all data except the ones where data is invalid or empty
data=data.where(F.col('timestamp').contains('Jul')).cache()
print("Number of rows after removing invalid data ",data.count())


# Convert the time data such that it can be parsed using the pyspark Timestamp functions

myfunc =  udf(lambda x: datetime.strptime(x, '%d/%b/%Y:%H:%M:%S'), TimestampType())

# new df to accomodate Day of week and Week of Year
time_data = data.withColumn('casted_timestamp', myfunc(col('timestamp'))) \
            .withColumn("DayOfweek", dayofweek(col("casted_timestamp"))) \
            .withColumn("Week", weekofyear(col("casted_timestamp")))

time_data.show(5,False)

print("Checking for records on or after 29th July")
time_data.filter(time_data["casted_timestamp"]>='1995-07-29').show()


final=time_data.groupBy('Week','DayOfweek').count() # Data with days of week

day_dict={1:'Sunday',2:'Monday',3:'Tuesday',4:'Wednesday',5:'Thursday',6:'Friday',7:'Saturday'} # In dayOfWeek, the week starts on Sunday

day_map_func = udf(lambda num : day_dict.get(num,num)) # To find the name of Day based on weekday number
final = final.withColumn("Day", day_map_func(col("DayOfweek")))


final_sorted=final.orderBy(["DayOfweek","count"], ascending=True)
print("Filtered data according to DayOfWeek")
final_sorted.show(20,False)


def get_values(day_num, data): # Filter data based on day and count the total requests on each weekday
    day_data= data.filter(data.DayOfweek == day_num)
    return day_data.collect()[0]['count'] , day_data.collect()[-1]['count'] 
# 1st row has lowest number of requests, last row has highest number of requests because we sorted in ascending order

low_list, high_list=[], []

#getting the values to a list for plotting and table
for i in range(7):
    low, high= get_values(i+1,final_sorted)
    low_list.append(low)
    high_list.append(high)

for i in range(7):
    print("Lowest requests on {} are-{} and Highest are-{}".format(day_dict[i+1],low_list[i], high_list[i]))
    print("-------------------------------------------------------------")

data_pd_df = pd.DataFrame({"WeekDay": ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday",
                         "Friday", "Saturday"],
                           "Highest requests": high_list,
                           "Lowest requests": low_list})
table_data= spark.createDataFrame(data_pd_df)
print("Maximum and Minimum requests on each day of week in July 1995")
table_data.show(10,False)

### plotting
ind = np.arange(7)
width=0.3
fig = plt.figure(figsize=(11,7))
ax = fig.add_axes([0,0,1,1])
ax.bar(ind + 0.0, high_list, color = 'r', width=width, label='Highest')
ax.bar(ind + width, low_list, color = 'b', width=width, label='Lowest')

plt.ylabel('Number of requests')
plt.title('Highest and Lowest requests on each Week Day')
plt.xticks(ind + width / 2, ('Sunday', 'Monday', 'Tuesday', 'Wednesday'
                             , 'Thursday', 'Friday', 'Saturday'))
plt.legend(loc='upper right' , numpoints = 1)
plt.savefig("""../Output/Q1fig_A.png"""
            , bbox_inches='tight')


print("=========================================================== MPG requests data part ============================================================")


mpg_filtered_logs=data.filter(data.request.contains('.mpg')) # checking for .mpg in the request column
print(".mpg filtered data")
mpg_filtered_logs.show(10,False)

most=mpg_filtered_logs.groupBy('request').count().sort('count', ascending=False).limit(12) # 12 most requested
least=mpg_filtered_logs.groupBy('request').count().sort('count', ascending=True).limit(12) # 12 least requested
print("Most requested mpg videos")
most.show(12,False)
print("Least requested mpg videos")
least.show(12,False)

print("Combining both most and least")
all_mpg_requests= most.union(least)
all_mpg_requests.show(24,False)

print("After removing HTTP requests") # removing request type so as to find video filename
all_mpg_requests= all_mpg_requests.withColumn('request', F.regexp_replace('request', 'HTTP/1.0', ''))
all_mpg_requests.show(24,False)

print("Extracting video name")
pattern_for_names='([^\/]+$)' # reg exp parrtern to extract filename
all_mpg_requests= all_mpg_requests.withColumn('request', F.regexp_extract('request', pattern_for_names ,1)) \
            .withColumnRenamed("request", "video_name").withColumnRenamed("count", "total_number_of_requests")

all_mpg_requests.show(24,False)


all_mpg_requests_pdf=all_mpg_requests.toPandas() # To plot

### plotting
bar_plot = all_mpg_requests_pdf.plot(figsize = (10,7), kind = "barh", color = "#cc9900", width = 0.6,
                               x = "video_name", y = "total_number_of_requests", legend = False)

bar_plot.invert_yaxis()

plt.xlabel("Count")
plt.ylabel("Video Name")
plt.title("Count of most and least request .mpg file")
plt.xticks()
plt.yticks()
plt.savefig("""../Output/Q1fig_B.png"""
            , bbox_inches='tight')
