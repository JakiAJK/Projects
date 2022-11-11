from pyspark.sql import SparkSession
import pyspark.sql.functions as F


spark = SparkSession.builder \
        .master("local[4]") \
        .appName("Lab 2 Exercise") \
        .config("spark.local.dir","/fastdata/acp21jka") \
        .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("WARN") 

logFile = spark.read.text("../Data/NASA_access_log_Aug95.gz").cache()  # add it to cache, so it can be used in the following steps efficiently
logFile.show(20, False)

# split into 5 columns using regex and split
data = logFile.withColumn('host', F.regexp_extract('value', '^(.*) - -.*', 1)) \
                .withColumn('timestamp', F.regexp_extract('value', '.* - - \[(.*)\].*',1)) \
                .withColumn('request', F.regexp_extract('value', '.*\"(.*)\".*',1)) \
                .withColumn('HTTP reply code', F.split('value', ' ').getItem(F.size(F.split('value', ' ')) -2).cast("int")) \
                .withColumn('bytes in the reply', F.split('value', ' ').getItem(F.size(F.split('value', ' ')) - 1).cast("int")).drop("value").cache()
data.show(20,False)

# number of unique hosts

n_hosts = data.select('host').distinct().count()
print("==================== Question 2 ====================")
print(f"There are {n_hosts} unique hosts")
print("====================================================")

# most visited host

host_count = data.select('host').groupBy('host').count().sort('count', ascending=False)
host_max = host_count.select("host").first()['host']
print("==================== Question 3 ====================")
print(f"The most frequently visited host is {host_max}")
print("====================================================")

host_count = data.select('host','timestamp').groupBy(F.hour('time').alias('weekday')).count()

host_day_df = data.select(data.host, F.dayofweek('time').alias('day'))




from pyspark.sql.functions import udf

month_map = {'Jul':7}

def parse_clf_time(text):
    """ Convert Common Log time format into a Python datetime object
    Args:
        text (str): date and time in Apache time format [dd/mmm/yyyy:hh:mm:ss (+/-)zzzz]
    Returns:
        a string suitable for passing to CAST('timestamp')
    """
    # NOTE: We're ignoring the time zones here, might need to be handled depending on the problem you are solving
    return "{0:04d}-{1:02d}-{2:02d}".format(
      int(float(text[7:11])),
      int(float(month_map[text[3:6]])),
      int(float(text[0:2]))
    )

udf_parse_time = udf(parse_clf_time)

data = (data.select('*', udf_parse_time(data['timestamp'])
                                  .cast('timestamp')
                                  .alias('time'))
                  .drop('timestamp'))
        
hourly_avg_errors_sorted_df = (data
                                   .groupBy(F.dayofweek('time')
                                             .alias('weekday'))
                                   .count()
                                   .sort('weekday'))
