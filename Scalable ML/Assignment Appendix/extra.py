
from pyspark.sql.functions import udf

month_map = {
  'Jan': 1, 'Feb': 2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6, 'Jul':7,
  'Aug':8,  'Sep': 9, 'Oct':10, 'Nov': 11, 'Dec': 12
}

def parse_clf_time(text):
    """ Convert Common Log time format into a Python datetime object
    Args:
        text (str): date and time in Apache time format [dd/mmm/yyyy:hh:mm:ss (+/-)zzzz]
    Returns:
        a string suitable for passing to CAST('timestamp')
    """
    # NOTE: We're ignoring the time zones here, might need to be handled depending on the problem you are solving
    return "{0:04d}-{1:02d}-{2:02d} {3:02d}:{4:02d}:{5:02d}".format(
      int(text[7:11]),
      month_map[text[3:6]],
      int(text[0:2]),
      int(text[12:14]),
      int(text[15:17]),
      int(text[18:20])
    )

logspp.where(F.col('time').contains('-07-')).show()



udf_parse_time = udf(parse_clf_time)

logspp = (data.select('*', udf_parse_time(data['timestamp3'])
                                  .cast('timestamp')
                                  .alias('time'))
                  .drop('timestamp'))
          
logspp.show(10, truncate=True)

host_day_df = logspp.select(logspp.host, 
                             F.dayofweek('time').alias('day'))
host_day_df.show(5, truncate=False)
hourly_avg_errors_sorted_df = (data
                                   .groupBy(F.dayofweek('timestamp')
                                             .alias('weekday'))
                                   .count()
                                   .sort('weekday'))
hourly_avg_errors_sorted_df.show(5)







GET /shuttle/missions/sts-70/movies/woodpecker.mpg 
GET /shuttle/missions/sts-71/movies/crew-arrival-t38.mpg 
GET /shuttle/missions/sts-71/movies/sts-71-launch.mpg 
GET /shuttle/missions/sts-71/movies/sts-71-launch-3.mpg 
GET /shuttle/missions/sts-71/movies/sts-71-tcdt-crew-walkout.mpg 
GET /shuttle/missions/sts-71/movies/sts-71-mir-dock.mpg 
GET /history/apollo/apollo-13/movies/apo13damage.mpg 
GET /shuttle/missions/sts-70/movies/sts-70-launch.mpg 
GET /shuttle/missions/sts-71/movies/sts-71-mir-dock-2.mpg 
GET /shuttle/missions/sts-70/movies/sts-70-launch-srbsep.mpg 
GET /history/apollo/apollo-13/movies/apo13launch.mpg 
GET /shuttle/missions/sts-53/movies/sts-53-launch.mpg 
GET /shuttle/missions/sts-70/movies/sts-70-launch.mpg
HEAD /shuttle/missions/sts-71/movies/sts-71-launch-2.mpg
GET /shuttle/missions/sts-70/movies/sts-70-landing-approach.mpg
GET /shuttle/missions/sts-71/movies/sts-71-rollout.mpg 
GET /shuttle/countdown/lps/sts-71-s-5-i4.mpg HTTP/1.0 
GET /shuttle/countdown/lps/sts-71-s-5-i2.mpg HTTP/1.0 
GET /shuttle/missions/sts-71/movies/sts-71-undocking.mpg 
GET /shuttle/missions/sts-71/movies/sts-71-hand-shake.mpg 
GET /shuttle/missions/sts-71/movies/sts-71-launch-3.mpg
GET /shuttle/missions/sts-71/movies/sts-71-launch-2.mpg 
GET /wxworld/mpegs/MPEG6pNgmSfc9/95072712_48.mpg  
GET /shuttle/missions/sts-71/movies/sts-71-hatch-open.mpg        






from pyspark.sql.functions import udf

month_map = {'Jul':7}

def parse_clf_time(text):
    
    return "{0:04d}-{1:02d}-{2:02d}".format( int(text[7:11]),
      month_map[text[3:6]],
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
hourly_avg_errors_sorted_df.show(5)



















