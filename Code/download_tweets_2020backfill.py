#==================================
#Download 10k tweets per day up until current day
#Use this until download_tweets_day is properly set up
#==================================

import twint
import time
from datetime import date, timedelta
from pathlib import Path

path = str(Path(__file__).parent / "../Data")

#Initial Configuration - Basic Keyword Search, Tweet engagement values
#Get 10000 tweets per day
def bidenSearch():
    completed_dates = []
    #Skip completed dates
    with open("biden_finished_dates.txt", 'r') as f:
        lines = f.readlines()
        dates_to_skip = [line.rstrip() for line in lines]
    f.close()
    today = date.today()
    start_date = date(2020,1,1)
    delta_days = today - start_date
    for i in range(delta_days.days + 1):
        search_date = start_date + timedelta(days = i)
        date_string = search_date.strftime("%Y-%m-%d")
        c = twint.Config()
        c.Search = "biden OR @JoeBiden -filter:replies"
        c.Lang = "en"
        c.Store_csv = True
        c.Output = path + "/DiamondJoe/biden_tweets_{}.csv".format(date_string)
        c.Since = date_string
        c.Debug = True
        c.Update = True
        c.Resume = "twint-request_urls.log"
        c.Hide_output = True
        c.Limit = 10000
        c.Custom["tweet"] = ["date","username","tweet","likes_count", "retweets_count"]
        twint.run.Search(c) #Search for biden data
        completed_dates.append(date_string)
    with open("biden_finished_dates.txt", 'a+') as f:
        for d in completed_dates:
            f.write("{}\n".format(d))
    f.close()
    return True

'''#Use exponential backoff retries
errors = 0
ret = False
while not ret:
    try:
        ret = bidenSearch()
    except:
        print("Errors {}, sleeping for {} seconds...".format(errors, 2**(errors)))
        errors += 1
        time.sleep(2**(errors)) #Wait for the IP ban to wear off'''


def bernieSearch():
    completed_dates = []
    #Skip completed dates
    with open("biden_finished_dates.txt", 'r') as f:
        lines = f.readlines()
        dates_to_skip = [line.rstrip() for line in lines]
    f.close()
    today = date.today()
    start_date = date(2020,1,1)
    delta_days = today - start_date
    for i in range(delta_days.days+1):
        search_date = start_date + timedelta(days = i)
        date_string = search_date.strftime("%Y-%m-%d")
        c = twint.Config()
        c.Search = "bernie OR @BernieSanders -filter:replies"
        c.Lang = "en"
        c.Store_csv = True
        c.Output = path + "/Bernard/bernie_tweets_{}.csv".format(date_string)
        c.Since = date_string
        c.Debug = True
        c.Update = True
        c.Resume = "twint-request_urls.log"
        c.Hide_output = True
        c.Limit = 10000
        c.Custom["tweet"] = ["date","username","tweet","likes_count", "retweets_count"]
        twint.run.Search(c) #Search for biden data
    with open("bernie_finished_dates.txt", 'a+') as f:
        for d in completed_dates:
            f.write("{}\n".format(d))
    f.close()
    return True

bernieSearch()
bidenSearch()
