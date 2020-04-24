#======================================
# Get tweets from specified date and writes to CSV
# Input: string date "YYYY-MM-DD"
#======================================

import twint
import time
from datetime import date
from pathlib import Path

path = str(Path(__file__).parent / "../Data")

#Initial Configuration - Basic Keyword Search, Tweet engagement values
#Get 10000 tweets per day
def bidenSearch(date):
    c = twint.Config()
    c.Search = "biden OR @JoeBiden -filter:replies"
    c.Lang = "en"
    c.Store_csv = True
    c.Output = path + "/DiamondJoe/biden_tweets_{}.csv".format(date)
    c.Since = date
    c.Debug = True
    c.Update = True
    c.Resume = "twint-request_urls.log"
    c.Until = date
    c.Hide_output = True
    c.Limit = 10000
    c.Custom["tweet"] = ["date","username","tweet","likes_count", "retweets_count"]
    twint.run.Search(c) #Search for biden data
    return True

def bernieSearch(date):
    c = twint.Config()
    c.Search = "bernie OR @BernieSanders -filter:replies"
    c.Lang = "en"
    c.Store_csv = True
    c.Output = path + "/Bernard/bernie_tweets_{}.csv".format(date)
    c.Since = date
    c.Until = date
    c.Debug = True
    c.Update = True
    c.Hide_output = True
    c.Limit = 10000
    c.Custom["tweet"] = ["date","username","tweet","likes_count", "retweets_count"]
    twint.run.Search(c) #Search for bernie data
    return True

def trumpSearch(date):
    c = twint.Config()
    c.Search = "trump OR @realDonaldTrump -filter:replies"
    c.Lang = "en"
    c.Store_csv = True
    c.Output = path + "/Donald/trump_tweets_{}.csv".format(date)
    c.Since = date
    c.Until = date
    c.Debug = True
    c.Update = True
    c.Hide_output = True
    c.Limit = 10000
    c.Custom["tweet"] = ["date","username","tweet","likes_count", "retweets_count"]
    twint.run.Search(c) #Search for bernie data
    return True

if __name__ == "__main__":
