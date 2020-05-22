#======================================
# Get tweets from specified date and writes to CSV
# Input: string date "YYYY-MM-DD"
#======================================

import twint
import time
from datetime import date, timedelta, datetime
from pathlib import Path
import os
from time import sleep

#Initial Configuration - Basic Keyword Search, Tweet engagement values
#Get 10000 tweets per day
def bidenSearch(date):
    path = str(Path(__file__).parent / "../Data")
    d = datetime.strptime(date, "%Y-%m-%d")
    yesterday = d - timedelta(days = 1)
    yesterday_str = yesterday.strftime("%Y-%m-%d")

    c = twint.Config()
    c.Search = "biden OR @JoeBiden -filter:replies"
    c.Lang = "en"
    c.Filter_retweets = True
    c.Store_csv = True
    c.Output = path + "/DiamondJoe/biden_tweets_{}.csv".format(date)
    c.Since = yesterday_str
    c.Until = date
    c.Debug = False
    c.Update = True
    c.Resume = "D:/Code Projects/Twitter sentiment/Code/biden_resume.log"
    c.Hide_output = True
    c.Limit = 10000
    c.Custom["tweet"] = ["id","date","username","tweet","likes_count", "retweets_count"]
    searched = 0
    while searched < 1:
        try:
            twint.run.Search(c)
            searched += 1
        except Exception as e:
            print(e, "Sleeping for 420 seconds...")
            sleep(420)
    os.remove("D:/Code Projects/Twitter sentiment/Code/biden_resume.log")
    with open(path + "/DiamondJoe/biden_finished_dates.txt", 'a+') as f:
        f.write(f"{date}\n")
    return True

def bernieSearch(date):
    path = str(Path(__file__).parent / "../Data")
    d = datetime.strptime(date, "%Y-%m-%d")
    yesterday = d - timedelta(days = 1)
    yesterday_str = yesterday.strftime("%Y-%m-%d")

    c = twint.Config()
    c.Search = "bernie OR @BernieSanders -filter:replies -RT"
    c.Lang = "en"
    c.Store_csv = True
    c.Output = path + "/Bernard/bernie_tweets_{}.csv".format(date)
    c.Since = yesterday_str
    c.Until = date
    c.Debug = False
    c.Resume = "D:/Code Projects/Twitter sentiment/Code/bernie_resume.log"
    c.Update = True
    c.Hide_output = True
    c.Limit = 10000
    c.Custom["tweet"] = ["id","date","username","tweet","likes_count", "retweets_count"]
    searched = 0
    while searched < 1:
        try:
            twint.run.Search(c)
            searched += 1
        except Exception as e:
            print(e, "Sleeping for 420 seconds...")
            sleep(420)
    return True

def trumpSearch(date):
    path = str(Path(__file__).parent / "../Data")
    d = datetime.strptime(date, "%Y-%m-%d")
    yesterday = d - timedelta(days = 1)
    yesterday_str = yesterday.strftime("%Y-%m-%d")

    c = twint.Config()
    c.Search = "trump OR @realDonaldTrump -filter:replies"
    c.Lang = "en"
    c.Store_csv = True
    c.Filter_retweets = True
    c.Output = path + "/Donald/trump_tweets_{}.csv".format(date)
    c.Since = yesterday_str
    c.Until = date
    c.Debug = False
    c.Resume = "D:/Code Projects/Twitter sentiment/Code/trump_resume.log"
    c.Update = True
    c.Hide_output = True
    c.Limit = 10000
    c.Custom["tweet"] = ["id","date","username","tweet","likes_count", "retweets_count"]
    searched = 0
    while searched < 1:
        try:
            twint.run.Search(c)
            searched += 1
        except Exception as e:
            print(e, "Sleeping for 420 seconds...")
            sleep(420)
    os.remove("D:/Code Projects/Twitter sentiment/Code/trump_resume.log")
    with open(path + "/Donald/trump_finished_dates.txt", 'a+') as f:
        f.write(f"{date}\n")
    return True

if __name__ == "__main__":
    date = input("Please input a date to scrape: ")
    bidenSearch(date)
    trumpSearch(date)
