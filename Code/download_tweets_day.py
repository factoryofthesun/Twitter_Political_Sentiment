#======================================
# Get tweets from specified date and writes to CSV
# Input: string date "YYYY-MM-DD"
#======================================

import twint
import time
from datetime import date
from pathlib import Path
import os
from time import sleep

path = str(Path(__file__).parent / "../Data")

#Initial Configuration - Basic Keyword Search, Tweet engagement values
#Get 10000 tweets per day
def bidenSearch(date):
    c = twint.Config()
    c.Search = "biden OR @JoeBiden -filter:replies -RT"
    c.Lang = "en"
    c.Store_csv = True
    c.Output = path + "/DiamondJoe/biden_tweets_{}.csv".format(date)
    c.Since = date
    c.Debug = True
    c.Update = True
    c.Resume = "biden_resume.log"
    c.Until = date
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
    #os.remove("biden_resume.log")
    with open(path + "/DiamondJoe/biden_finished_dates.txt", 'a+') as f:
        f.write(f"{date}\n")
    return True

def bernieSearch(date):
    c = twint.Config()
    c.Search = "bernie OR @BernieSanders -filter:replies -RT"
    c.Lang = "en"
    c.Store_csv = True
    c.Output = path + "/Bernard/bernie_tweets_{}.csv".format(date)
    c.Since = date
    c.Until = date
    c.Debug = True
    c.Resume = "bernie_resume.log"
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
    c = twint.Config()
    c.Search = "trump OR @realDonaldTrump -filter:replies -RT"
    c.Lang = "en"
    c.Store_csv = True
    c.Output = path + "/Donald/trump_tweets_{}.csv".format(date)
    c.Since = date
    c.Until = date
    c.Debug = True
    c.Resume = "trump_resume.log"
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
    #os.remove("trump_resume.log")
    with open(path + "/Donald/trump_finished_dates.txt", 'a+') as f:
        f.write(f"{date}\n")
    return True

if __name__ == "__main__":
    date = input("Please input a date to scrape: ")
    bidenSearch(date)
    trumpSearch(date)
