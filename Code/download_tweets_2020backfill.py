#==================================
#Download 10k tweets per day up until current day
#Use this until download_tweets_day is properly set up
#==================================

import twint
import time
import os
from datetime import date, timedelta
from pathlib import Path
from time import sleep

path = str(Path(__file__).parent / "../Data")

#Initial Configuration - Basic Keyword Search, Tweet engagement values
#Get 10000 tweets per day
def bidenSearch():
    c = twint.Config()
    c.Search = "(biden OR @JoeBiden) -RT -filter:replies"
    c.Lang = "en"
    c.Store_csv = True
    c.Debug = True
    c.Update = True
    c.Resume = "biden_resume.log"
    c.Hide_output = True
    c.Limit = 10000
    c.Custom["tweet"] = ["id","date","username","tweet","likes_count", "retweets_count"]
    completed_dates = []
    try:
        with open(path + "/DiamondJoe/biden_finished_dates.txt", 'r') as f:
            lines = f.readlines()
            dates_to_skip = [line.rstrip() for line in lines]
        f.close()
    except:
        dates_to_skip = []
    today = date.today()
    start_date = date(2020,4,8) # RIP Bernard
    delta_days = today - start_date
    for i in range(delta_days.days):
        searched = 0
        search_date = start_date + timedelta(days = i)
        date_string = search_date.strftime("%Y-%m-%d")
        search_date_until = start_date + timedelta(days = i+1)
        search_date_until_string = search_date_until.strftime("%Y-%m-%d")
        if date_string in dates_to_skip:
            continue #Skip completed dates
        c.Since = date_string
        c.Until = search_date_until_string
        c.Output = path + "/DiamondJoe/biden_tweets_{}.csv".format(date_string)
        while searched < 1:
            try:
                twint.run.Search(c) #Search for biden data
                searched += 1
            except Exception as e:
                print(e, "Sleeping for 420 seconds...")
                sleep(420)
        completed_dates.append(date_string)
        os.remove("biden_resume.log")
    with open(path + "/DiamondJoe/biden_finished_dates.txt", 'a+') as f:
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
    c = twint.Config()
    c.Search = "(bernie OR @BernieSanders) -RT -filter:replies"
    c.Lang = "en"
    c.Store_csv = True
    c.Debug = True
    c.Update = True
    c.Resume = "bernie_resume.log"
    c.Hide_output = True
    c.Limit = 10000
    c.Custom["tweet"] = ["id","date","username","tweet","likes_count", "retweets_count"]
    completed_dates = []
    #Skip completed dates
    try:
        with open(path + "/Bernard/bernie_finished_dates.txt", 'r') as f:
            lines = f.readlines()
            dates_to_skip = [line.rstrip() for line in lines]
        f.close()
    except:
        dates_to_skip = []
    today = date.today()
    start_date = date(2020,1,1)
    delta_days = today - start_date
    for i in range(delta_days.days):
        searched = 0
        search_date = start_date + timedelta(days = i)
        date_string = search_date.strftime("%Y-%m-%d")
        search_date_until = start_date + timedelta(days = i+1)
        search_date_until_string = search_date_until.strftime("%Y-%m-%d")
        if date_string in dates_to_skip:
            continue #Skip completed dates
        c.Since = date_string
        c.Until = search_date_until_string
        c.Output = path + "/Bernard/bernie_tweets_{}.csv".format(date_string)
        while searched < 1:
            try:
                twint.run.Search(c)
                searched += 1
            except Exception as e:
                print(e, "Sleeping for 420 seconds...")
                sleep(420)
        completed_dates.append(date_string)
        os.remove("bernie_resume.log")
    with open(path + "/Bernard/bernie_finished_dates.txt", 'a+') as f:
        for d in completed_dates:
            f.write("{}\n".format(d))
    f.close()
    return True

def trumpSearch():
    c = twint.Config()
    c.Search = "(trump OR @realDonaldTrump) -RT -filter:replies"
    c.Lang = "en"
    c.Store_csv = True
    c.Debug = True
    c.Update = True
    c.Resume = "trump_resume.log"
    c.Hide_output = True
    c.Limit = 10000
    c.Custom["tweet"] = ["id","date","username","tweet","likes_count", "retweets_count"]
    completed_dates = []
    #Skip completed dates
    try:
        with open(path + "/Donald/trump_finished_dates.txt", 'r') as f:
            lines = f.readlines()
            dates_to_skip = [line.rstrip() for line in lines]
        f.close()
    except:
        dates_to_skip = []
    today = date.today()
    start_date = date(2020,4,8) # RIP Bernard
    delta_days = today - start_date
    for i in range(delta_days.days):
        searched = 0
        search_date = start_date + timedelta(days = i)
        date_string = search_date.strftime("%Y-%m-%d")
        search_date_until = start_date + timedelta(days = i+1)
        search_date_until_string = search_date_until.strftime("%Y-%m-%d")
        if date_string in dates_to_skip:
            continue #Skip completed dates
        c.Since = date_string
        c.Until = search_date_until_string
        c.Output = path + "/Donald/trump_tweets_{}.csv".format(date_string)
        while searched < 1:
            try:
                twint.run.Search(c)
                searched += 1
            except Exception as e:
                print(e, "Sleeping for 420 seconds...")
                sleep(420)
        completed_dates.append(date_string)
        os.remove("trump_resume.log")
    with open(path + "/Donald/trump_finished_dates.txt", 'a+') as f:
        for d in completed_dates:
            f.write("{}\n".format(d))
    f.close()
    return True

trumpSearch()
bidenSearch()
