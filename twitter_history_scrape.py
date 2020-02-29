import pandas as pd
import tweepy
import scrapy
import regex as re
import time
import json
import os
import sys
from datetime import datetime as dt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

#Generate monthly national heat maps going back a year for general twitter sentiment on bernie
analyzer = SentimentIntensityAnalyzer()

#Datetime values
now = dt.now()
current_month = now.month
start_date = now.replace(year = now.year - 1, day = 1)
end_date = now.replace(day = 1)

#Use keys if using API or tweepy
ckey=""
csecret=""
atoken=""
asecret=""

#Use Selenium to use Twitter's advanced search, then scroll through the page until you hit the bottom
#Collect all info in a pandas csv
search_link = f"https://twitter.com/search?q={must_contain}%20\"{strict_phrase}\"%20({any_of})%20{none_of}%20({these_hashtags})%20lang%3Aen%20until%3A{until_date}%20since%3A{since_date}&src=typed_query"

must_contain_list = ['bernie']
must_contain = "%20".join(must_contain_list) #list to string using proper formatting
strict_phrase = ""
any_of_list = ['sanders']
any_of = "%20OR%20".join(any_of_list) #list to string
none_of_list = []
none_of = " -" + " -".join(none_of_list) #list to string
these_hashtags_list = []
these_hashtags = "%23" + "%20OR%20%23".join(these_hashtags_list) #list to string
until_date = end_date.strftime("%Y-%m-%d")
since_date = start_date.strftime("%Y-%m-%d")
