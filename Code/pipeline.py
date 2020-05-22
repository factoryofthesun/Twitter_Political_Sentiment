'''
Script that consolidates the data pipeline for daily running. First download the tweets for the current day, then analyze them.
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from pathlib import Path
from datetime import date, timedelta, datetime
from download_tweets_day import bidenSearch, trumpSearch
from run_analysis import analyzeDay
from posting import makePost

today = date.today()
day_str = today.strftime("%Y-%m-%d")
bidenSearch(day_str)
trumpSearch(day_str)
biden_score, trump_score = analyzeDay(day_str)

# Technically, we are analyzing yesterday's tweets
d = datetime.strptime(date, "%Y-%m-%d")
yesterday = d - timedelta(days = 1)
yesterday_str = yesterday.strftime("%Y-%m-%d")
msg = f"Sentiment analysis for {yesterday_str} (-1 to 1):\nBiden score: {biden_score}\nTrump score: {trump_score}"
# Make post
makePost(biden_score, trump_score, msg, day_str)
