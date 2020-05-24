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

# Technically, we are analyzing yesterday's tweets
yesterday = date.today() - timedelta(days = 1)
yesterday_str = yesterday.strftime("%Y-%m-%d")

bidenSearch(yesterday_str)
trumpSearch(yesterday_str)
biden_score, trump_score = analyzeDay(yesterday_str)

msg = f"Twitter sentiment analysis for {yesterday_str}\nBiden score: {biden_score}\nTrump score: {trump_score}"

# Make post
makePost(biden_score, trump_score, msg, yesterday_str)
