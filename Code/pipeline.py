'''
Script that consolidates the data pipeline. First download the tweets for the current day, then analyze them.
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from pathlib import Path
from download_tweets_day import bidenSearch, trumpSearch
from run_analysis import analyzeDay
from datetime import date

today = day.today()
day_str = today.strftime("%Y-%m-%d")
bidenSearch(day_str)
trumpSearch(day_str)
analyzeDay(day_str)
