'''
Backfill and plot sentiment records
'''

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from apply_sentiment import classifyDate
from pathlib import Path
from datetime import datetime as dt
import sys
import os
import glob
import re

outpath = str(Path(__file__).parent / "../Outputs")
plotpath = str(Path(__file__).parent / "../Outputs/Plots")

records = pd.DataFrame(columns = ["Date", "Biden_Score", "Trump_Score"])
records = records.set_index("Date")
records.index = pd.to_datetime(records.index)
dates = []
# Backfill by looping through Biden files - if Trump is missing for that date throw an error
for fpath in glob.iglob(f"{outpath}/Sentiment_Tagged/Biden/*.csv"):
    match = re.search('biden_(.*)_sentiment.csv', fpath)
    if match:
        date = match.group(1)
        dates.append(date)
        biden_temp = pd.read_csv(fpath)
        try:
            trump_temp = pd.read_csv(f'{outpath}/Sentiment_Tagged/Trump/trump_{date}_sentiment.csv')
        except:
            raise Exception(f"Trump file missing for {date}!")
        date_strp = dt.strptime(date, '%Y-%m-%d')
        date_strp = date_strp.date()
        biden_score = (biden_temp['Prediction'] * (biden_temp['Likes'] + biden_temp['Retweets'])).sum()/(biden_temp['Likes'] + biden_temp['Retweets']).sum()
        trump_score = (trump_temp['Prediction'] * (trump_temp['Likes'] + trump_temp['Retweets'])).sum()/(trump_temp['Likes'] + trump_temp['Retweets']).sum()
        scores_to_add = pd.Series({'Biden_Score': biden_score, 'Trump_Score': trump_score}, name = date_strp)
        records = records.append(scores_to_add)

        # Make separate plot for each date
        fig, ax = plt.subplots()
        plt.plot_date(x = records.index, y = records['Trump_Score'], fmt = 'o--r', markersize=5, xdate = True)
        plt.plot_date(x = records.index, y = records['Biden_Score'], fmt = 'o--b', markersize=5, xdate = True)

        # Format date tick marks - Month for major tick, week for minor ticks
        months = mdates.MonthLocator()
        weeks = mdates.WeekdayLocator()
        days = mdates.DayLocator()
        label_fmt = mdates.DateFormatter("%m-%d-%y")

        # Set axis formats
        ax.xaxis.set_major_locator(weeks)
        ax.xaxis.set_major_formatter(label_fmt)
        ax.xaxis.set_minor_locator(days)
        datemin =  np.datetime64(records.index[0], 'D') - np.timedelta64(7, 'D')
        datemax = np.datetime64(records.index[-1], 'D') + np.timedelta64(7, 'D')
        ax.set_xlim(datemin, datemax)

        plt.legend(['Trump', 'Biden'])
        plt.xlabel("Date")
        plt.ylabel("Likes/RTs Weighted Average Sentiment")
        plt.title("Daily Twitter Sentiment of Biden vs Trump")
        plt.gcf().autofmt_xdate(rotation=25)
        plt.savefig(f"{plotpath}/{date}_sentiment_plot.png")
        plt.close()
    else:
        raise Exception(f"{fpath} does not contain the proper naming convention!")

records.to_csv(f"{outpath}/sentiment_records.csv")
