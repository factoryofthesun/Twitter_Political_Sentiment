'''
Backfill and plot sentiment records
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from apply_sentiment import classifyDate
from pathlib import Path
import sys
import os
import glob
import re

outpath = str(Path(__file__).parent / "../Outputs")

records = pd.DataFrame(columns = ["Date", "Biden_Score", "Trump_Score"])
records = records.set_index("Date")

# Backfill by looping through Biden files - if Trump is missing for that date throw an error
for fpath in glob.iglob(f"{outpath}/Sentiment_Tagged/Biden/*.csv"):
    match = re.search('biden_(.*)_sentiment.csv', fpath)
    if match:
        date = match.group(1)
        biden_temp = pd.read_csv(fpath)
        try:
            trump_temp = pd.read_csv(f'{outpath}/Sentiment_Tagged/Trump/trump_{date}_sentiment.csv')
        except:
            raise Exception(f"Trump file missing for {date}!")
        biden_score = (biden_temp['Prediction'] * (biden_temp['Likes'] + biden_temp['Retweets'])).sum()/(biden_temp['Likes'] + biden_temp['Retweets']).sum()
        trump_score = (trump_temp['Prediction'] * (trump_temp['Likes'] + trump_temp['Retweets'])).sum()/(trump_temp['Likes'] + trump_temp['Retweets']).sum()
        scores_to_add = pd.Series({'Biden_Score': biden_score, 'Trump_Score': trump_score}, name = date)
        records = records.append(scores_to_add)
    else:
        raise Exception(f"{fpath} does not contain the proper naming convention!")

records.index = pd.to_datetime(records.index)
records.to_csv(f"{outpath}/sentiment_records.csv")

# Plot out average tweet sentiment by date
plotpath = str(Path(__file__).parent / "../Outputs/Plots")
plt.plot_date(x = records.index, y = records['Trump_Score'], fmt = 'o--r', xdate = True)
plt.plot_date(x = records.index, y = records['Biden_Score'], fmt = 'o--b', xdate = True)
plt.legend(['Trump', 'Biden'])
plt.xlabel("Date")
plt.ylabel("Likes/RTs Weighted Average Sentiment")
plt.title("Twitter Sentiment of Biden vs Trump")
plt.gcf().autofmt_xdate(rotation=25)
plt.savefig(f"{plotpath}/{date}_sentiment_plot.png") # Date here will be the last read date in the backfill
