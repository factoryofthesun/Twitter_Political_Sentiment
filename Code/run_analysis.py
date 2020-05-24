'''
Main program to run daily for sentiment analysis and plot generation
'''
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import sys
import os
from datetime import datetime as dt
from apply_sentiment import classifyDate


def analyzeDay(date):
    # Try applying sentiment if the dated files exist
    try:
        biden_sent_df, trump_sent_df = classifyDate(date)
    except:
        raise Exception(f"Sentiment classification failed for the date {date}! Have you checked if the data exists?")

    # Write sentiment score to records file
    from pathlib import Path

    # Compute retweets/likes-weighted average of sentiment
    biden_score = (biden_sent_df['Prediction'] * (biden_sent_df['Likes'] + biden_sent_df['Retweets'])).sum()/(biden_sent_df['Likes'] + biden_sent_df['Retweets']).sum()
    trump_score = (trump_sent_df['Prediction'] * (trump_sent_df['Likes'] + trump_sent_df['Retweets'])).sum()/(trump_sent_df['Likes'] + trump_sent_df['Retweets']).sum()

    date_strp = dt.strptime(date, '%Y-%m-%d')
    date_strp = date_strp.date()
    scores_to_add = pd.Series({'Biden_Score': biden_score, 'Trump_Score': trump_score}, name = date_strp)

    outpath = str(Path(__file__).parent / "../Outputs")
    if os.path.exists(outpath + "/sentiment_records.csv"):
        records = pd.read_csv(outpath + "/sentiment_records.csv", index_col="Date")
    else:
        records = pd.DataFrame(columns = ["Date", "Biden_Score", "Trump_Score"])
        records = records.set_index("Date")
        records.index = pd.to_datetime(records.index)

    if date_strp not in records.index:
        records = records.append(scores_to_add)
        records.to_csv(f"{outpath}/sentiment_records.csv")

    # Plot out average tweet sentiment by date
    plotpath = str(Path(__file__).parent / "../Outputs/Plots")
    records.index = pd.to_datetime(records.index)

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
    return biden_score, trump_score
if __name__ == "__main__":
    # Read the date from arguments
    # Date format: YYYY-MM-DD
    date = sys.argv[1]
    analyzeDay(date)
