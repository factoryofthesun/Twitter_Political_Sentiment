'''
Main program to run daily for sentiment analysis and plot generation
TODO: Make this a twitter bot
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from apply_sentiment import classifyDate
import sys
import os

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

    scores_to_add = pd.Series({'Biden_Score': biden_score, 'Trump_Score': trump_score}, name = date)

    outpath = str(Path(__file__).parent / "../Outputs")
    if os.path.exists(outpath + "/sentiment_records.csv"):
        records = pd.read_csv(outpath + "/sentiment_records.csv")
        records = records.append(scores_to_add)
    else:
        records = pd.DataFrame(columns = ["Date", "Biden_Score", "Trump_Score"])
        records = records.set_index("Date") # Set date as index
        records = records.append(scores_to_add)

    records.to_csv(f"{outpath}/sentiment_records.csv")

    # Plot out average tweet sentiment by date
    plotpath = str(Path(__file__).parent / "../Outputs/Plots")
    plt.plot(records['Date'], records['Trump_Score'], 'o--r')
    plt.plot(records['Date'], records['Biden_Score'], 'o--b')
    plt.legend(['Trump', 'Biden'])
    plt.xlabel("Date")
    plt.ylabel("Likes/RTs Weighted Average Sentiment")
    plt.title("Twitter Sentiment of Biden vs Trump")
    plt.savefig(f"{plotpath}/{date}_sentiment_plot.png")

if __name__ == "__main__":
    # Read the date from arguments
    # Date format: YYYY-MM-DD
    date = sys.argv[1]
    analyzeDay(date)
