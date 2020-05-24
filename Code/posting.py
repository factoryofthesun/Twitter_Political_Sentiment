import tweepy
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

def makePost(biden_score, trump_score, msg, date):
    # Twitter authentication
    api_key = os.environ.get("TWITTER_API_KEY")
    api_secret_key = os.environ.get("TWITTER_API_SECRET_KEY")
    access_token = os.environ.get("TWITTER_ACCESS_TOKEN")
    secret_access_token = os.environ.get("TWITTER_SECRET_ACCESS_TOKEN")

    auth = tweepy.OAuthHandler(api_key, api_secret_key)
    auth.set_access_token(access_token, secret_access_token)

    api = tweepy.API(auth)
    try:
        api.verify_credentials()
        print("Authentication OK")
    except:
        print("Error during authentication")

    # Make post
    plotpath = str(Path(__file__).parent / "../Outputs/Plots")
    img = api.media_upload(f"{plotpath}/{date}_sentiment_plot.png")
    api.update_status(media_ids = [img.media_id_string], status = msg)

if __name__ == "__main__":
    # Read the date from arguments
    # Date format: YYYY-MM-DD
    date = sys.argv[1]

    # Run test post
    makePost(-1, 1, "test post", date)
