import tweepy
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

def makePost(biden_score, trump_score, msg, date):
    # Twitter authentication

    api = tweepy.API(auth)

    # Make post
    plotpath = str(Path(__file__).parent / "../Outputs/Plots")
    img = api.media_upload(f"{plotpath}/{date}_sentiment_plot.png")
    api.update_status(media_ids = [img.media_id_string], status = msg)
    
