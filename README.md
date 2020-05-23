# Twitter_Political_Sentiment
Sentiment analysis of 2020 Presidential Candidates on Twitter

This is a personal project where I scrape twitter (using twint) on a daily basis for tweets that mention either Joe Biden or Donald Trump, classify their sentiment on each of the candidates using a pre-trained ABSA model as described [in this paper](https://www.aclweb.org/anthology/N19-1035.pdf). This data is what is fed into the daily posts made by @PotusSentiBot. 

## Implementation Details 
* \# Tweets Scraped: 10k for each candidate 
* Sentiment scoring: every tweet assigned {-1,0,1} for negative, neutral, positive, respectively. The final daily score is the daily average of the tweets for each candidate, **weighted** by likes + retweets 
* Model training: data for training and testing sourced from the paper [Adaptive Recursive Neural Network for Target-dependent Twitter Sentiment Classification](https://www.aclweb.org/anthology/P14-2009.pdf) 
