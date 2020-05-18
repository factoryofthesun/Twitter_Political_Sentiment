'''
In this script I take the raw twitter annotated data from https://www.aclweb.org/anthology/P14-2009.pdf
and generate sentence-pairs as dictated by the QA-B process in https://arxiv.org/pdf/1903.09588.pdf.
'''

import pandas as pd
import os
import re

# Input: Series of tweets
# Returns: Dataframe with 3 auxiliary sentences for each tweet
def process_data(sr, subject_words):
    tweets = []
    aux = []
    for i in range(len(sr)):
        redata = re.compile('|'.join(subject_words), re.IGNORECASE)
        masked_text = redata.sub('$T$', sr[i])
        tweets.extend([masked_text] * 3)
        for word in ["negative", "none", "positive"]:
            aux.append(f"the polarity of $T$ is {word}")
    ret_df = pd.DataFrame({'tweet':tweets, 'aux':aux})
    return ret_df

if __name__ == "__main__":
    '''
    Every instance is 3 lines
    sentence
    target
    polarity
    '''

    train_sentence1 = []
    train_sentence2 = []
    train_target = []

    with open("Data/target_twitter_data/train.raw", "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
        for i in range(len(lines)):
            if i%3 == 0:
                train_sentence1.extend([lines[i]] * 3)
                polarity = lines[i+2]

                # Generate targets depending on true polarity
                if polarity == "1": train_target.extend([0,0,1])
                elif polarity == "-1": train_target.extend([1,0,0])
                else: train_target.extend([0,1,0])

                # Generate auxiliary sentences
                for word in ["negative", "none", "positive"]:
                    sentence2 = "the polarity of $T$ is {}".format(word)
                    train_sentence2.append(sentence2)

    train_ids = range(1, len(train_sentence1) + 1)
    assert len(train_sentence1) == len(train_sentence2) == len(train_target)

    train_df = pd.DataFrame({'id':train_ids, 'sentence1':train_sentence1, 'sentence2':train_sentence2, 'target':train_target})
    train_df.to_csv("Data/train.csv", index=False)

    test_sentence1 = []
    test_sentence2 = []
    test_target = []

    with open("Data/target_twitter_data/test.raw", "r",encoding="utf-8") as f:
        lines = f.read().splitlines()
        for i in range(len(lines)):
            if i%3 == 0:
                test_sentence1.extend([lines[i]] * 3)
                polarity = lines[i+2]

                # Generate targets depending on true polarity
                if polarity == "1": test_target.extend([0,0,1])
                elif polarity == "-1": test_target.extend([1,0,0])
                else: test_target.extend([0,1,0])

                # Generate auxiliary sentences
                for word in ["negative", "none", "positive"]:
                    sentence2 = "the polarity of $T$ is {}".format(word)
                    test_sentence2.append(sentence2)

    test_ids = range(1, len(test_sentence1) + 1)
    assert len(test_sentence1) == len(test_sentence2) == len(test_target)

    test_df = pd.DataFrame({'id':test_ids, 'sentence1':test_sentence1, 'sentence2':test_sentence2, 'target':test_target})
    test_df.to_csv("Data/test.csv", index=False)
