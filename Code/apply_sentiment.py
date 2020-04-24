import pandas as pd
import stanza
from pathlib import Path

path = str(Path(__file__).parent / "../Data")

bernie_tweets_df = pd.read_csv(path + "/bernie_tweets.csv")
bernie_tweets = bernie_tweets_df['tweet']

#Stanford NLP dependency parser returns multi sentence input in following format
# [Sentence 1, Sentence 2, ...]
# Sentence: list of words
# Word: id, text, lemma, upos (universal pos), xpos (treebank-specific pos), head (id of syntactic head),
#       deprel (dependency relation of text to head)
#
depparser = stanza.Pipeline("en", processors="tokenize,mwt,pos,lemma,depparse", use_gpu = True)
test = depparser(bernie_tweets[0])

#====================Candidate Word Association=======================
# 1) Demarcate tweets by whether they only mention Bernie, Biden, or both
# 2) If just one, then we can take all the subtrees of pronouns
# 2a) Compute word dependency distances and gather all the words surrounding the aspect word (Biden/Bernie)
# 3) If both are mentioned, then will need to figure out way to allocate words 
#Collect all words/ids related to "Bernie" and "Biden"; no overlap!
bernie_words_sent = []
biden_words_sent = []

for sent in test.sentences:
    bernie_words = []
    biden_words = []
    for word in sent.words:
        if word.text.lower() in ['bernie', 'sanders', 'berniesanders']:
            if word.head == 0:
                continue
            else: #Get word associated with head id
                head_pos = word.head - 1
                head_text = sent.words[head_pos].text
                bernie_words.append(head_text)
        elif word.text.lower() in ['biden', 'joebiden']:
            if word.head == 0:
                continue
            else: #Get word associated with head id
                head_pos = word.head - 1
                head_text = sent.words[head_pos].text
                biden_words.append(head_text)
        elif word.head == 0:
            continue
        elif sent.words[word.head-1].text in ['bernie', 'sanders', 'berniesanders']:
            bernie_words.append(word.text)
        elif sent.words[word.head-1].text in ['biden', 'joebiden']:
            biden_words.append(word.text)
        else:
            continue
    bernie_words_sent.append(bernie_words)
    biden_words_sent.append(biden_words)

print(bernie_words_sent)
print(biden_words_sent)

#Apply bag-of-words
