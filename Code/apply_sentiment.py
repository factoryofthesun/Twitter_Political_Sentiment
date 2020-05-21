import pandas as pd
import stanza
import torch
import numpy as np
import time
from pathlib import Path
from data_prep import process_data
import os
import glob
import re

# Input: date string ("YYYY-MM-DD")
# Returns: biden/trump sentiment classified dataframes
def classifyDate(date):
    #====================Entity-Level Sentiment Analysis=======================
    # 1) Select tweets by whether they only mention Trump, Biden, or both
    # 2) For single-entity tweets, "mask" the target entity and use the pre-trained model to classify sentiment
    #   - processData: generates auxiliary sentences per tweet
    path = str(Path(__file__).parent / "../Data")
    trump_sentiment_path = str(Path(__file__).parent / "../Outputs/Sentiment_Tagged/Trump")
    biden_sentiment_path = str(Path(__file__).parent / "../Outputs/Sentiment_Tagged/Biden")

    # If the classified file for the date already exists, then just read that file and return it
    if os.path.exists(f"{trump_sentiment_path}/trump_{date}_sentiment.csv") and os.path.exists(f"{biden_sentiment_path}/biden_{date}_sentiment.csv"):
        print(f"Sentiment classified file already exists for {date}!")
        trump_out = pd.read_csv(f"{trump_sentiment_path}/trump_{date}_sentiment.csv")
        biden_out = pd.read_csv(f"{biden_sentiment_path}/biden_{date}_sentiment.csv")
        return biden_out, trump_out

    # If there's a GPU available...
    if torch.cuda.is_available():

        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")

        print('There are %d GPU(s) available.' % torch.cuda.device_count())

        print('We will use the GPU:', torch.cuda.get_device_name(0))

    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    trump_tweets_df = pd.read_csv(path + f"/Donald/trump_tweets_{date}.csv")
    biden_tweets_df = pd.read_csv(path + f"/DiamondJoe/biden_tweets_{date}.csv")

    trump_tweets = trump_tweets_df['tweet']
    trump_ids = trump_tweets_df['id']
    trump_likes = trump_tweets_df['likes_count']
    trump_rt = trump_tweets_df['retweets_count']

    biden_tweets = biden_tweets_df['tweet']
    biden_ids = biden_tweets_df['id']
    biden_likes = biden_tweets_df['likes_count']
    biden_rt = biden_tweets_df['retweets_count']

    # Preprocess
    trump_aux_df = process_data(trump_tweets_df['tweet'], ['donald', 'trump'])
    biden_aux_df = process_data(biden_tweets_df['tweet'], ['biden', 'joe'])

    trump_s1 = trump_aux_df['tweet'].tolist()
    trump_s2 = trump_aux_df['aux'].tolist()

    biden_s1 = biden_aux_df['tweet'].tolist()
    biden_s2 = biden_aux_df['aux'].tolist()

    # Remove all links from text
    trump_s1 = [re.sub(r'https?:\/\/.*[\r\n\s]*', '', tweet, flags=re.MULTILINE) for tweet in trump_s1]
    biden_s1 = [re.sub(r'https?:\/\/.*[\r\n\s]*', '', tweet, flags=re.MULTILINE) for tweet in biden_s1]

    # Tokenize
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    # Get max length sequence
    maxlen = 0
    for i in range(len(trump_s1)):
        input_ids = tokenizer.encode(text = trump_s1[i], text_pair = trump_s2[i], add_special_tokens=True)
        maxlen = max(maxlen, len(input_ids))

    for i in range(len(biden_s1)):
        input_ids = tokenizer.encode(text = biden_s1[i], text_pair = biden_s2[i], add_special_tokens=True)
        maxlen = max(maxlen, len(input_ids))

    print("Maximum encoded sequence length:", maxlen)

    if maxlen > 512:
        print("Encoded tweet found that's over max length of 512! Please check for links.")
        maxlen = 512

    # Full encoding
    trump_input_ids = []
    trump_input_masks = []
    trump_token_ids = []

    biden_input_ids = []
    biden_input_masks = []
    biden_token_ids = []

    for i in range(len(trump_s1)):
        encoded_dict = tokenizer.encode_plus(text = trump_s1[i], text_pair = trump_s2[i], add_special_tokens=True,
                                        max_length = maxlen, pad_to_max_length=True, return_attention_mask=True,
                                        return_token_type_ids = True, return_tensors="pt")
        trump_input_ids.append(encoded_dict['input_ids'])
        trump_input_masks.append(encoded_dict['attention_mask'])
        trump_token_ids.append(encoded_dict['token_type_ids'])
    for i in range(len(biden_s1)):
        encoded_dict = tokenizer.encode_plus(text = biden_s1[i], text_pair = biden_s2[i], add_special_tokens=True,
                                        max_length = maxlen, pad_to_max_length=True, return_attention_mask=True,
                                        return_token_type_ids = True, return_tensors="pt")
        biden_input_ids.append(encoded_dict['input_ids'])
        biden_input_masks.append(encoded_dict['attention_mask'])
        biden_token_ids.append(encoded_dict['token_type_ids'])

    # Load NLP model
    from transformers import BertForSequenceClassification
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                          num_labels = 2,
                                                          output_attentions=False,
                                                          output_hidden_states=False)
    model.load_state_dict(torch.load("ML/Models/BERT_aux0.pt"))
    model.cuda()
    model.eval()

    # Convert list of tensors to tensors
    biden_input_tensor = torch.cat(biden_input_ids, dim=0)
    biden_mask_tensor = torch.cat(biden_input_masks, dim=0)
    biden_type_tensor = torch.cat(biden_token_ids, dim=0)

    trump_input_tensor = torch.cat(trump_input_ids, dim=0)
    trump_mask_tensor = torch.cat(trump_input_masks, dim=0)
    trump_type_tensor = torch.cat(trump_token_ids, dim=0)

    from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

    # Create dataloaders to batch predictions
    biden_dataloader = DataLoader(TensorDataset(biden_input_tensor, biden_mask_tensor, biden_type_tensor), batch_size = 24)
    trump_dataloader = DataLoader(TensorDataset(trump_input_tensor, trump_mask_tensor, trump_type_tensor), batch_size = 24)

    # Make predictions
    import torch.nn.functional as F

    t0 = time.time()
    biden_prob_pos = []
    for batch in biden_dataloader:
        input_ids = batch[0].to(device)
        input_mask = batch[1].to(device)
        input_token_types = batch[2].to(device)

        with torch.no_grad():
            outputs = model(input_ids, token_type_ids = input_token_types, attention_mask=input_mask)
        logits = outputs[0]
        probs = F.softmax(logits, dim=1)
        probs = probs.detach().cpu().numpy()
        biden_prob_pos.extend(probs[:,1])

    trump_prob_pos = []
    for batch in trump_dataloader:
        input_ids = batch[0].to(device)
        input_mask = batch[1].to(device)
        input_token_types = batch[2].to(device)

        with torch.no_grad():
            outputs = model(input_ids, token_type_ids = input_token_types, attention_mask=input_mask)
        logits = outputs[0]
        probs = F.softmax(logits, dim=1)
        probs = probs.detach().cpu().numpy()
        trump_prob_pos.extend(probs[:,1])

    tot_time = time.time() - t0
    print("Sentiment tagging took:", round(tot_time), "seconds")
    print("Trump tweets count:", len(trump_tweets))
    print("Biden tweets count:", len(biden_tweets))

    tmp = []
    preds = []
    targets = []
    prob_neg = []
    prob_neutral = []
    prob_positive = []
    for i in range(len(biden_prob_pos)):
        tmp.append(biden_prob_pos[i])
        if (i+1) % 3 == 0:
            preds.append(np.argmax(tmp)-1)
            prob_neg.append(tmp[0])
            prob_neutral.append(tmp[1])
            prob_positive.append(tmp[2])
            tmp = []

    biden_out = pd.DataFrame({'ID': biden_ids, "Date":date, "Tweet":biden_tweets, "Prob_Neg":prob_neg, "Prob_Neutral":prob_neutral,
                            "Prob_Positive":prob_positive, "Prediction":preds, "Likes": biden_likes, 'Retweets':biden_rt})

    biden_out.to_csv(biden_sentiment_path + f"/biden_{date}_sentiment.csv", index=False)

    avg_biden_sent = (biden_out["Prob_Neg"] * -1 + biden_out['Prob_Positive']).mean()
    print(f"Average biden sentiment on {date}:", avg_biden_sent)

    tmp = []
    preds = []
    targets = []
    prob_neg = []
    prob_neutral = []
    prob_positive = []
    for i in range(len(trump_prob_pos)):
        tmp.append(trump_prob_pos[i])
        if (i+1) % 3 == 0:
            preds.append(np.argmax(tmp)-1)
            prob_neg.append(tmp[0])
            prob_neutral.append(tmp[1])
            prob_positive.append(tmp[2])
            tmp = []

    trump_out = pd.DataFrame({"ID": trump_ids, "Date":date, "Tweet":trump_tweets, "Prob_Neg":prob_neg, "Prob_Neutral":prob_neutral,
                            "Prob_Positive":prob_positive, "Prediction":preds, "Likes":trump_likes, "Retweets":trump_rt})


    trump_out.to_csv(trump_sentiment_path + f"/trump_{date}_sentiment.csv", index=False)

    avg_trump_sent = (trump_out["Prob_Neg"] * -1 + trump_out['Prob_Positive']).mean()
    print(f"Average trump sentiment on {date}:", avg_trump_sent)

    return biden_out, trump_out



#====================Candidate Word Association=======================
# 1) Select tweets by whether they only mention Bernie, Biden, or both
# 2) If just one, then we can run the entire tweet through an out-of-the-box sentiment model
# 3) If both are mentioned, then we have a lot more work to do
#   a) First try: construct bag-of-words for each candidate using dependency parsing
# Save parsed text data in separate lists, then apply the sentiment classification

# == Filter out single-subject tweets first ==
'''trump_parsed_tweets = trump_tweets_df[~trump_tweets_df['tweet'].lower().contains("trump|donaldtrump"), 'tweet']
trump_parsed_ids = trump_tweets_df[~trump_tweets_df['tweet'].lower().contains("trump|donaldtrump"), 'id']
biden_parsed_tweets = biden_tweets_df[~biden_tweets_df['tweet'].lower().contains("biden|joebiden"), 'tweet']
biden_parsed_ids = biden_tweets_df[~biden_tweets_df['tweet'].lower().contains("biden|joebiden"), 'id']

print("Parsed {} out of {} total Trump tweets.".format(len(trump_parsed_ids), len(trump_tweets_df)))
print("Parsed {} out of {} total Biden tweets.".format(len(biden_parsed_ids), len(biden_tweets_df)))

# == Use dependency parsing to get bag-of-words for each candidate ==
#Stanford NLP dependency parser returns multi sentence input in following format
# [Sentence 1, Sentence 2, ...]
# Sentence: list of words
# Word: id, text, lemma, upos (universal pos), xpos (treebank-specific pos), head (id of syntactic head),
#       deprel (dependency relation of text to head)
#
depparser = stanza.Pipeline("en", processors="tokenize,mwt,pos,lemma,depparse", use_gpu = True)

# Input: tweet
# Outputs: trump-associated words, biden-associated words
def parseTweet(tweet):
    return

trump_remainder = trump_tweets_df[~trump_tweets_df.id.isin(trump_parsed_ids),["tweet", "id"]]
biden_remainder = biden_tweets_df[~biden_tweets_df.id.isin(biden_parsed_ids),["tweet", "id"]]

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

print(bernie_tweets[0])
print(bernie_words_sent)
print(biden_words_sent)'''

if __name__ == "__main__":
    # Apply sentiment for all the existing date files
    datapath =  str(Path(__file__).parent / "../Data")

    # Biden
    for fpath in glob.iglob(f"{datapath}/DiamondJoe/*.csv"):
        match = re.search('biden_tweets_(.*).csv', fpath)
        if match:
            classifyDate(match.group(1))
        else:
            raise Exception(f"{fpath} does not contain the proper naming convention!")

    # Trump
    for fpath in glob.iglob(f"{datapath}/Donald/*.csv"):
        match = re.search('trump_tweets_(.*).csv', fpath)
        if match:
            classifyDate(match.group(1))
    else:
        raise Exception(f"{fpath} does not contain the proper naming convention!")
