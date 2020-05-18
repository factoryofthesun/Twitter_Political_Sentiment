'''
We will be adapting the auxiliary-sentence method QA-B from this paper: https://arxiv.org/pdf/1903.09588.pdf
and fine-tune the HuggingFace BertForSequenceClassification model.
Since we are performing entity-level sentiment classification, our data-preprocessing will be much simpler.
Params:
    - # Targets to consider: 1 (written as $T$ in the data)
    - # Auxiliary sentences for each target: 3 (Positive, Negative, Neutral)

We will use manually-annotated twitter target dataset from: https://www.aclweb.org/anthology/P14-2009.pdf
'''

from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange
import tensorflow as tf
import torch
import pandas as pd
import numpy as np
import time
import datetime
import random
from pathlib import Path

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

data_path = str(Path(__file__).parent / "../Data")
# Load data
train_df = pd.read_csv(data_path + "/train.csv")
test_df = pd.read_csv(data_path + "/test.csv")

train_labels = train_df['target'].tolist()
test_labels = test_df['target'].tolist()

train_s1 = train_df['sentence1'].tolist()
train_s2 = train_df['sentence2'].tolist()

test_s1 = test_df['sentence1'].tolist()
test_s2 = test_df['sentence2'].tolist()

print("Training set size:", len(train_s1)/3)
print("Test set size:", len(test_s1)/3)

# Preprocessing
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

# Find longest sequence to set MAXLEN to
maxlen = 0

for i in range(len(train_s1)):
    input_ids = tokenizer.encode(text = train_s1[i], text_pair= train_s2[i], add_special_tokens=True)
    if i == 0:
        print(tokenizer.decode(input_ids))
    maxlen = max(maxlen, len(input_ids))

for i in range(len(test_s1)):
    input_ids = tokenizer.encode(text = test_s1[i], text_pair= test_s2[i], add_special_tokens=True)
    maxlen = max(maxlen, len(input_ids))

print("Maximum encoded sequence length:", maxlen)


# Use encode_plus to set max length and return tensors
train_input_ids = []
train_attention_masks = []
train_token_types = []

for i in range(len(train_s1)):
    encoded_dict = tokenizer.encode_plus(text=train_s1[i], text_pair=train_s2[i],
                                        add_special_tokens=True, max_length=maxlen,
                                        pad_to_max_length=True, return_attention_mask=True,
                                        return_token_type_ids = True, return_tensors="pt")
    train_input_ids.append(encoded_dict['input_ids'])
    train_attention_masks.append(encoded_dict['attention_mask'])
    train_token_types.append(encoded_dict['token_type_ids']) # Identifying sentence 1 vs sentence 2

test_input_ids = []
test_attention_masks = []
test_token_types = []

for i in range(len(test_s1)):
    encoded_dict = tokenizer.encode_plus(text=test_s1[i], text_pair=test_s2[i],
                                        add_special_tokens=True, max_length=maxlen,
                                        pad_to_max_length=True, return_attention_mask=True,
                                        return_token_type_ids = True, return_tensors="pt")
    test_input_ids.append(encoded_dict['input_ids'])
    test_attention_masks.append(encoded_dict['attention_mask'])
    test_token_types.append(encoded_dict['token_type_ids']) # Identifying sentence 1 vs sentence 2

# Convert list of tensors to tensors
train_input_ids_tensor = torch.cat(train_input_ids, dim=0)
train_attention_mask_tensor = torch.cat(train_attention_masks,dim=0)
train_token_types_tensor = torch.cat(train_token_types, dim = 0)

test_input_ids_tensor = torch.cat(test_input_ids, dim=0)
test_attention_mask_tensor = torch.cat(test_attention_masks,dim=0)
test_token_types_tensor = torch.cat(test_token_types, dim=0)

train_label_tensor = torch.tensor(train_labels)
test_label_tensor = torch.tensor(test_labels)


from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, random_split

# Split training dataset into train and validation
dataset = TensorDataset(train_input_ids_tensor, train_attention_mask_tensor, train_token_types_tensor, train_label_tensor)
train_size = int(0.9*len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

test_dataset = TensorDataset(test_input_ids_tensor, test_attention_mask_tensor, test_token_types_tensor, test_label_tensor)

# Batch size according to authors
batch_size = 24

train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
validation_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size)

from transformers import BertForSequenceClassification, AdamW, BertConfig

model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels = 2,
                                                      output_attentions=False,
                                                      output_hidden_states=False)

model.cuda()

#Define helper functions
#Accuracy function
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def train(model, optimizer, scheduler, train_dataloader, val_dataloader, epochs):
  #Record loss and accuracy
  train_loss = []
  val_loss = []
  train_acc = []
  val_acc = []
  t_start = time.time()

  for epoch_i in range(epochs):
    #====================
    #     Training
    #====================
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')
    t0 = time.time()
    total_loss = 0
    total_accuracy = 0
    steps = 0
    model.train() #Put into training mode

    for step, batch in enumerate(train_dataloader):
      #Progress update every 5 batches
      if step % 5 == 0 and not step == 0:
        elapsed = format_time(time.time() - t0)
        print('  Batch {}  of  {}.    Elapsed: {}.'.format(step, len(train_dataloader), elapsed))

      #Unpack batch and copy tensors to GPU
      b_input_ids = batch[0].to(device)
      b_input_mask = batch[1].to(device)
      b_token_types = batch[2].to(device)
      b_labels = batch[3].to(device)

      #Make sure to zero out the gradient
      model.zero_grad()

      #Perform forward pass
      outputs = model(b_input_ids,
                      token_type_ids=b_token_types,
                      attention_mask=b_input_mask,
                      labels=b_labels)

      #Get loss
      loss = outputs[0]
      total_loss += loss.item()
      logits = outputs[1] #Logits are the output values prior to applying activation function

      #Move logits and labels to CPU - can't perform accuracy calculation on GPU
      logits = logits.detach().cpu().numpy()
      label_ids = b_labels.to('cpu').numpy()

      #Calculate batch accuracy
      tmp_accuracy = flat_accuracy(logits, label_ids)
      total_accuracy += tmp_accuracy
      steps += 1

      #Backward pass to calculate gradients
      loss.backward()

      #Clip norms to prevent "exploding gradients"
      # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

      optimizer.step() #Update parameters using optimizer
      scheduler.step() #Update learning rate

    #Average loss/accuracy
    avg_loss = total_loss/steps
    avg_acc = total_accuracy/steps

    train_loss.append(avg_loss)
    train_acc.append(avg_acc)
    print("")
    print("  Average training loss: {0:.2f}".format(avg_loss))
    print("  Average training accuracy: {0:.2f}".format(avg_acc))
    print("  Training epoch took: {:}".format(format_time(time.time() - t0)))

    #====================
    #     Validation
    #====================
    print("")
    print("Running validation...")
    t0 = time.time()
    total_loss = 0
    total_accuracy = 0
    steps = 0

    model.eval() #Evaluation mode

    for batch in val_dataloader:
      #Unpack batch and copy tensors to GPU
      b_input_ids = batch[0].to(device)
      b_input_mask = batch[1].to(device)
      b_token_types = batch[2].to(device)
      b_labels = batch[3].to(device)

      #Tell model not to store gradients - speeds up testing
      with torch.no_grad():
        outputs = model(b_input_ids,
                        token_type_ids=b_token_types,
                        attention_mask=b_input_mask,
                        labels = b_labels)

      #Get loss
      loss = outputs[0]
      total_loss += loss.item()
      logits = outputs[1] #Logits are the output values prior to applying activation function

      #Move logits and labels to CPU - can't perform accuracy calculation on GPU
      logits = logits.detach().cpu().numpy()
      label_ids = b_labels.to('cpu').numpy()

      #Calculate batch accuracy
      tmp_accuracy = flat_accuracy(logits, label_ids)
      total_accuracy += tmp_accuracy

      steps += 1 #Track batch num
    #Report the final accuracy for this validation run.
    val_loss.append(total_loss/steps)
    val_acc.append(total_accuracy/steps)
    print("  Average validation loss: {0:.2f}".format(total_loss/steps))
    print("  Average validation accuracy: {0:.2f}".format(total_accuracy/steps))
    print("  Validation took: {:}".format(format_time(time.time() - t0)))

  print("")
  print("Training complete!")
  print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-t_start)))

  return train_loss, train_acc, val_loss, val_acc

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plots_path = str(Path(__file__).parent / "../Plots/Twitter")
# Plot results
def plotTrainResults(train_loss, train_acc, val_loss, val_acc):
    loss_fname = "loss_model0.png"
    acc_fname = "accuracy_model0.png"

    plt.plot(train_loss, 'r--')
    plt.plot(val_loss, 'b-')
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.title("Loss by Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(plots_path + '/' + loss_fname)
    plt.clf()

    plt.plot(train_acc, 'r--')
    plt.plot(val_acc, 'b-')
    plt.legend(['Training Accuracy', 'Validation Accuracy'])
    plt.title("Accuracy by Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(plots_path + '/' + acc_fname)
    plt.clf()

# == Set hyperparameters and run training ==
from transformers import get_linear_schedule_with_warmup

# Parameters according to authors
optimizer = AdamW(model.parameters(),
                  lr = 2e-5,
                  eps = 1e-8)

epochs = 5 # Training accuracy hits 100% here
total_steps = len(train_dataloader) * epochs

# Linear schedule = triangular learning rate
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

train_loss, train_acc, val_loss, val_acc = train(model, optimizer, scheduler, train_dataloader, validation_dataloader, epochs)
plotTrainResults(train_loss, train_acc, val_loss, val_acc)

# Test model
import torch.nn.functional as F

print("Predicting on test set of size {}".format(len(test_dataloader)))

model.eval()

# Sentiment prediction is just: argmax(prob_neg, prob_neutral, prob_pos)
prob_pos = []
for batch in test_dataloader:
    #Unpack batch and copy tensors to GPU
    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)
    b_token_types = batch[2].to(device)
    b_labels = batch[3].to(device)

    #Tell model not to store gradients - speeds up testing
    with torch.no_grad():
        outputs = model(b_input_ids,
                      token_type_ids=b_token_types,
                      attention_mask=b_input_mask)
    logits = outputs[0]
    probs = F.softmax(logits, dim = 1)
    probs = probs.detach().cpu().numpy()
    prob_pos.extend(probs[:,1])

test_sentences = test_s1[0::3]
print("Test tweets count:", len(test_sentences))

tmp = []
preds = []
targets = []
prob_neg = []
prob_neutral = []
prob_positive = []
for i in range(len(prob_pos)):
    tmp.append(prob_pos[i])
    if (i+1) % 3 == 0:
        preds.append(np.argmax(tmp)-1)
        targets.append(np.argmax(test_labels[i-2:(i+1)])-1)
        prob_neg.append(tmp[0])
        prob_neutral.append(tmp[1])
        prob_positive.append(tmp[2])
        tmp = []

assert len(preds) == len(test_sentences) == len(targets) == len(prob_neg) == len(prob_neutral) == len(prob_positive)

test_out = pd.DataFrame({"Tweet":test_sentences, "Prob_Neg":prob_neg, "Prob_Neutral":prob_neutral,
                        "Prob_Positive":prob_positive, "Prediction":preds, "Target":targets})

accuracy = sum(test_out['Prediction']==test_out['Target'])/len(preds)
print("Test accuracy:", accuracy)

# Save outputs
test_out.to_csv("Outputs/pred_out.csv", index=False)
torch.save(model.state_dict(),  "Models/BERT_aux0.pt")
