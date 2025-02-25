import json
import pickle
import time
import datetime
import random
import os
import csv

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
# from seqeval.metrics import f1_score

import matplotlib.pyplot as plt

device = torch.device("cpu")

SEED_VAL = 42

random.seed(SEED_VAL)
np.random.seed(SEED_VAL)
torch.manual_seed(SEED_VAL);

SNIPS_PATH = "../../data/snips"
TRAIN_PATH = f"{SNIPS_PATH}/train.tsv"
VAL_PATH = f"{SNIPS_PATH}/dev.tsv"
TEST_PATH = f"{SNIPS_PATH}/test.tsv"
df = pd.read_csv(TEST_PATH,sep='\t')

def load_snips_file(file_path):
    list_pair =[]
    with open(file_path,'r',encoding="utf8") as f:
        for line in f:
            split_line = line.split('\t')
            pair = split_line[0],split_line[1]
            list_pair.append(pair)
    return list_pair

all_train_examples = load_snips_file(TRAIN_PATH)
valid_examples = load_snips_file(VAL_PATH)
test_examples = load_snips_file(TEST_PATH)

intents = np.unique(np.array(all_train_examples)[:,0]).tolist()

intent_labeltoid = {intents[i]: i  for i in range(len(intents))}

intent_series = pd.Series(np.array(all_train_examples)[:,0])
intent_series.value_counts()

def create_mini_training_set(examples_per_intent):
    intent_array = np.array(all_train_examples)[:,0]
    mini_batch =[]
    for intent in intents:
        add = intent_array[intent_array==intent]
        shuffled_indicies=np.random.RandomState(seed=42).permutation(len(add))
        class_indicies=shuffled_indicies[:examples_per_intent]
        sampled_set = np.array(all_train_examples)[class_indicies]
        mini_batch.append(sampled_set)
    mini_batch = np.array(mini_batch)
    mini_set = mini_batch.transpose(1,0,2).reshape(-1,mini_batch.shape[2])
    return mini_set

import re
def get_pad_length():
    all_train_examples_sentences = np.array(all_train_examples)[:,1]
    word_length = []
    for sentence in all_train_examples_sentences:
        number_words = len(re.findall(r'\w+',sentence))
        word_length.append(number_words)
    return max(word_length)

PAD_LEN = get_pad_length()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

test_utterance = "Alyssa told Ben the error of his ways"

print(tokenizer.encode_plus(
            test_utterance, add_special_tokens=True, max_length=PAD_LEN, pad_to_max_length=True,
            truncation=True, return_attention_mask=True, return_tensors='pt'
    ))


def examples_to_dataset(examples):
    input_ids = []
    attention_masks = []
    labels = []
    for instance in examples:
        token_dict = tokenizer.encode_plus(
            instance[1], add_special_tokens=True, max_length=PAD_LEN, pad_to_max_length=True,
            truncation=True, return_attention_mask=True, return_tensors='pt')
        input_ids.append(token_dict['input_ids'])
        attention_masks.append(token_dict['attention_mask'])
        labels.append(torch.tensor(intent_labeltoid[instance[0]]).type(torch.LongTensor))

    input_ids = torch.cat(input_ids)
    attention_masks = torch.cat(attention_masks)
    labels = torch.stack(labels)

    dataset = TensorDataset(input_ids, attention_masks, labels)

    return dataset



BATCH_SIZE = 50


def get_accuracy(preds, labels):
    pred_convd = np.argmax(preds,1).flatten()
    labels_flat = labels.flatten()
    correct_labels = np.equal(pred_convd,labels_flat).sum()
    accuracy_value = correct_labels/len(labels)
    return accuracy_value


def evaluate(model, dataloader):
    model.eval()

    accuracy = []

    for batch in tqdm(list(dataloader)):
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            (loss, logits) = model(b_input_ids,
                                   token_type_ids=None,
                                   attention_mask=b_input_mask,
                                   labels=b_labels, return_dict=False)

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # logit_probability =torch.nn.Softmax(logits)
        batch_accuracy = get_accuracy(logits, label_ids)
        accuracy.append(batch_accuracy)
    avg_accuracy = np.mean(accuracy)  # TODO Compute final accuracy
    print("Validation Accuracy: {}".format(avg_accuracy))
    return avg_accuracy


def train(model, dataloader, epochs):
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    for epoch_i in range(0, EPOCHS):
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))

        model.train()
        # n_iteration = 0
        accuracy = []
        total_train_loss = []

        for step, batch in tqdm(list(enumerate(dataloader))):
            # get input IDs, input mask, and labels from batch
            b_input_ids, b_input_mask, b_labels = batch

            model.zero_grad()
            # pass inputs through model
            (loss, logits) = model(b_input_ids,
                                   token_type_ids=None,
                                   attention_mask=b_input_mask,
                                   labels=b_labels, return_dict=False)
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            # Add to total_train_loss
            total_train_loss.append(loss.detach())
            # logit_probability =torch.nn.Softmax(logits)
            batch_accuracy = get_accuracy(logits, label_ids)
            accuracy.append(batch_accuracy)
            # n_iteration += 1
        # Compute average train loss
        new_loss = [x.cpu().detach().numpy() for x in total_train_loss]
        avg_train_loss = np.mean(new_loss)
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Average Training accuracy: {0:.2f}".format(np.mean(accuracy)))
    # validation_accuracy =evaluate(bert_model, validation_dataloader)

BATCH_SIZE = 16
INTENT_DIM = 7
EPOCHS = 5
EXAMPLES_PER_INTENT = 250

# save intents to id mapping
with open('intent2id.json', 'w') as file:
    json.dump(intent_labeltoid, file)

mini_train_set = examples_to_dataset(create_mini_training_set(EXAMPLES_PER_INTENT))

train_dataloader = DataLoader(mini_train_set, sampler=RandomSampler(mini_train_set), batch_size=BATCH_SIZE)

bert_model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = INTENT_DIM,
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)

train(bert_model, train_dataloader, EPOCHS)
torch.save(bert_model.state_dict(), './trained_model.pt')
bert_model.save_pretrained('/home/ankur/dev/apps/ML/learn/ray/serve/test/intent_prediction', from_pt=True)

val_dataset = examples_to_dataset(valid_examples)
test_dataset = examples_to_dataset(test_examples)

validation_dataloader = DataLoader(val_dataset, sampler=RandomSampler(val_dataset), batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=BATCH_SIZE)

print("Evaluating on test set:")
print("Test accuracy:", evaluate(bert_model, test_dataloader))