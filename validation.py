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

intent_labeltoid = json.load(open('intent2id.json'))

bert_model = BertForSequenceClassification.from_pretrained("/home/ankur/dev/apps/ML/learn/ray/serve/test/intent_prediction")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

PAD_LEN = 45

def examples_to_dataset(examples):
    input_ids = []
    attention_masks = []
    labels = []
    for instance in examples[0:200]:
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

def get_accuracy(preds, labels):
    pred_convd = np.argmax(preds,1).flatten()
    labels_flat = labels.flatten()
    correct_labels = np.equal(pred_convd,labels_flat).sum()
    accuracy_value = correct_labels/len(labels)
    return accuracy_value


def evaluate(model, dataloader):
    model.eval()

    accuracy = []
    start = time.time()
    for batch in tqdm(list(dataloader)):
        b_input_ids, b_input_mask, b_labels = batch
        start_ = time.time()
        with torch.no_grad():
            logits = model(b_input_ids,
                                   token_type_ids=None,
                                   attention_mask=b_input_mask,
                                    return_dict=False)[0]
        print('batch execution time: %.4f' % (time.time() - start_))
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # logit_probability =torch.nn.Softmax(logits)
        batch_accuracy = get_accuracy(logits, label_ids)
        accuracy.append(batch_accuracy)
    avg_accuracy = np.mean(accuracy)  # TODO Compute final accuracy
    print('execution time: %.4f' % (time.time() - start))
    print("Validation Accuracy: {}".format(avg_accuracy))
    return avg_accuracy

SNIPS_PATH = "../../data/snips"
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


test_examples = load_snips_file(TEST_PATH)
test_dataset = examples_to_dataset(test_examples)
BATCH_SIZE = 50
test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=BATCH_SIZE)
num_threads = 2
torch.set_num_threads(num_threads)
torch.set_num_interop_threads(num_threads)
print("Evaluating on test set:")
print("Test accuracy:", evaluate(bert_model, test_dataloader))