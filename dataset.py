import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import os
import numpy as np
from nltk import word_tokenize, ngrams
import nltk
import time
from tqdm import tqdm
import torchtext
from torchtext.vocab import GloVe
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import regex as re
import math
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from dataset import *
from datasets import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = load_dataset("sst", "default")
data_file = open("data.txt", "w")

train_data = dataset["train"]
test_data = dataset["test"]
dev_data = dataset["validation"]

def preprocess(sentence):
    # We are avoiding URLs, hashtags, mentions, numbers, punctuations
    sentence = re.sub(r"(?:\@|https?\://)\S+", "", sentence)
    sentence = re.sub(r"\d+", "", sentence)
    sentence = re.sub(r"[^\w\s]", "", sentence)
    sentence = sentence.lower()
    return sentence

def deal_data(data):
    cleaned_sentences = []
    for sentence in tqdm(data):
        cleaned_sentences.append(preprocess(sentence))
    return cleaned_sentences

train_sentences_clean = deal_data(train_data["sentence"])
test_sentences_clean = deal_data(test_data["sentence"])
dev_sentences_clean = deal_data(dev_data["sentence"])

glove_embeddings = []

def get_glove_embeddings():
    embeddings_index = {}
    glove = torchtext.vocab.GloVe(name="6B", dim=200) # basically a dictionary
    for word in glove:
        embeddings_index[word] = glove[word]
        glove_embeddings.append(glove[word])
    return embeddings_index

embeddings_index = get_glove_embeddings()
glove_embeddings = np.array(glove_embeddings)

count = 0
for keys in embeddings_index.keys():
    count += 1
    print(keys, embeddings_index[keys])
    if count == 10:
        break
    

# intialize random glove embeddings to sos eos unk and pad

word2idx = {}
idx2word = {}







