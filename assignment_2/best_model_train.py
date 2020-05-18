#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn.functional as F
from torch import nn, optim

torch.manual_seed(42)

# ————————————————————CSV————————————————————

import pandas as pd

# Training CSV
csv_train = pd.read_csv('stanford_sentiment_binary_train.tsv.gz',
                       sep='\t', header=0, compression='gzip')

# Development CSV
csv_dev = pd.read_csv('stanford_sentiment_binary_dev.tsv.gz',
                       sep='\t', header=0, compression='gzip')

# Full CSV
csv_train = pd.concat([csv_train, csv_dev])

# ————————————————————FIELDS————————————————————

from torchtext.data import Field

# Labels Field
field_label = Field(sequential=False,
                    is_target=True,
                    batch_first=True,
                    preprocessing=lambda label: 0 if label=="negative" else 1,
                    use_vocab=False)

# Lemmatized_POS Field
field_lemmatized = Field(sequential=True,
                         batch_first=True,
                         preprocessing=lambda x : [i.split("_")[0] for i in x])

# Fields List
fields = [('sent_id', None),
          ('phrase_id', None),
          ('label', field_label),
          ('tokens', None),
          ('lemmatized', field_lemmatized)]

# ————————————————————DATASET————————————————————

from torchtext.data import Example

# Build list (collection) of 'Example' training instances:
train_examples = []
for index, entry in csv_train.iterrows():
    example = Example.fromlist(entry, fields)
    train_examples.append(example)

from torchtext.data import Dataset

# Development Set (list of examples -> 'Dataset' instance)
train_set = Dataset(train_examples, fields)

print("Test Set: {} examples.".format(len(train_set)))

from torchtext.data import Iterator

# Build Iterator
iterTrain = Iterator(train_set,
                     batch_size=32,
                     shuffle=True)

# ————————————————————VOCABS————————————————————

# Vectors
from torchtext.vocab import Vectors
# Gigaword (Lemmatized, No-POS-Tag)
word2vec = Vectors('11.vec.gz')         # word2vec (11.zip)

# Build Lemmatized Vocab
field_lemmatized.build_vocab(train_set, max_size=9000, vectors=word2vec)
# Build Labels Vocab
field_label.build_vocab(train_set)

# ————————————————————NETWORK————————————————————

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        # Embeddings Layer
        self.embeddings = nn.Embedding(9002, 300)
        self.embeddings.weight.data.copy_(field_lemmatized.vocab.vectors)

        # Convolutions (Filters) Layers
        self.conv_1 = nn.Conv1d(300, 250, 2)             # K = 2
        self.conv_2 = nn.Conv1d(300, 250, 3, padding=1)  # K = 3
        self.conv_3 = nn.Conv1d(300, 250, 4, padding=1)  # K = 4

        # Dropout Layer
        self.dropout = nn.Dropout(p=0.5)

        # Fully Connected Layers
        self.linear_1 = nn.Linear(750, 150)
        self.linear_2 = nn.Linear(150, 2)

    def forward(self, input):

        # Embeddings Layer
        with torch.no_grad():
            embeds = self.embeddings(input).transpose(1, 2)

        # Convolutions (Filters) Layers
        conv_1 = F.relu(self.conv_1(embeds)).max(dim=-1)[0]
        conv_2 = F.relu(self.conv_2(embeds)).max(dim=-1)[0]
        conv_3 = F.relu(self.conv_3(embeds)).max(dim=-1)[0]
        # Concatenate filters
        input = torch.cat([conv_1, conv_2, conv_3], dim=-1)

        # Dropout Layer
        input = self.dropout(input)

        # Fully Connected Layers
        hidden_1 = F.relu(self.linear_1(input))
        output = self.linear_2(hidden_1)

        return output

# ————————————————————TRAINING————————————————————

from helpers import train_model

model = CNN()

train_model(model, iterTrain, 50)

# ————————————————————SAVING————————————————————

# Vectorizer
import pickle

with open('fields.bin', 'wb') as f:
    pickle.dump((field_lemmatized.vocab, field_label.vocab), f)

# PyTorch Model
torch.save(model, 'cnn_best_model.pt')
