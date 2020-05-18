import sys

experiment = sys.argv[1]
hidden_units = int(sys.argv[2])

# —————————————————————————CSV—————————————————————————

import pandas as pd

# Training CSV
csv_train = pd.read_csv('stanford_sentiment_binary_train.tsv.gz',
                        sep='\t', header=0, compression='gzip')

# Development CSV
csv_dev = pd.read_csv('stanford_sentiment_binary_dev.tsv.gz',
                        sep='\t', header=0, compression='gzip')

# —————————————————————————FIELDS—————————————————————————

from torchtext.data import Field

# Lemmatized_POS Field
field_lemmatized = Field(sequential=True,
                         batch_first=True,
                         preprocessing=lambda x : [i.split("_")[0] for i in x])

# Labels Field
field_label = Field(sequential=False,
                    is_target=True,
                    batch_first=True,
                    preprocessing=lambda label: 0 if label=="negative" else 1,
                    use_vocab=False)

# Fields List
fields = [('sent_id', None),
          ('phrase_id', None),
          ('label', field_label),
          ('tokens', None),
          ('lemmatized', field_lemmatized)]

# —————————————————————————DATASETS—————————————————————————

from torchtext.data import Example

# Build list (collection) of 'Example' training instances:
train_examples = []
for index, entry in csv_train.iterrows():
    example = Example.fromlist(entry, fields)
    train_examples.append(example)

# Build list (collection) of 'Example' development instances:
dev_examples = []
for index, entry in csv_dev.iterrows():
    example = Example.fromlist(entry, fields)
    dev_examples.append(example)

from torchtext.data import Dataset

# Training Set (list of examples -> 'Dataset' instance)
train_set = Dataset(train_examples, fields)

print("Training Set: {} examples.".format(len(train_set)))

# Development Set (list of examples -> 'Dataset' instance)
dev_set = Dataset(dev_examples, fields)

print("Development Set: {} examples.".format(len(dev_set)))

# —————————————————————————VOCABULARY—————————————————————————

# Vectors
from torchtext.vocab import Vectors

# Word2Vec Embeddings
pretrained_embeddings_word2vec = Vectors('11.vec.gz')
# GloVe Embeddings
pretrained_embeddings_gloVe = Vectors('13.vec.gz')
# fastText Embeddings
pretrained_embeddings_fastText = Vectors('15.vec.gz')

# —————————————————————————ITERATORS—————————————————————————

from torchtext.data import Iterator

# Training Iterator
iterTrain = Iterator(train_set,
                     batch_size=32,
                     shuffle=True)

# Development Iterator
iterDev = Iterator(dev_set,
                   batch_size=1,
                   shuffle=False)

# —————————————————————————TRAINING FUNCTION—————————————————————————

def train_model(model, iterator, num_epochs):

    # Loss Function
    loss_fn = nn.CrossEntropyLoss()
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # Scheduler (Decreasing LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1)

    model.train()

    # Epoch Loop
    for epoch in range(num_epochs):
        epoch_loss = 0
        # Iterator Loop
        for dataBatch in iterator:
            input, label = dataBatch
            optimizer.zero_grad()           # Zero Gradient
            output = model(input)           # Forward Pass
            loss = loss_fn(output, label)   # Compute Loss
            loss.backward()                 # Backward Pass
            optimizer.step()                # Update Parameter
        scheduler.step()                    # Update Learning Rate

# —————————————————————————EVAL FUNCTION—————————————————————————

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def eval_model(model, iterator):

    model.eval()

    y_true = []
    y_pred = []

    for batch in iterator:
        input, label = batch
        pred = model(input).argmax()

        y_true.append(label.item())
        y_pred.append(pred.item())

    accuracy = round(accuracy_score(y_true, y_pred), 4)*100
    precision = round(precision_score(y_true, y_pred, average='macro'), 4)*100
    recall = round(recall_score(y_true, y_pred, average='macro'), 4)*100
    f1 = round(f1_score(y_true, y_pred, average='macro'), 4)*100

    print(f"Model: {model} \n")
    print(f"Accuracy: {accuracy}% \t| Precision: {precision} \t| Recall: {recall} \t| F1 Score: {f1} \n")

    return f1

# ————————————————————(6.2) HYPERPARAMETER TUNING—————————————————

import torch
import torch.nn.functional as F
from torch import nn, optim

torch.manual_seed(42)

class HyperparametersTuningCNN(nn.Module):

    """
    Hyperparameters Grid:

    — pretrained_embeddings : 'word2vec', 'GloVe', 'fastText'
    — hidden_layers : 1 or 2
    — kernel_size : 2, 3 or 4
    — output_channels : 50, 100 or 150
    — pooling : 'max' or 'mean'

    The total number of possible combinations is 108.

    """

    def __init__(self, pretrained_embeddings, hidden_layers, kernel_size, output_channels, pooling,
            hidden_dim=hidden_units, vocab_size=3000):
        super(HyperparametersTuningCNN, self).__init__()

        # Build Lemmatized_POS Vocab
        field_lemmatized.build_vocab(train_set, max_size=vocab_size, vectors=pretrained_embeddings) # Pretrained Embeddings
        # Build Labels Vocab
        field_label.build_vocab(train_set)

        # Embeddings Layer
        self.embeddings = nn.Embedding.from_pretrained(field_lemmatized.vocab.vectors)

        self.kernel_size = kernel_size
        # Convolutions (Filters) Layers
        self.conv_1 = nn.Conv1d(300, output_channels, 2)             # K = 2
        self.conv_2 = nn.Conv1d(300, output_channels, 3, padding=1)  # K = 3
        self.conv_3 = nn.Conv1d(300, output_channels, 4, padding=1)  # K = 4
        self.conv_4 = nn.Conv1d(300, output_channels, 5, padding=1)  # K = 5
        self.pooling = pooling

        # Dropout Layer
        self.dropout = nn.Dropout(p=0.5)

        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        # Fully Connected Layers
        self.linear_1 = nn.Linear(output_channels * (kernel_size - 1), self.hidden_dim)
        self.linear_2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear_3 = nn.Linear(self.hidden_dim, 2)

    def forward(self, input):

        # Embeddings Layer
        embeds = self.embeddings(input).transpose(1, 2)

        # Convolutions (Filters) Layers
        if self.pooling == 'max':
            if self.kernel_size == 2:
                conv_1 = F.relu(self.conv_1(embeds)).max(dim=-1)[0]
                # Concatenate Filters
                input = torch.cat([conv_1], dim=-1)
            elif self.kernel_size == 3:
                conv_1 = F.relu(self.conv_1(embeds)).max(dim=-1)[0]
                conv_2 = F.relu(self.conv_2(embeds)).max(dim=-1)[0]
                # Concatenate Filters
                input = torch.cat([conv_1, conv_2], dim=-1)
            elif self.kernel_size == 4:
                conv_1 = F.relu(self.conv_1(embeds)).max(dim=-1)[0]
                conv_2 = F.relu(self.conv_2(embeds)).max(dim=-1)[0]
                conv_3 = F.relu(self.conv_3(embeds)).max(dim=-1)[0]
                # concatenate filters
                input = torch.cat([conv_1, conv_2, conv_3], dim=-1)
            elif self.kernel_size == 5:
                conv_1 = F.relu(self.conv_1(embeds)).max(dim=-1)[0]
                conv_2 = F.relu(self.conv_2(embeds)).max(dim=-1)[0]
                conv_3 = F.relu(self.conv_3(embeds)).max(dim=-1)[0]
                conv_4 = F.relu(self.conv_3(embeds)).max(dim=-1)[0]
                # concatenate filters
                input = torch.cat([conv_1, conv_2, conv_3, conv_4], dim=-1)

        elif self.pooling == 'mean':
            if self.kernel_size == 2:
                conv_1 = F.relu(self.conv_1(embeds)).mean(dim=-1)
                # Concatenate Filters
                input = torch.cat([conv_1], dim=-1)
            elif self.kernel_size == 3:
                conv_1 = F.relu(self.conv_1(embeds)).mean(dim=-1)
                conv_2 = F.relu(self.conv_2(embeds)).mean(dim=-1)
                # Concatenate Filters
                input = torch.cat([conv_1, conv_2], dim=-1)
            elif self.kernel_size == 4:
                conv_1 = F.relu(self.conv_1(embeds)).mean(dim=-1)
                conv_2 = F.relu(self.conv_2(embeds)).mean(dim=-1)
                conv_3 = F.relu(self.conv_3(embeds)).mean(dim=-1)
                # Concatenate Filters
                input = torch.cat([conv_1, conv_2, conv_3], dim=-1)
            elif self.kernel_size == 5:
                conv_1 = F.relu(self.conv_1(embeds)).mean(dim=-1)
                conv_2 = F.relu(self.conv_2(embeds)).mean(dim=-1)
                conv_3 = F.relu(self.conv_3(embeds)).mean(dim=-1)
                conv_4 = F.relu(self.conv_3(embeds)).mean(dim=-1)
                # concatenate filters
                input = torch.cat([conv_1, conv_2, conv_3, conv_4], dim=-1)

        # Dropout Layer
        input = self.dropout(input)

        # Fully Connected Layers
        if self.hidden_layers == 1:
            hidden_1 = F.relu(self.linear_1(input))
            output = self.linear_2(hidden_1)

        elif self.hidden_layers == 2:
            hidden_1 = F.relu(self.linear_1(input))
            hidden_2 = F.relu(self.linear_2(hidden_1))
            output = self.linear_2(hidden_2)

        return output

# —————————————————————————EXPERIMENT Nº1—————————————————————————

#pretrained_embeddings = [pretrained_embeddings_word2vec,
#                         pretrained_embeddings_gloVe,
#                         pretrained_embeddings_fastText]
#hidden_layers = [1, 2]
#kernel_size = [2, 3, 4, 5]
#output_channels = [50, 100, 150, 200, 250]
#pooling = ['max', 'mean']
#
#models = []
#
#for pretrained in pretrained_embeddings:
#    for hidden in hidden_layers:
#        for kernel in kernel_size:
#            for channel in output_channels:
#                for pool in pooling:
#                    model = HyperparametersTuningCNN(pretrained, hidden, kernel, channel, pool)
#                    train_model(model, iterTrain, 10)
#                    score = eval_model(model, iterDev)
#
#                    if pretrained == pretrained_embeddings_word2vec:
#                        models.append(['word2vec', hidden, kernel, channel, pool, score])
#                    elif pretrained == pretrained_embeddings_gloVe:
#                        models.append(['GloVe', hidden, kernel, channel, pool, score])
#                    elif pretrained == pretrained_embeddings_fastText:
#                        models.append(['fastText', hidden, kernel, channel, pool, score])
#
#df = pd.DataFrame(models, columns=['Embeddings', 'Nº Layers', 'Nº Filters', 'Nº Channels', 'Pooling', 'F1 Score'])
#
#print(df)
#
#df.to_csv(experiment + ".csv")

# ————————————————————————FINE TUNING——————————————————————

# Best Model (from Grid Search): Hidden Units: 150, Embeddings: word2vec, Nº Layers: 1, Nº Channels:
# 150, Pooling: 'max'.

# models = []

# output_channels = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]
# vocab_sizes = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

# for channel in output_channels:

#    CNN = HyperparametersTuningCNN(pretrained_embeddings_word2vec, 1, 4, channel, 'max', vocab_size=V)
#    train_model(CNN, iterTrain, 10)
#    score = eval_model(CNN, iterDev)
    
#    models.append(hidden_units, channel, vocab_size, score)

# df = pd.DataFrame(models, columns=['Hidden Units', 'Nº Channels', 'Vocab Size', 'F1 Score'])

# print(df)

# ———————————————————————EXPERIMENT Nº2 AND Nº3———————————————————————

class HyperparametersTuningCNN(nn.Module):

    def __init__(self, hidden_dim=400):
        super(HyperparametersTuningCNN, self).__init__()
  
        # Build Lemmatized_POS Vocab


        # Embeddings Layer
        self.embeddings = nn.Embedding.from_pretrained(field_review.vocab.vectors)
        self.output_channels=[250,450,450]

        # Convolutions (Filters) Layers
        self.conv_1 = nn.Conv1d(300, self.output_channels[0], 2)             # K = 2
        self.conv_2 = nn.Conv1d(300, self.output_channels[1], 3, padding=1)  # K = 3
        self.conv_3 = nn.Conv1d(300, self.output_channels[2], 4, padding=1)  # K = 4


        
        # Fully Connected Layers
        self.linear_1 = nn.Linear(sum(self.output_channels), hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_3 = nn.Linear(hidden_dim, 2)

    def forward(self, input):

        # Embeddings Layer
        embeds = self.embeddings(input).transpose(1, 2)

        # Convolutions (Filters) Layers



        conv_1 = F.relu(self.conv_1(embeds)).max(dim=-1)[0]
        conv_2 = F.relu(self.conv_2(embeds)).max(dim=-1)[0]
        conv_3 = F.relu(self.conv_3(embeds)).max(dim=-1)[0]
        # Concatenate Filters
        input = torch.cat([conv_1, conv_2, conv_3], dim=-1)

 


        hidden_1 = F.relu(self.linear_1(input))
        output = self.linear_2(hidden_1)


        return output

# Grid Search
embeddings = pretrained_embeddings_word2vec
                         

model = HyperparametersTuningCNN()
train_model(model, iterTrain,30)
score = eval_model(model, iterDev)
#######################################

models = []
hidden_neurons=400
vocabulary=10000
channels=[250,450,450]


fields=["word2vec"]
fields.extend(channels)
fields.append(vocabulary )
fields.append(score)
models.append(fields)
df = pd.DataFrame(models, columns=['Embeddings', 'C1', 'C2', 'C3',"vocabulary" ,'F1_Score'])
torch.save(model, 'model.pt')


print(df)
print("channels",channels)
print("experiment", experiment )
df.to_csv(experiment + ".csv")
