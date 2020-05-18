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
                         tokenize=None)

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

pretrained_embeddings_Wikipedia = Vectors('200.vec.gz')
pretrained_embeddings_GigaWorld = Vectors('29.vec.gz')
pretrained_embeddings_GoogleNews = Vectors('/Users/X/Documents/model.txt')
# We used GoogleNews embeddings locally

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

import matplotlib.pyplot as plt
from tqdm import tqdm

def train_model(model, iterator, num_epochs):

    # Loss Function
    loss_fn = nn.CrossEntropyLoss()
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Scheduler (Decreasing LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1)

    # List with each Epoch's Loss
    loss_values = []

    model.train()

    # Epoch Loop
    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0
        # Iterator Loop
        for dataBatch in iterator:
            input, label = dataBatch
            optimizer.zero_grad()           # Zero Gradient
            output = model(input)           # Forward Pass
            loss = loss_fn(output, label)   # Compute Loss
            loss.backward()                 # Backward Pass
            optimizer.step()                # Update Parameter

            # Update Epoch's Loss
            epoch_loss += loss.item()

        scheduler.step()                    # Update Learning Rate

        # Append Epoch's Loss to List with Losses
        loss_values.append(epoch_loss)

        print("Epoch: {}/{}\tLoss:{}".format(epoch, num_epochs, loss.item()))

    print(loss_values)

    plt.plot(range(num_epochs), loss_values)

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

# —————————————————————————(5) FFNN—————————————————————————

import torch
import torch.nn.functional as F
from torch import nn, optim

torch.manual_seed(42)

# Build Lemmatized_POS Vocab
field_lemmatized.build_vocab(train_set, max_size=3000, vectors=pretrained_embeddings_GigaWorld) # Pretrained Embeddings: Wikipedia
# Build Labels Vocab
field_label.build_vocab(train_set)

class TwoHiddenLayersFFNN(nn.Module):

    def __init__(self, fingerprint=0, freeze_embed=True):

        super(TwoHiddenLayersFFNN, self).__init__()

        # Embeddings Layer
        self._embed = nn.Embedding.from_pretrained(field_lemmatized.vocab.vectors, freeze=freeze_embed)

        # Dropout Layer
        self.dropout = nn.Dropout(p=0.5)

        # Fingerprint
        self.fingerprint = fingerprint

        # Fully Connected Layers
        self._linear1 = nn.Linear(300, 128)
        self._linear2 = nn.Linear(128, 128)
        self._linear3 = nn.Linear(128, 2)

    def forward(self, batch):

        # Embeddings Layer
        embeds = self._embed(batch)

        # Semantic Fingerprint
        if self.fingerprint == 0:
            input = embeds.sum(dim=1)   # Sum
        elif self.fingerprint == 1:
            input = embeds.mean(dim=1)  # Mean

        # Dropout
        input = self.dropout(input)

        # Fully Connected Layers
        hidden_1 = F.relu(self._linear1(input))
        hidden_2 = F.relu(self._linear2(hidden_1))
        output = self._linear3(hidden_2)

        return output

train_model(FFNN_sum_freeze, iterTrain, 30)
train_model(FFNN_mean_freeze, iterTrain, 30)
train_model(FFNN_sum_unfreeze, iterTrain, 30)
train_model(FFNN_mean_unfreeze, iterTrain, 30)

# —————————————————————————(6.1) CNN—————————————————————————

# Build Lemmatized_POS Vocab
field_lemmatized.build_vocab(train_set, max_size=3000, vectors=pretrained_embeddings_GoogleNews) # Pretrained Embeddings
# Build Labels Vocab
field_label.build_vocab(train_set)

class BaselineCNN(nn.Module):

    def __init__(self, output_channels=100, hidden_dim=128):
        super(BaselineCNN, self).__init__()

        # Embeddings Layer
        self.embeddings = nn.Embedding.from_pretrained(field_lemmatized.vocab.vectors)

        # Convolutions (Filters) Layers
        self.conv_1 = nn.Conv1d(300, output_channels, 3, padding=1)  # K = 3
        self.conv_2 = nn.Conv1d(300, output_channels, 4, padding=1)  # K = 4
        self.conv_3 = nn.Conv1d(300, output_channels, 5, padding=2)  # K = 5

        # Dropout Layer
        self.dropout = nn.Dropout(p=0.5)

        # Fully Connected Layers
        self.linear_1 = nn.Linear(output_channels * 3, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_3 = nn.Linear(hidden_dim, 2)

    def forward(self, input):

        # Embeddings Layer
        embeds = self.embeddings(input).transpose(1, 2)

        # Convolutions (Filters) Layers
        conv_1 = F.relu(self.conv_1(embeds)).max(dim=-1)[0] # ReLU & Max Pooling
        conv_2 = F.relu(self.conv_2(embeds)).max(dim=-1)[0] # ReLU & Max Pooling
        conv_3 = F.relu(self.conv_3(embeds)).max(dim=-1)[0] # ReLU & Max Pooling

        # Concatenate Filters
        input = torch.cat([conv_1, conv_2, conv_3], dim=-1)

        # Dropout Layer
        input = self.dropout(input)

        # Fully Connected Layers
        hidden_1 = F.relu(self.linear_1(input))        # ReLU
        hidden_2 = F.relu(self.linear_2(hidden_1))     # ReLU
        output = self.linear_2(hidden_2)

        return output

baselineCNN = BaselineCNN()

train_model(baselineCNN, iterTrain, 20)

eval_model(baselineCNN, iterDev)

# —————————————————————————(6.2) CNN—————————————————————————

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

    def __init__(self, pretrained_embeddings, hidden_layers, kernel_size, output_channels, pooling, hidden_dim=128):
        super(HyperparametersTuningCNN, self).__init__()

        # Build Lemmatized_POS Vocab
        field_lemmatized.build_vocab(train_set, max_size=3000, vectors=pretrained_embeddings) # Pretrained Embeddings
        # Build Labels Vocab
        field_label.build_vocab(train_set)

        # Embeddings Layer
        self.embeddings = nn.Embedding.from_pretrained(field_lemmatized.vocab.vectors)

        self.kernel_size = kernel_size
        # Convolutions (Filters) Layers
        self.conv_1 = nn.Conv1d(300, output_channels, 2)             # K = 2
        self.conv_2 = nn.Conv1d(300, output_channels, 3, padding=1)  # K = 3
        self.conv_3 = nn.Conv1d(300, output_channels, 4, padding=1)  # K = 4
        self.pooling = pooling

        self.hidden_layers = hidden_layers
        # Fully Connected Layers
        self.linear_1 = nn.Linear(output_channels * (kernel_size - 1), hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_3 = nn.Linear(hidden_dim, 2)

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
                conv_2 = F.relu(self.conv_1(embeds)).max(dim=-1)[0]
                conv_3 = F.relu(self.conv_1(embeds)).max(dim=-1)[0]
                # Concatenate Filters
                input = torch.cat([conv_1, conv_2, conv_3], dim=-1)

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
                conv_2 = F.relu(self.conv_1(embeds)).mean(dim=-1)
                conv_3 = F.relu(self.conv_1(embeds)).mean(dim=-1)
                # Concatenate Filters
                input = torch.cat([conv_1, conv_2, conv_3], dim=-1)

        # Fully Connected Layers
        if self.hidden_layers == 1:
            hidden_1 = F.relu(self.linear_1(input))
            output = self.linear_2(hidden_1)

        elif self.hidden_layers == 2:
            hidden_1 = F.relu(self.linear_1(input))
            hidden_2 = F.relu(self.linear_2(hidden_1))
            output = self.linear_2(hidden_2)

        return output

# Grid Search
pretrained_embeddings = [pretrained_embeddings_Wikipedia, pretrained_embeddings_GigaWorld, pretrained_embeddings_GoogleNews]
hidden_layers = [1, 2]
kernel_size = [2, 3, 4]
output_channels = [50, 100, 150]
pooling = ['max', 'mean']

models = []
f1_scores = []

for pretrained in pretrained_embeddings:
    for hidden in hidden_layers:
        for kernel in kernel_size:
            for channel in output_channels:
                for pool in pooling:
                    model = HyperparametersTuningCNN(pretrained, hidden, kernel, channel, pool)
                    train_model(model, iterTrain, 30)
                    score = eval_model(model, iterDev)
                    models.append([str(pretrained), hidden, kernel, channel, pool])
                    f1_scores.append(score)
