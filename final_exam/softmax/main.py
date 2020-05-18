# ------------------------------IMPORTS----------------------------
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

from torchtext.data import Field, Dataset, Iterator, Example
from torchtext.vocab import Vectors

from helpers import best_model, count_parameters, train_model, eval_model

from sklearn.metrics import classification_report
import pandas as pd
import pickle

torch.manual_seed(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ------------------------------HYPERPARAMETERS------------------------------

if __name__ == "__main__":

    # ------------------------------DATA------------------------------
    train_data =  pd.read_csv("train_data.csv")
    val_data =  pd.read_csv("val_data.csv")
    test_data =  pd.read_csv("test_data.csv") # index_col=0

    for dataFrame in [train_data, val_data, test_data]:
        # Remove B-MISC and I-MISC tags
        dataFrame['entity'] = dataFrame['entity'].str.replace('B-MISC','O')
        dataFrame['entity'] = dataFrame['entity'].str.replace('I-MISC','O')
        # Merge GPE_ORG and GPE_LOC into GPE
        dataFrame['entity'] = dataFrame['entity'].str.replace('B-GPE_ORG','B-GPE')
        dataFrame['entity'] = dataFrame['entity'].str.replace('I-GPE_ORG','I-GPE')
        dataFrame['entity'] = dataFrame['entity'].str.replace('B-GPE_LOC','B-GPE')
        dataFrame['entity'] = dataFrame['entity'].str.replace('I-GPE_LOC','I-GPE')

    # ------------------------------FIELDS----------------------------
    text_field = Field(sequential=True, batch_first=True, include_lengths=True)
    tag_field = Field(sequential=True, batch_first=True, pad_token=None, unk_token=None)
    fields = [('text', text_field), ('', None), ('', None), ('tag', tag_field)]

    # ------------------------------SETS------------------------------
    train_list, val_list, test_list = [], [], []

    for n, entry in train_data.iterrows():
        train_list.append(Example.fromlist(entry, fields))

    for n, entry in val_data.iterrows():
        val_list.append(Example.fromlist(entry, fields))

    for n, entry in test_data.iterrows():
        test_list.append(Example.fromlist(entry, fields))

    train_set = Dataset(train_list, fields)
    val_set = Dataset(val_list, fields)
    test_set = Dataset(test_list, fields)

    # ------------------------------VOCABS----------------------------
    print('Building vocab ..')
    text_field.build_vocab(train_set, max_size=10000, min_freq=2, vectors=Vectors("58.txt"))
    tag_field.build_vocab(train_set)
    print("We have {} words in the vocabulary, including UNK and PAD".format(len(text_field.vocab)))
    print("We have {} ENTITY tags to predict".format(len(tag_field.vocab)))

    # UNK & PAD Vectors
    text_field.vocab.vectors[0] = torch.randn(1, 100)
    text_field.vocab.vectors[1] = torch.randn(1, 100)

    # ------------------------------ITERATORS-------------------------

    train_iter = Iterator(train_set, batch_size=32, shuffle=True, device=device)
    val_iter = Iterator(val_set, batch_size=1, shuffle=False, device=device)
    test_iter = Iterator(test_set, batch_size=1, shuffle=False, device=device)

    # ------------------------------NETWORK---------------------------
    model = Network(
        text_field.vocab,
        tag_field.vocab,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    count_parameters(model)

# ------------------------------TRAINING--------------------------

tic = time.time()
print('Training...')

scores = []

for epoch in range(num_epochs):

    print(f"epoch {epoch + 1}/{num_epochs}")

    train_model(model, train_iter, criterion, optimizer)

    y_true, y_pred = eval_model(model, val_iter)

    score = f1_score(y_true, y_pred, average='macro')
    scores.append(score)

    print(classification_report(y_true, y_pred, zero_division=0))

toc = time.time()
training_time = toc - tic
print("Training time:", training_time)

# ------------------------------EVALUATION--------------------------
y_true, y_pred = eval_model(model, test_iter)
print(classification_report(y_true, y_pred, zero_division=0))
cm = confusion_matrix(y_true, y_pred)
cm[0, 0] = 0
print(cm)

# ------------------------------SAVING----------------------------
with open('char_BiLSTM_word_BiLSTM_scores.pkl', 'wb') as f:
  pickle.dump(scores, f)

with open('char_BiLSTM_word_BiLSTM_time.pkl', 'wb') as f:
  pickle.dump(training_time, f)

with open('char_BiLSTM_word_BiLSTM_cm.pkl', 'wb') as f:
  pickle.dump(cm.tolist(), f)

print("Saving model to 'char_CNN_word_BiLSTM.pt'...")
torch.save(model.state_dict(), 'char_BiLSTM_word_BiLSTM.pt')
