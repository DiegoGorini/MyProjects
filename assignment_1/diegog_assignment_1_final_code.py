#!/usr/bin/python3
# coding: utf-8

# ————————————————————LIBRARIES————————————————————
from argparse import ArgumentParser
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torch.utils.data import DataLoader
from helpers import *
import time

if __name__ == "__main__":
    # ————————————————————ARGS–––———————————————————––––––
    parser = ArgumentParser()
    parser.add_argument('--PATH', help="Path to the training corpus", action='store')
    parser.add_argument('--VOCAB_SIZE', help="How many words types to consider", action='store', type=int, default=3000)
    parser.add_argument('--HIDDEN_DIM', help="Size of the hidden layer(s)", action='store', type=int, default=128)
    parser.add_argument('--BATCH_SIZE', help="Size of mini-batches", action='store', type=int, default=32)
    parser.add_argument('--LR', action='store', help="Learning rate", type=float, default=1e-3)
    # I added args STEP_SIZE and GAMMA for scheduler to experiment with different values. Default worked best.
    parser.add_argument('--STEP_SIZE', action='store', help="Update LR every STEP_SIZE epochs", type=int, default=1)
    parser.add_argument('--GAMMA', action='store', help="Multiply LR by GAMMA every STEP_SIZE", type=float, default=.1)
    parser.add_argument('--EPOCHS', action='store', help="Max number of epochs", type=int, default=10)
    parser.add_argument('--SPLIT', action='store', help="Ratio of train/dev split", type=float, default=0.9)
    args = parser.parse_args()

    # ————————————————————DATASET————————————————————––––––
    datafile = args.PATH

    print('Loading the dataset...')
    train_set = pd.read_csv(datafile, sep='\t', header=0, compression='gzip')
    print('Finished loading the dataset')

    # Set RNG seed for reproducibility
    torch.manual_seed(42)

    # ————————————————————TEXT VECTORIZER——————————————————
    texts = train_set['text']

    text_vectorizer = CountVectorizer(max_features=args.VOCAB_SIZE,
                                      strip_accents='unicode',
                                      lowercase=False,
                                      binary=True)

    input_features = text_vectorizer.fit_transform(texts.values).toarray().astype(np.float32)

    print('Train data:', input_features.shape)

    with open('vectorizer.pickle', 'wb') as f:
                pickle.dump(text_vectorizer, f)

    # ————————————————————LABELS ENCODING——————————————————
    classes = train_set['source']

    label_vectorizer = LabelEncoder()

    gold_classes = label_vectorizer.fit_transform(classes.values)

    classes = label_vectorizer.classes_
    num_classes = len(classes)
    print(num_classes, 'classes:')
    print(classes)

    # ————————————————————SPLITTING————————————————————––––

    # From Array to Tensor: data.TensorDataset(X, y)
    input_features = torch.from_numpy(input_features)
    gold_classes = torch.from_numpy(gold_classes)
    dataset = data.TensorDataset(input_features, gold_classes)

    # Train/Dev Ratio
    train_size = int(args.SPLIT * len(dataset))
    dev_size = len(dataset) - train_size

    # Split Dataset into Training Set and Development Set
    train_set, dev_set = data.random_split(dataset, [train_size, dev_size])

    print('Training instances after split:', len(train_set))

    # ————————————————————ITERATORS—————————————————————————
    iterTrain = DataLoader(train_set, batch_size=args.BATCH_SIZE, shuffle=True)
    iterDev = DataLoader(dev_set, batch_size=dev_size, shuffle=False)

    # ————————————————————MODEL—————————————————————————————
    class FFNN(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(FFNN, self).__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            out = F.relu(self.fc1(x))
            out = F.relu(self.fc2(out))
            out = self.fc3(out)

            return out
    # ————————————————————OBJECTS——————————————————————————
    model = FFNN(args.VOCAB_SIZE, args.HIDDEN_DIM, num_classes)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.LR)
    # scheduler adjusts the Learning Rate every STEP_SIZE epochs by a multiple GAMMA
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=args.STEP_SIZE,
                                          gamma=args.GAMMA)

    print(f"Model: {model}")

    # ————————————————————TRAINING–————————————————————————
    tic = time.time()

    # This list I simply used to plot the Loss
    # loss_values = []

    for epoch in range(args.EPOCHS):
        for dataBatch in iterTrain:
            input, y = dataBatch
            optimizer.zero_grad()
            output = model(input)       # Forward Pass
            loss = loss_fn(output, y)   # Compute Loss
            loss.backward()             # Backward Pass
            optimizer.step()            # Update Parameters
            #  loss_values.append(loss)
        scheduler.step()                # Update Learning Rate
        # print('Epoch [%d/%d], Loss: %.4f' %(epoch+1, epochs, loss))

    toc = round(time.time() - tic, 2)

    # ————————————————————METRICS–––––––––—————————————–––
    dev_accuracy, dev_predictions, dev_labels = eval_func(iterDev, model)
    # Accuracy, Precision, Recall, F1 Score
    dev_accuracy = accuracy_score(dev_labels, dev_predictions)
    dev_precision = precision_score(dev_labels, dev_predictions, average='weighted')
    dev_recall = recall_score(dev_labels, dev_predictions, average='weighted')
    dev_f1 = f1_score(dev_labels, dev_predictions, average='weighted')

    # ————————————————————REPORT–––––––––—————————————––––
    print('Classification report for the Development Set:')
    gold_classes_human = [classes[x] for x in dev_labels]
    predicted_dev_human = [classes[x] for x in dev_predictions]
    print(classification_report(gold_classes_human, predicted_dev_human))

    # ————————————————————EVALUATION––––––—————————————–––
    # I used this to experiment with the hyperparameters and to report the results in the written
    # assignment.
    # print("No. Layers: 2")
    # print(f"VOCAB_SIZE: {args.VOCAB_SIZE} \tHIDDEN_DIM: {args.HIDDEN_DIM} \tBATCH_SIZE: {args.BATCH_SIZE} \tLEARNING_RATE: {args.LR} \tSTEP_SIZE: {args.STEP_SIZE} \tGAMMA: {args.GAMMA} \tEPOCHS: {args.EPOCHS} \tTIME: {toc}")
    # print("ACCURACY: {} \tPRECISION: {} \tRECALL: {} \tF1 SCORE: {}".format(round(dev_accuracy*100, 2), round(dev_precision*100, 2), round(dev_recall*100, 2), round(dev_f1*100, 2)))
    # print("––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––")

    # ————————————————————SAVE MODEL––––––—————————————–––
    torch.save(model, 'diegog_final_model_2layers.pt')
