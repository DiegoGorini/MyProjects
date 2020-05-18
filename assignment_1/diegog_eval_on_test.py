#!/bin/env python3
# coding: utf-8

from argparse import ArgumentParser
import pickle
import numpy as np
import pandas as pd
import torch
from torch import nn # I added this library
import torch.nn.functional as F # and this one
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from torch.utils import data
from helpers import eval_func, generate_dataset

#  This scripts load your pre-trained model and your vectorizer object (essentialy, vocabulary)
#  and evaluates it on a test set.

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--modelfile', help="PyTorch model saved with model.save()", action='store')
    parser.add_argument('--vectorizerfile',
                        help="CountVectorizer() object saved as a Python pickle", action='store')
    parser.add_argument('--testfile', help="Test dataset)", action='store')
    args = parser.parse_args()

    print('Loading the model...')
    with open(args.vectorizerfile, 'rb') as f:
        text_vectorizer = pickle.load(f)  # Loading the vectorizer

    # I simply copy-pasted the class from my code
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

    model = torch.load(args.modelfile)  # Loading the model itself
    model.eval()
    print(model)

    print('Loading the test set...')
    test_dataset = pd.read_csv(args.testfile, sep='\t', header=0, compression="gzip")
    print('Finished loading the test set')

    (x_test, y_test) = test_dataset['text'], test_dataset['source']

    print(len(x_test), 'test texts')

    text_length = np.mean(list(map(len, x_test.str.split())))
    print('Average test text length: {0:.{1}f} words'.format(text_length, 1))

    label_vectorizer = LabelEncoder()

    # Processing the test data in exactly the same way we processed the training data:
    input_features = text_vectorizer.transform(x_test.values).toarray().astype(np.float32)
    gold_classes = label_vectorizer.fit_transform(y_test.values)

    print('Test data:', input_features.shape)

    classes = label_vectorizer.classes_
    num_classes = len(classes)
    print(num_classes, 'classes')
    print('===========================')
    print('Class distribution in the testing data:')
    print(test_dataset.groupby('source').count())
    print('===========================')
    print('Test labels shape:', gold_classes.shape)

    dataset = generate_dataset(input_features, gold_classes)

    # Creating one big batch to evaluate the whole dataset at once:
    dataset = data.DataLoader(dataset, batch_size=len(x_test), shuffle=False)

    print('===========================')
    print('Evaluation:')
    print('===========================')

    test_accuracy, test_predictions, test_labels = eval_func(dataset, model)

    print("Accuracy on the test set:", round(test_accuracy, 3))
    print('Classification report for the test set:')
    gold_classes_human = [classes[x] for x in test_labels]
    predicted_test_human = [classes[x] for x in test_predictions]
    print(classification_report(gold_classes_human, predicted_test_human))
