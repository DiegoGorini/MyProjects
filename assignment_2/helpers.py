#!/bin/env python3
# coding: utf-8

# ————————————————————LIBRARIES————————————————————

import torch
import torch.nn.functional as F
from torch import nn, optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

# ————————————————————TRAINING————————————————————

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

            # Update Epoch's Loss
            epoch_loss += loss.item()

        scheduler.step()                    # Update Learning Rate

# ————————————————————EVALUATION————————————————————

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

    print(f"Accuracy: {accuracy}% \t| Precision: {precision} \t| Recall: {recall} \t| F1 Score: {f1} \n")
    print("Classification Report:\n", classification_report(y_true, y_pred))
