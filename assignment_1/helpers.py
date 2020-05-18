#!/bin/env python3
# coding: utf-8
import torch
from torch.utils import data


def eval_func(batched_data, model2use):
    # This function uses a model to compute predictions on data coming in batches.
    # Then it calculates the accuracy of predictions with respect to the gold labels.
    correct = 0
    total = 0
    predicted = None
    gold_label = None

    # Iterating over all batches (can be 1 batch as well):
    for n, (input_data, gold_label) in enumerate(batched_data):
        out = model2use(input_data)
        predicted = out.argmax(1)
        correct += len((predicted == gold_label).nonzero())
        total += len(gold_label)
    accuracy = correct / total
    return accuracy, predicted, gold_label


def generate_dataset(input_features, gold_classes):
    # This function creates a PyTorch object for the data

    torch_input_features = torch.from_numpy(input_features)
    torch_gold_classes = torch.from_numpy(gold_classes)
    dataset = data.TensorDataset(torch_input_features, torch_gold_classes)
    return dataset
