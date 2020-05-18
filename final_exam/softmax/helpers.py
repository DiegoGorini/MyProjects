import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import itertools
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train_model(model, batches, criterion, optimizer):

    model.train()

    for batch in batches:
        pred = model(batch).transpose(1, 2)  # to correspond to what Cross Entropy expects
        loss = criterion(pred, batch.tag)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def confusion_matrix(y_true, y_pred):

    K = len(set(y_true)) # K classes
    cm = torch.zeros(K, K, dtype=torch.int32)
    for true, pred in zip(y_true, y_pred):
        cm[true - 1, pred - 1] = cm[true - 1, pred - 1] + 1

    return cm

def plot_confusion_matrix(cm, tags, title='Confusion Matrix', cmap=plt.cm.Blues):

    plt.figure(figsize=(10, 10))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(tags))
    plt.xticks(tick_marks, tags, rotation=45)
    plt.yticks(tick_marks, tags)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

def count_parameters(model):
    num_params = []
    for param in model.parameters():
      num_params.append(param.shape.numel())
    print("Nº Parameters:", sum(num_params))

def eval_model(model, iterator, report=True, cm_plot=True):

    """
    Returns: Labels, Predictions.
    Prints: Classification Report.
    """

    model.eval()

    y_true, y_pred = [], []
    for batch in iterator:
        pred = model(batch)[0].argmax(dim=-1)
        for i in pred:
            y_pred.append(i.item())
        for j in batch.tag[0]:
            y_true.append(j.item())

    return y_true, y_pred

    #if report:
        #tags = ['O', 'PER', 'ORG', 'GPE', 'LOC', 'DRV', 'EVT', 'PROD']
        #dic = {1:1, 2:2, 3:3, 4:1, 5:2, 6:7, 7:4, 8:7, 9:5, 10:4, 11:3, 12:6, 13:5, 14:6}
        #y_true = [dic.get(n, n) for n in y_true]
        #y_pred = [dic.get(n, n) for n in y_pred]
        #print(classification_report(y_true, y_pred, target_names=tag_field.vocab.itos, zero_division=0))
