#!/bin/env python3
# coding: utf-8

if __name__ == '__main__':

    import sys
    path = sys.argv[1]

    print('\nTest set...')

    # ————————————————————CSV————————————————————

    import pandas as pd

    # Training CSV
    csv_test = pd.read_csv(path, sep='\t', header=0, compression='gzip')

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
    test_examples = []
    for index, entry in csv_test.iterrows():
        example = Example.fromlist(entry, fields)
        test_examples.append(example)

    from torchtext.data import Dataset

    # Development Set (list of examples -> 'Dataset' instance)
    test_set = Dataset(test_examples, fields)

    from torchtext.data import Iterator

    # Build Iterator
    iterTest = Iterator(test_set,
                        batch_size=1,
                        shuffle=False)

    print("...checked.")
    print("Number of examples: {}.\n".format(len(test_set)))

    # ————————————————————VECTORIZER————————————————————

    print('Vectorizer...')

    import torch
    import torch.nn.functional as F
    from torch import nn, optim

    torch.manual_seed(42)

    import pickle

    with open('fields.bin', 'rb') as f:
        lemmatized_vocab, label_vocab = pickle.load(f)
        field_lemmatized.vocab = lemmatized_vocab
        field_label.vocab =label_vocab

    print('...checked.\n')

    # ————————————————————NETWORK————————————————————

    print('Model...')

    class CNN(nn.Module):

        def __init__(self):
            super(CNN, self).__init__()

            # Embeddings Layer
            self.embeddings = nn.Embedding(9002, 300)
            # self.embeddings.weight.data.copy_(field_lemmatized.vocab.vectors)

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

    model = CNN()

    # PyTorch Model
    model = torch.load('cnn_best_model.pt')

    print(model)

    print('...checked.\n')

    # ————————————————————TESTING————————————————————

    print('===========================')
    print('Evaluation:')
    print('===========================')

    from helpers import eval_model

    eval_model(model, iterTest)
