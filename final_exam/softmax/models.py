import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ------------------------------(1) word_CNN------------------------------

class word_CNN(nn.Module):

    def __init__(self, text_vocab, tag_vocab):

        super().__init__()

        # Embedding Layer
        self._embed = nn.Embedding(len(text_vocab), 100)
        self._embed.weight.data.copy_(text_vocab.vectors)
        self._embed.weight.requires_grad=False
        # Dropout Layer
        self._dropout = nn.Dropout(p=0.5)
        # Convolution Layer
        self._conv_1 = nn.Conv1d(in_channels=100, out_channels=200, kernel_size=3, padding=1)
        self._conv_2 = nn.Conv1d(in_channels=200, out_channels=200, kernel_size=3, padding=1)
        self._conv_3 = nn.Conv1d(in_channels=200, out_channels=200, kernel_size=3, padding=1)
        self._conv_4 = nn.Conv1d(in_channels=200, out_channels=200, kernel_size=3, padding=1)
        # Softmax Layer
        self._output = nn.Linear(200, len(tag_vocab))

    def forward(self, batch):

        text, lengths = batch.text

        # Embedding Layer
        embeds = self._embed(text).transpose(1, 2)
        # Convolution Layer
        conv_1 = self._dropout(F.relu(self._conv_1(embeds)))
        conv_2 = self._dropout(F.relu(self._conv_2(conv_1)))
        conv_3 = self._dropout(F.relu(self._conv_3(conv_2)))
        conv_4 = self._dropout(F.relu(self._conv_4(conv_3))).transpose(1, 2)
        # Softmax Layer
        outputs = self._output(conv_4)

        return outputs

# ------------------------------(2) word_BiLSTM------------------------------

class word_BiLSTM(nn.Module):

    def __init__(self, text_vocab, tag_vocab):

        super().__init__()

        # Embedding Layer
        self._embed = nn.Embedding(len(text_vocab), 100)
        self._embed.weight.data.copy_(text_vocab.vectors)
        self._embed.weight.requires_grad=False
        # Recurrent Layer
        self._rnn = nn.LSTM(input_size=100, hidden_size=100, num_layers=2, bidirectional=True, batch_first=True)
        # Softmax Layer
        self._output = nn.Linear(100 * 2, len(tag_vocab))

    def forward(self, batch):

        text, lengths = batch.text

        # Embedding Layer
        embeds = self._embed(text)
        # Recurrent Layer
        packed = nn.utils.rnn.pack_padded_sequence(embeds, lengths, batch_first=True, enforce_sorted=False)
        hidden, _ = nn.utils.rnn.pad_packed_sequence(self._rnn(packed)[0], batch_first=True)
        # Softmax Layer
        outputs = self._output(hidden)

        return outputs

# ------------------------------(3) word_CNN_word_BiLSTM------------------------------

class word_CNN_word_BiLSTM(nn.Module):

    def __init__(self, text_vocab, tag_vocab):

        super().__init__()

        # Embedding Layer
        self._embed = nn.Embedding(len(text_vocab), 100)
        self._embed.weight.data.copy_(text_vocab.vectors)
        self._embed.weight.requires_grad=False
        # Dropout Layer
        self._dropout = nn.Dropout(p=0.5)
        # Convolution Layer
        self._conv_1 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=3, padding=1)
        self._conv_2 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=3, padding=1)
        self._conv_3 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=3, padding=1)
        self._conv_4 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=3, padding=1)
        # Recurrent Layer
        self._rnn = nn.LSTM(input_size=100, hidden_size=100, num_layers=2, bidirectional=True, batch_first=True)
        # Softmax Layer
        self._output = nn.Linear(100 * 2, len(tag_vocab))

    def forward(self, batch):

        text, lengths = batch.text

        # Embedding Layer
        embeds = self._embed(text).transpose(1, 2)
        # Convolution Layer
        conv_1 = self._dropout(F.relu(self._conv_1(embeds)))
        conv_2 = self._dropout(F.relu(self._conv_2(conv_1)))
        conv_3 = self._dropout(F.relu(self._conv_3(conv_2)))
        conv_4 = self._dropout(F.relu(self._conv_4(conv_3))).transpose(1, 2)
        # Recurrent Layer
        packed = nn.utils.rnn.pack_padded_sequence(conv_4, lengths, batch_first=True, enforce_sorted=False)
        hidden, _ = nn.utils.rnn.pad_packed_sequence(self._rnn(packed)[0], batch_first=True)
        # Softmax Layer
        outputs = self._output(self._dropout(hidden))

        return outputs

# ------------------------------(4) word_BiLSTM_word_CNN------------------------------

class word_BiLSTM_word_CNN(nn.Module):

    def __init__(self, text_vocab, tag_vocab):

        super().__init__()

        # Embedding Layer
        self._embed = nn.Embedding(len(text_vocab), 100)
        self._embed.weight.data.copy_(text_vocab.vectors)
        self._embed.weight.requires_grad=False
        # Dropout Layer
        self._dropout = nn.Dropout(p=0.5)
        # Recurrent Layer
        self._rnn = nn.LSTM(input_size=100, hidden_size=100, num_layers=2, bidirectional=True, batch_first=True)
        # Convolution Layer
        self._conv_1 = nn.Conv1d(in_channels=200, out_channels=200, kernel_size=3, padding=1)
        self._conv_2 = nn.Conv1d(in_channels=200, out_channels=200, kernel_size=3, padding=1)
        self._conv_3 = nn.Conv1d(in_channels=200, out_channels=200, kernel_size=3, padding=1)
        self._conv_4 = nn.Conv1d(in_channels=200, out_channels=200, kernel_size=3, padding=1)
        # Softmax Layer
        self._output = nn.Linear(200, len(tag_vocab))

    def forward(self, batch):

        text, lengths = batch.text

        # Embedding Layer
        embeds = self._embed(text)
        # Recurrent Layer
        packed = nn.utils.rnn.pack_padded_sequence(embeds, lengths, batch_first=True, enforce_sorted=False)
        hidden, _ = nn.utils.rnn.pad_packed_sequence(self._rnn(packed)[0], batch_first=True)
        # Convolution Layer
        conv_1 = self._dropout(F.relu(self._conv_1(self._dropout(hidden.transpose(1, 2)))))
        conv_2 = self._dropout(F.relu(self._conv_2(conv_1)))
        conv_3 = self._dropout(F.relu(self._conv_3(conv_2)))
        conv_4 = self._dropout(F.relu(self._conv_4(conv_3))).transpose(1, 2)
        # Softmax Layer
        outputs = self._output(conv_4)

        return outputs

# ------------------------------(5) char_CNN_word_CNN------------------------------

class char_CNN_word_CNN(nn.Module):

    def __init__(self, text_vocab, tag_vocab):
        super().__init__()

        self.text_vocab = text_vocab

        # --------------------CHARACTERS--------------------
        char_vocab = list(set(' '.join(text_vocab.freqs.keys()))) + ['<', '>']
        self.word_to_id = {word : i for i, word in enumerate(char_vocab)}
        self.id_to_word = {i : word for i, word in enumerate(char_vocab)}

        # Embedding Layer
        self._embed_char = nn.Embedding(len(char_vocab), 50)
        # Convolutional Layer
        self._cnn = nn.Conv1d(in_channels=50, out_channels=50, kernel_size=3, padding=1)
        # Dropout Layer
        self._dropout = nn.Dropout(p=0.5)

        # --------------------WORDS-------------------------

        # Embedding Layer
        self._embed_word = nn.Embedding(len(text_vocab), 100)
        self._embed_word.weight.data.copy_(text_vocab.vectors)
        self._embed_word.weight.requires_grad=False
        # Convolution Layer
        self._conv_1 = nn.Conv1d(in_channels=150, out_channels=200, kernel_size=3, padding=1)
        self._conv_2 = nn.Conv1d(in_channels=200, out_channels=200, kernel_size=3, padding=1)
        self._conv_3 = nn.Conv1d(in_channels=200, out_channels=200, kernel_size=3, padding=1)
        self._conv_4 = nn.Conv1d(in_channels=200, out_channels=200, kernel_size=3, padding=1)
        # Softmax Layer
        self._output = nn.Linear(200, len(tag_vocab))

    def forward(self, batch):

        text, lengths = batch.text

        # --------------------CHARACTERS--------------------

        words_from_index = []
        for sentence in text:
            words_from_index.append([self.text_vocab.itos[index] for index in sentence])

        sentences_with_tensors = []
        for sentence in words_from_index:
            sentence_with_tensors = []
            for word in sentence:

                tensor = torch.tensor([self.word_to_id[char] for char in word], device=device)

                embed = self._embed_char(tensor)

                conv = F.relu(self._cnn(self._dropout(embed.transpose(0, 1).unsqueeze(0)))).max(dim=2)[0] # max returns tuple

                sentence_with_tensors.append(conv)

            sentences_with_tensors.append(torch.stack(sentence_with_tensors))

        sentences_with_tensors = torch.stack(sentences_with_tensors).squeeze(dim=2)

        # --------------------WORDS-------------------------

        # Embedding Layer
        embeds_word = self._embed_word(text)
        embeds = torch.cat([embeds_word, sentences_with_tensors], dim=2)
        # Convolution Layer
        conv_1 = self._dropout(F.relu(self._conv_1(self._dropout(embeds.transpose(1, 2)))))
        conv_2 = self._dropout(F.relu(self._conv_2(conv_1)))
        conv_3 = self._dropout(F.relu(self._conv_3(conv_2)))
        conv_4 = self._dropout(F.relu(self._conv_4(conv_3))).transpose(1, 2)
        # Softmax Layer
        outputs = self._output(conv_4)

        return outputs
# ------------------------------(6) char_BiLSTM_word_CNN------------------------------

class char_BiLSTM_word_CNN(nn.Module):

    def __init__(self, text_vocab, tag_vocab):
        super().__init__()

        self.text_vocab = text_vocab

        # --------------------CHARACTERS--------------------

        char_vocab = list(set(' '.join(text_vocab.freqs.keys()))) + ['<', '>']
        self.word_to_id = {word : i for i, word in enumerate(char_vocab)}
        self.id_to_word = {i : word for i, word in enumerate(char_vocab)}

        # Embedding Layer
        self._embed_char = nn.Embedding(len(char_vocab), 50)
        # Convolutional Layer
        self._rnn_char = nn.LSTM(input_size=50, hidden_size=25, num_layers=2, bidirectional=True, batch_first=True)
        # Dropout Layer
        self._dropout = nn.Dropout(p=0.5)

        # --------------------WORDS-------------------------

        # Embedding Layer
        self._embed_word = nn.Embedding(len(text_vocab), 100)
        self._embed_word.weight.data.copy_(text_vocab.vectors)
        self._embed_word.weight.requires_grad=False
        # Convolution Layer
        self._conv_1 = nn.Conv1d(in_channels=150, out_channels=200, kernel_size=3, padding=1)
        self._conv_2 = nn.Conv1d(in_channels=200, out_channels=200, kernel_size=3, padding=1)
        self._conv_3 = nn.Conv1d(in_channels=200, out_channels=200, kernel_size=3, padding=1)
        self._conv_4 = nn.Conv1d(in_channels=200, out_channels=200, kernel_size=3, padding=1)
        # Softmax Layer
        self._output = nn.Linear(200, len(tag_vocab))

    def forward(self, batch):

        text, lengths = batch.text

        # --------------------CHARACTERS--------------------

        id_to_word = []
        for sentence in text:
            id_to_word.append([self.text_vocab.itos[id] for id in sentence])

        sentences_with_tensors = []
        for sentence in id_to_word:
            #print(len(sentence))
            sentence_with_tensors = []
            for word in sentence:
                tensor = torch.tensor([self.word_to_id[char] for char in word], device=device)
                #print(tensor.shape)
                embeds = self._embed_char(tensor)
                #print(embeds.shape)
                hidden, last = self._rnn_char(embeds.unsqueeze(0))
                #print(hidden.squeeze(0)[-1].shape)
                sentence_with_tensors.append(hidden.squeeze(0)[-1])
            #print(torch.stack(sentence_with_tensors).shape)
            sentences_with_tensors.append(torch.stack(sentence_with_tensors))
        #print(torch.stack(sentences_with_tensors).shape)
        sentences_with_tensors = torch.stack(sentences_with_tensors)

        # --------------------WORDS-------------------

        # Embedding Layer
        embeds_word = self._embed_word(text)
        embeds = torch.cat([embeds_word, sentences_with_tensors], dim=2)
        # Convolution Layer
        conv_1 = self._dropout(F.relu(self._conv_1(self._dropout(embeds.transpose(1, 2)))))
        conv_2 = self._dropout(F.relu(self._conv_2(conv_1)))
        conv_3 = self._dropout(F.relu(self._conv_3(conv_2)))
        conv_4 = self._dropout(F.relu(self._conv_4(conv_3))).transpose(1, 2)
        # Softmax Layer
        outputs = self._output(conv_4)

        return outputs

# ------------------------------(7) char_CNN_word_BiLSTM------------------------------

class char_CNN_word_BiLSTM(nn.Module):

    def __init__(self, text_vocab, tag_vocab):
        super().__init__()

        self.text_vocab = text_vocab

        # --------------------CHARACTERS--------------------
        char_vocab = list(set(' '.join(text_vocab.freqs.keys()))) + ['<', '>']
        self.word_to_id = {word : i for i, word in enumerate(char_vocab)}
        self.id_to_word = {i : word for i, word in enumerate(char_vocab)}

        # Embedding Layer
        self._embed_char = nn.Embedding(len(char_vocab), 50)
        # Convolutional Layer
        self._cnn = nn.Conv1d(in_channels=50, out_channels=50, kernel_size=3, padding=1)
        # Dropout Layer
        self._dropout = nn.Dropout(p=0.5)

        # --------------------WORDS-------------------------

        # Embedding Layer
        self._embed_word = nn.Embedding(len(text_vocab), 100)
        self._embed_word.weight.data.copy_(text_vocab.vectors)
        self._embed_word.weight.requires_grad=False
        # Recurrent Layer
        self._rnn = nn.LSTM(input_size=150, hidden_size=100, num_layers=2, bidirectional=True, batch_first=True)
        # Softmax Layer
        self._output = nn.Linear(100 * 2, len(tag_vocab))

    def forward(self, batch):

        text, lengths = batch.text

        # --------------------CHARACTERS--------------------

        words_from_index = []
        for sentence in text:
            words_from_index.append([self.text_vocab.itos[index] for index in sentence])

        sentences_with_tensors = []
        for sentence in words_from_index:
            sentence_with_tensors = []
            for word in sentence:

                tensor = torch.tensor([self.word_to_id[char] for char in word], device=device)

                embed = self._embed_char(tensor)

                conv = F.relu(self._cnn(self._dropout(embed.transpose(0, 1).unsqueeze(0)))).max(dim=2)[0] # max returns tuple

                sentence_with_tensors.append(conv)

            sentences_with_tensors.append(torch.stack(sentence_with_tensors))

        sentences_with_tensors = torch.stack(sentences_with_tensors).squeeze(dim=2)

        # --------------------WORDS-------------------------

        # Embedding Layer
        embeds_word = self._embed_word(text)
        embeds = torch.cat([embeds_word, sentences_with_tensors], dim=2)
        # Recurrent Layer
        packed = nn.utils.rnn.pack_padded_sequence(self._dropout(embeds), lengths, batch_first=True, enforce_sorted=False)
        hidden, _ = nn.utils.rnn.pad_packed_sequence(self._rnn(packed)[0], batch_first=True)
        # Softmax Layer
        outputs = self._output(self._dropout(hidden))

        return outputs

# ------------------------------(8) char_BiLSTM_word_BiLSTM------------------------------

class char_BiLSTM_word_BiLSTM(nn.Module):

    def __init__(self, text_vocab, tag_vocab):
        super().__init__()

        self.text_vocab = text_vocab

        # --------------------CHARACTERS--------------------

        char_vocab = list(set(' '.join(text_vocab.freqs.keys()))) + ['<', '>']
        self.word_to_id = {word : i for i, word in enumerate(char_vocab)}
        self.id_to_word = {i : word for i, word in enumerate(char_vocab)}

        # Embedding Layer
        self._embed_char = nn.Embedding(len(char_vocab), 50)
        # Convolutional Layer
        self._rnn_char = nn.LSTM(input_size=50, hidden_size=25, num_layers=2, bidirectional=True, batch_first=True)
        # Dropout Layer
        self._dropout = nn.Dropout(p=0.5)

        # --------------------WORDS-----------------------

        # Embedding Layer
        self._embed_word = nn.Embedding(len(text_vocab), 100)
        self._embed_word.weight.data.copy_(text_vocab.vectors)
        self._embed_word.weight.requires_grad=False
        # Recurrent Layer
        self._rnn_word = nn.LSTM(input_size=150, hidden_size=100, num_layers=2, bidirectional=True, batch_first=True)
        # Softmax Layer
        self._output = nn.Linear(100 * 2, len(tag_vocab))

    def forward(self, batch):

        text, lengths = batch.text

        # --------------------CHARACTERS--------------------

        id_to_word = []
        for sentence in text:
            id_to_word.append([self.text_vocab.itos[id] for id in sentence])

        sentences_with_tensors = []
        for sentence in id_to_word:
            #print(len(sentence))
            sentence_with_tensors = []
            for word in sentence:
                tensor = torch.tensor([self.word_to_id[char] for char in word], device=device)
                #print(tensor.shape)
                embeds = self._embed_char(tensor)
                #print(embeds.shape)
                hidden, last = self._rnn_char(embeds.unsqueeze(0))
                #print(hidden.squeeze(0)[-1].shape)
                sentence_with_tensors.append(hidden.squeeze(0)[-1])
            #print(torch.stack(sentence_with_tensors).shape)
            sentences_with_tensors.append(torch.stack(sentence_with_tensors))
        #print(torch.stack(sentences_with_tensors).shape)
        sentences_with_tensors = torch.stack(sentences_with_tensors)

        # --------------------WORDS-------------------

        # Embedding Layer
        embeds_word = self._embed_word(text)
        embeds = torch.cat([embeds_word, sentences_with_tensors], dim=2) # concat char and word embeddings
        # Recurrent Layer
        packed = nn.utils.rnn.pack_padded_sequence(self._dropout(embeds), lengths, batch_first=True, enforce_sorted=False)
        hidden, _ = nn.utils.rnn.pad_packed_sequence(self._rnn_word(packed)[0], batch_first=True)
        # Softmax Layer
        outputs = self._output(self._dropout(hidden))

        return outputs
