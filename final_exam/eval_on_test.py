import pandas as pd
#from conllu import parse
import torch
from torch import nn
from torch import optim
from torchtext.vocab import Vectors
from sklearn.metrics import accuracy_score, precision_score ,confusion_matrix, f1_score, classification_report,confusion_matrix
import numpy as np
import torch.nn.functional as F

import time
import math



train_data =  pd.read_csv("train_data.csv", index_col=0)
val_data =  pd.read_csv("val_data.csv", index_col=0)
test_data =  pd.read_csv("test_data.csv", index_col=0)

for dataFrame in [train_data, val_data, test_data]:
    # Remove B-MISC and I-MISC tags
    dataFrame['entity'] = dataFrame['entity'].str.replace('B-MISC','O')
    dataFrame['entity'] = dataFrame['entity'].str.replace('I-MISC','O')
    # Merge GPE_ORG and GPE_LOC into GPE
    dataFrame['entity'] = dataFrame['entity'].str.replace('B-GPE_ORG','B-GPE')
    dataFrame['entity'] = dataFrame['entity'].str.replace('I-GPE_ORG','I-GPE')
    dataFrame['entity'] = dataFrame['entity'].str.replace('B-GPE_LOC','B-GPE')
    dataFrame['entity'] = dataFrame['entity'].str.replace('I-GPE_LOC','I-GPE')
    

def create_data(df, fields):
    fields_list = []
    for n, entry in df.iterrows():
        fields_list.append(Example.fromlist(entry, fields))
    data = Dataset(fields_list, fields)

        
    return data

from torchtext.data import Field, Dataset, Iterator, Example
text_field = Field(sequential=True, batch_first=True, include_lengths=True)
pos_field = Field(sequential=True, batch_first=True, unk_token=None)

fields=[('text', text_field),  ('', None),  ('', None), ('pos', pos_field)]
train_data=create_data(train_data, fields)
val_data=create_data(val_data, fields)
test_data=create_data(test_data, fields)

text_field.build_vocab(train_data, max_size=10000, min_freq=2)
pos_field.build_vocab(train_data)
print("We have {} words in the vocabulary, including UNK and PAD".format(len(text_field.vocab)))
print("We have {} POS tags to predict".format(len(pos_field.vocab)))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_iter = Iterator(train_data, batch_size=32, shuffle=True, device=device)
val_iter = Iterator(val_data, batch_size=1, shuffle=False, device=device)
test_iter = Iterator(test_data, batch_size=1, shuffle=False, device=device)



from torch.nn import TransformerEncoder, TransformerEncoderLayer

import math
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, ntoken, emsize, nhead, nhid, nlayers,noutputs, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(emsize, dropout)
        encoder_layers = TransformerEncoderLayer(emsize, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, emsize)
        self.emsize=emsize
        self.src_mask = None
        self.decoder = nn.Linear(emsize, noutputs)
        self.init_weights()
    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
    def _generate_square_subsequent_mask(self,lengths):
        max_length=torch.max(lengths)
        mask_arr=[]
        for i in range(len(lengths)):
            line_arr=[0 if j<lengths[i] else 1 for j in range( max_length)]
            mask_arr.append(line_arr)
        mask=torch.tensor(mask_arr, device=device).bool().transpose(0,1)
        return mask
        

    def forward(self,batch):
        text, lengths = batch.text    
        mask = self._generate_square_subsequent_mask(lengths)
        embeddings = self.embedding(text) * math.sqrt(self.emsize)
        src = self.pos_encoder(embeddings)
        output =self.transformer_encoder(src ,  src_key_padding_mask=mask)
        output = self.decoder(output)
        return output

import tqdm



def weightedScore(report,score="f1-score",start=1,finish=15):
    report_Table = pd.DataFrame(report).transpose()
    table=report_Table.iloc[start:finish]
    score_sum=0
    support_sum=0
    for i,j in zip(table[score],table["support"]):
        score_sum+=i*j
        support_sum+=j
    score_average=score_sum/support_sum
    return score_average     


def evaluate(model_file):
    ntokens = len(text_field.vocab.stoi)
    noutputs=len(pos_field.vocab.stoi)
    emsize = 100 # embedding dimension
    nhid = 100 # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2 # the number of heads in the multiheadattention models
    dropout = 0.01 # the dropout value
    char_emsize=100
 
    lr=0.001
    model = TransformerModel(ntokens, emsize, nhead, emsize, nlayers,noutputs, dropout).to(device)
    model.load_state_dict(torch.load(model_file))
    model.eval()
    correct, total = 0, 0
    pred_values=[]
    real_values=[]
    for batch in tqdm.tqdm(test_iter):
        pred = model(batch)[0].argmax(dim=-1)
        gold = batch.pos

        for i in pred:
            pred_values.append(i.item())
        for j in gold[0]:
            real_values.append(j.item())
    f1=f1_score(pred_values,real_values,average="macro")
    f1_w=f1_score(pred_values,real_values,average="weighted")
    precision=precision_score(pred_values,real_values,average="macro")
    accuracy=accuracy_score(pred_values,real_values)
    report = classification_report(pred_values, real_values, output_dict=True)
    print("f1 macro",f1 )
    print("f1 weigthed",f1_w )
    print("f1 weighted adjuted", weightedScore(report,score="f1-score",start=1,finish=15))


    
evaluate("model.pt")