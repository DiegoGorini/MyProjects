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

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since):
    now = time.time()
    s = now - since
    print(asMinutes(s))
    return '%s' % (asMinutes(s))

train_data =  pd.read_csv("train_data.csv", index_col=0)
#train_data=train_data.iloc[:250]
val_data =  pd.read_csv("val_data.csv", index_col=0)
#val_data=val_data.iloc[:96]
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

text_field.build_vocab(train_data, max_size=6000, min_freq=2)
pos_field.build_vocab(train_data)
print("We have {} words in the vocabulary, including UNK and PAD".format(len(text_field.vocab)))
print("We have {} POS tags to predict".format(len(pos_field.vocab)))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_iter = Iterator(train_data, batch_size=32, shuffle=True, device=device)
val_iter = Iterator(val_data, batch_size=1, shuffle=False, device=device)
test_iter = Iterator(test_data, batch_size=1, shuffle=False, device=device)

# Char Vocab, Char Dicts

char_vocab = list(set(' '.join(text_field.vocab.freqs.keys()))) + ['<', '>']

char_to_id = {}
id_to_char = {}
id_to_char[len(char_to_id)]="<wp>"
char_to_id["<wp>"]=len(char_to_id)

for i, char in enumerate(char_vocab):
    char_to_id[char] = i+1
    id_to_char[i+1] = char

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
    def __init__(self, ntoken, emsize,char_emsize, nhead, nhid, nlayers,noutputs, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(emsize, dropout)
        encoder_layers = TransformerEncoderLayer(emsize, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, emsize)
        ##char encoding
        self.char_emb=nn.Embedding(len(char_to_id), char_emsize )
        char_encoder_layer=TransformerEncoderLayer(char_emsize , nhead, char_emsize , 0.01)
        self.char_transformer_encoder = TransformerEncoder(char_encoder_layer, nlayers)
        self.char_pos_encoder = PositionalEncoding(char_emsize, dropout)
        ##end of char encoding




        self.emsize=emsize
        self.char_emsize=char_emsize
        self.src_mask = None
        self.linear1= nn.Linear(emsize+char_emsize, 100)
        self.decoder = nn.Linear(100, noutputs)
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
        max_sentence_length=text.size(1)
        sentence_words_reps=torch.empty(0,max_sentence_length,self.char_emsize,device=device)
        for ind,sentence in enumerate(text):
            setence_len_wit_pad=len(sentence)
#             print("frases largo 1", len(sentence))
            sentence=sentence[0:lengths[ind]]
#             print("frases largo 2", len(sentence))
            len_words=[]
            words_indexes=[]
            for word in sentence:
                
                palabra= text_field.vocab.itos[word]
                len_words.append(len(palabra))
                palabra_indexes=[ char_to_id[j] for j in palabra]
                words_indexes.append(palabra_indexes)
            max_len=max(len_words)
            sentence_batch=torch.zeros(lengths[ind],max_len,device=device)
            for index_w, word in enumerate(sentence):
                for index_c in range(len_words[index_w]):
                    sentence_batch[index_w,index_c]=words_indexes[index_w][index_c]
            c_ems=self.char_emb(sentence_batch.long())
            c_ems_pos=self.char_pos_encoder(c_ems)
            mask_c=self._generate_square_subsequent_mask(torch.tensor(len_words,device=device))
            c_transformer=self.char_transformer_encoder(c_ems_pos, src_key_padding_mask=mask_c)
            pad_vecs=torch.zeros(setence_len_wit_pad-len(sentence),max_len,self.char_emsize).to(device)
            character_rep=torch.cat((c_transformer,  pad_vecs), 0)
            character_rep = torch.topk(character_rep,1,1).values.transpose(0,1)
            sentence_words_reps=torch.cat((sentence_words_reps,character_rep),0)



        mask = self._generate_square_subsequent_mask(lengths)
        embeddings = self.embedding(text) * math.sqrt(self.emsize)
        src = self.pos_encoder(embeddings)
        
        
        
        output =self.transformer_encoder(src ,src_key_padding_mask=mask)
        output=torch.cat((output,sentence_words_reps),2)
        hidden = F.relu(self.linear1(output))
        output2 = self.decoder(hidden)
        return output2

import tqdm
def train_model(model, batches,loss_f, optimizer):
    for batch in tqdm.tqdm(batches):
        pred = model(batch).transpose(1, 2)
        loss = loss_f(pred, batch.pos)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def evaluate(model, batches):
    correct, total = 0, 0
    for batch in tqdm.tqdm(batches):
        pred = model(batch)[0].argmax(dim=-1)
        gold = batch.pos
        correct += (pred == gold).nonzero().size(0)
        total += pred.size(0)
        
    
    return correct, total

def grid_search():

    ntokens = len(text_field.vocab.stoi)
    noutputs=len(pos_field.vocab.stoi)
    emsizes = [100] # embedding dimension
    nhid = 100 # the dimension of the feedforward network model in nn.TransformerEncoder
    nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 2 # the number of heads in the multiheadattention models
    dropout = 0.01 # the dropout value
    char_emsizes=[100]
    num_epochs=25
    lr=0.001
    exp_name="e29_char_less_dict"
    points = []
    for emsize in emsizes:
        for char_emsize in char_emsizes:            
            parameters=[]
            model = TransformerModel(ntokens, emsize,char_emsize, nhead, emsize, nlayers,noutputs, dropout).to(device)
            loss_f = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)
            for epoch in range(num_epochs):
                start = time.time()
                model.train()
                train_model(model, train_iter,loss_f, optimizer)
                endTime=timeSince(start)
                parameters=[]
                if (epoch%1==0) or epoch==num_epochs:
                    with torch.no_grad():
                        model.eval()
                        pred_values=[]
                        real_values=[]
                        for batch in tqdm.tqdm(val_iter):
                            pred = model(batch)[0].argmax(dim=-1)
                            gold = batch.pos

                            for i in pred:
                                pred_values.append(i.item())
                            for j in gold[0]:
                                real_values.append(j.item())

                        f1=f1_score(pred_values,real_values,average="macro")
                        precision=precision_score(pred_values,real_values,average="macro")
                        accuracy=accuracy_score(pred_values,real_values)
                        parameters.append(char_emsize)
                        parameters.append(endTime)
                        parameters.append(epoch)
                        parameters.append(f1)
                        parameters.append(precision)
                        parameters.append(accuracy)
                        points.append(parameters) 
                        report = classification_report(pred_values, real_values, output_dict=True)
                        df_rep = pd.DataFrame(report).transpose()
                        rep_name=exp_name+str(emsize)
                        df_rep.to_csv("results2/report_"+rep_name+"epoch"+str(epoch)+".csv")
                        model_name=rep_name+"dict"+"epoch"+str(epoch)+".pt"
                        torch.save(model.state_dict(), model_name )
    df1=pd.DataFrame(points, columns=['char_emsize',"time","epoch", "f1", "precision","accuracy"])
    df1.to_csv("results/"+exp_name +".csv")
                


grid_search() 