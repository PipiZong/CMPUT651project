#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 13:44:08 2019

@author: yakun
 
"""

import random
import torch
from torch import nn
import numpy as np
import pickle
from torch.autograd import Variable
from sklearn import metrics
from keras.preprocessing import text, sequence
from torch.utils import data
import pandas as pd
# load precessed comments abd Glove dictionary
dimension = 50
comments_train = pickle.load(open('Data/processed/train_data.pkl','rb'))
comments_test = pickle.load(open('Data/processed/test_data.pkl','rb'))
embedding = pickle.load(open('Data/Embedding/processed/glove.6B.'+str(dimension)+'d_sent.pkl','rb'))
#restore words to sentenses
def generate_sentence(comments):
    
    x = []
    y = []
    index = []
    for i in range(len(comments)):
        if comments[i][1] != []:           
            x.append(' '.join(comments[i][1]))
            y.append(comments[i][2])
            index.append(comments[i][0])
    return x, y, index

x_train, y_train, _ = generate_sentence(comments_train)
y_train = np.array(y_train)
x_test, y_test, test_index = generate_sentence(comments_test)
y_test = np.array(y_test)

# Tokenize sentenses and index the words, padding each sequence to the same length 220
MAX_LEN = 220
tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(x_train + x_test)

x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)
x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)
x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)

#embedding words with Glove dictionary
def build_matrix(word_index, dic):
    embedding_matrix = np.zeros((len(word_index) + 1, dimension+1))
    unknown_words = []
    
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = dic[word]
        except KeyError:
            unknown_words.append(word)
    return embedding_matrix
embedding_matrix = build_matrix(tokenizer.word_index, embedding)

# transfer to tensor data and load by data.TensorDataset and cuda to prepare for training
x_train_torch = torch.tensor(x_train, dtype=torch.long)
x_test_torch = torch.tensor(x_test, dtype=torch.long)
y_train_torch = torch.tensor(y_train, dtype=torch.float32)
y_test_torch = torch.tensor(y_test, dtype=torch.float32)

train_dataset = data.TensorDataset(x_train_torch, y_train_torch)
test_dataset = data.TensorDataset(x_test_torch, y_test_torch)
pickle.dump((embedding_matrix,train_dataset, test_dataset, test_index), open('Data/LSTM'+str(dimension)+'d.pkl',"wb"))
## load the data generated from datapre.py
#dimension = 50
#embedding_matrix, train_dataset, test_dataset, test_index=pickle.load(open('Data/LSTM'+str(dimension)+'d.pkl',"rb"))

vocab_size = len(embedding_matrix)
# set seed for reproduction
def set_seed(seed=1234):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
set_seed()   
# build standard lstm with embedding, linear layers    
class NeuralNet(nn.Module):
    def __init__(self, embedding_matrix):
        super(NeuralNet, self).__init__()
        embed_size = embedding_matrix.shape[1]
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False        
        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, 2, batch_first=True)

    
        self.linear1 = nn.Linear(LSTM_UNITS, 1)

        
    def forward(self, x):
        h_embedding = self.embedding(x)    #batch_size * 220*201    
        h_lstm1, _ = self.lstm1(h_embedding) # batch_size, 220, 128
        out = h_lstm1[:,-1,:]
        out = self.linear1(out)
        out = torch.sigmoid(out)       
        return out
    
LSTM_UNITS = 128 
model = NeuralNet(embedding_matrix).cuda()
print(model)

loss_fn=nn.BCELoss()
samples_train = len(train_dataset)
samples_test = len(test_dataset)
lr=0.001
batch_size=2048
n_epochs=6
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# load data    
train_dataset = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# train model and test on the testing data by batches
for epoch in range(n_epochs):
    
    model.train()
    run_loss = 0.
    run_acc = 0. 
    
    for i, (x_batch,y_batch) in enumerate(train_dataset, 1):
        x_batch = Variable(x_batch.cuda())
        y_batch = Variable(y_batch.cuda())
        y_pred = model(x_batch)
        y_pred = torch.squeeze(y_pred,1)            
        loss = loss_fn(y_pred, y_batch)
        run_loss = run_loss + loss * y_batch.size(0)
        prediction = torch.round(y_pred)
        num_correct = (prediction == y_batch).sum()
        run_acc = run_acc + num_correct
        run_acc = run_acc.float()
         
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 20 == 0 and i > 0:
            print('[{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(
            epoch + 1, n_epochs, run_loss / (batch_size * (i)),
            run_acc / (batch_size * i)))
    
    print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
    epoch + 1, run_loss / samples_train, run_acc / samples_train))
        
    model.eval()
    eval_acc = 0
    test_pred = []
    test_true = []
    for i, (x_batch, y_batch) in enumerate(test_dataset):
        
        x_batch = Variable(x_batch)
        y_batch=y_batch.numpy()
        out = model(x_batch.cuda())
        out = torch.squeeze(out,1)
        out = out.detach().cpu().numpy()
        test_pred.append(out)        
        test_true.append(y_batch)
        num_correct1 = (np.rint(out) == y_batch).sum()
        eval_acc = eval_acc + num_correct1
    print('Test Acc: {:.6f}'.format(eval_acc / samples_test))
    print()
#pickle.dump((test_pred, test_true, test_index), open('Data/results/LSTM_pred_'+str(dimension)+'d.pkl',"wb"))

# =============================================================================
# evaluation by AUCs
# =============================================================================

# convert list to array
def listtoarr(y):   
    y_ = np.zeros(len(test_index))
    for i in range(len(y)):
        if i == len(y) - 1:
            y_[i*batch_size:] = y[i]
        else:
            y_[i*batch_size: (i+1)*batch_size] = y[i]
    return y_

# load the target and other nine identities as a new dataset
def load_with_subgroups(path, subgroups):
    target = 'toxicity'
    data = pd.read_csv(path,
                       usecols=['id', target] + subgroups,
                       )
    return data
# compute auc by scikitlearn
def get_auc(y_true, y_pred):
    try:
        return metrics.roc_auc_score(y_true, y_pred)
    except ValueError:
        return np.nan

# compute overall auc, subgroup auc, BPSN auc, BNSP auc, the size of subgroups and score that we call gauc in the codes
def gauc(y_pred, df_y_test):
    sub_auc = {}
    BPSN = {}
    BNSP = {}
    subgroup_size = {}
    column_list = df_y_test.columns.values.tolist()
    subgroups = column_list[2:]
    label = column_list[1]
    df_y_test['pred'] = y_pred
    overall_auc = get_auc(np.rint(df_y_test[label]), y_pred)
    print_data = []
    for group in subgroups:
        # subgroup AUC
        subgroup_examples = df_y_test[(np.rint(df_y_test[group]) ==1)]
        sub_auc[group] = get_auc(np.rint(subgroup_examples[label]), subgroup_examples['pred'])
        # BPSN
        subgroup_negative_examples = df_y_test[(np.rint(df_y_test[group]) == 1) & (np.rint(df_y_test[label]) == 0)]
        non_subgroup_positive_examples = df_y_test[(np.rint(df_y_test[group]) == 0) & (np.rint(df_y_test[label]) == 1)]
        BPSN_examples = subgroup_negative_examples.append(non_subgroup_positive_examples)
        BPSN[group] = get_auc( np.rint(BPSN_examples[label]), BPSN_examples['pred'])
        # BNSP
        subgroup_positive_examples = df_y_test[(np.rint(df_y_test[group]) == 1) & (np.rint(df_y_test[label]) == 1)]
        non_subgroup_negative_examples = df_y_test[(np.rint(df_y_test[group]) == 0) & (np.rint(df_y_test[label]) == 0)]
        BNSP_examples = subgroup_positive_examples.append(non_subgroup_negative_examples)
        BNSP[group] = get_auc(np.rint(BNSP_examples[label]), BNSP_examples['pred'])
        # subgroup size
        subgroup_size[group] = subgroup_negative_examples.shape[0] + subgroup_positive_examples.shape[0]
        print_data.append([group, sub_auc[group], BPSN[group], BNSP[group], subgroup_size[group]])
    print_data = pd.DataFrame(print_data,
                              columns=['sub_group', 'sub_group_AUC', 'BPSN_AUC', 'BNSP_AUC', 'sub_group_size'])
    pd.set_option('display.max_columns', None)
    print(print_data)
    weight = 0.25
    power = -5
    bias_score = np.average([
        power_mean(list(sub_auc.values()), power),
        power_mean(list(BPSN.values()), power),
        power_mean(list(BNSP.values()), power)
    ])
    del df_y_test['pred']
    return (weight * overall_auc) + ((1 - weight) * bias_score) #GAUC

# set the weights of the AUC 
def power_mean(series, p):
    total = sum(np.power(series, p))
    return np.power(total / len(series), 1 / p)

#if __name__ == '__main__':
#    dimension = 50
batch_size = 2048
#test_pred, test_true, test_index=pickle.load(open('Data/results/LSTM_pred_'+str(dimension)+'d.pkl',"rb"))
test_predict = listtoarr(test_pred)
test_label = listtoarr(test_true)
sub_groups = [
        'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
        'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
path = 'Data/test_private.csv'
test_data = load_with_subgroups(path, sub_groups)
test_data[sub_groups] = test_data[sub_groups].fillna(0)
test_data = test_data[test_data['id'].isin(test_index)]
GAUC = gauc(test_predict, test_data)
print(GAUC)



    

