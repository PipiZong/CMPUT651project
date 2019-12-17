#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 13:44:08 2019

part of the code refers to  https://www.kaggle.com/bminixhofer/simple-lstm-pytorch-version
@author: yakun

"""

import pandas as pd
import os
import time
import random
from tqdm._tqdm_notebook import tqdm_notebook as tqdm
from keras.preprocessing import text, sequence
import torch
from torch import nn
from torch.utils import data
from torch.nn import functional as F
import numpy as np
import pickle
from sklearn import metrics
#set seed for reproduction
def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()
# compute the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# load the training data and testing data through dataloader, then train the model and test it by batches
def train_model(model, train, test, loss_fn, output_dim, lr=0.001,
                batch_size=512, n_epochs=4,
                enable_checkpoint_ensemble=True):
    param_lrs = [{'params': param, 'lr': lr} for param in model.parameters()]
    optimizer = torch.optim.Adam(param_lrs, lr=lr)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.6 ** epoch)
    
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
    all_test_preds = []
    checkpoint_weights = [2 ** epoch for epoch in range(n_epochs)]
    
    for epoch in range(n_epochs):
        start_time = time.time()
        
        scheduler.step()
        
        model.train()
        avg_loss = 0.
        
        for train_data in tqdm(train_loader, disable=False):
            x_batch = train_data[:-1]
            y_batch = train_data[-1]

            y_pred = model(*x_batch)            
            loss = loss_fn(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            avg_loss += loss.item() / len(train_loader)
            
        model.eval()
        test_preds = np.zeros((len(test), output_dim))
    
        for i, x_batch in enumerate(test_loader):
            y_pred = sigmoid(model(*x_batch).detach().cpu().numpy())# prediction probability

            test_preds[i * batch_size:(i+1) * batch_size, :] = y_pred

        all_test_preds.append(test_preds)
        elapsed_time = time.time() - start_time
        print('Epoch {}/{} \t loss={:.4f} \t time={:.2f}s'.format(
              epoch + 1, n_epochs, avg_loss, elapsed_time))

    if enable_checkpoint_ensemble:
        test_preds = np.average(all_test_preds, weights=checkpoint_weights, axis=0)# weighted average    
    else:
        test_preds = all_test_preds[-1]
        
    return test_preds
# set some value to zeros and change their dimension
class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # mask some features, (N, K, 1, T)
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x
#build two-layer biLSTM model with embedding, pooling and linear layers    
class NeuralNet(nn.Module):
    def __init__(self, embedding_matrix, num_aux_targets):
        super(NeuralNet, self).__init__()
        embed_size = embedding_matrix.shape[1]
        
        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = SpatialDropout(0.3)
        
        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)
    
        self.linear1 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)
        self.linear2 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)
        
        self.linear_out = nn.Linear(DENSE_HIDDEN_UNITS, 1)
        self.linear_aux_out = nn.Linear(DENSE_HIDDEN_UNITS, num_aux_targets)
        
    def forward(self, x):
        h_embedding = self.embedding(x)
        h_embedding = self.embedding_dropout(h_embedding)
        
        h_lstm1, _ = self.lstm1(h_embedding)
        h_lstm2, _ = self.lstm2(h_lstm1)
        
        # global average pooling
        avg_pool = torch.mean(h_lstm2, 1)
        # global max pooling
        max_pool, _ = torch.max(h_lstm2, 1)
        
        h_conc = torch.cat((max_pool, avg_pool), 1)
        h_conc_linear1  = F.relu(self.linear1(h_conc))
        h_conc_linear2  = F.relu(self.linear2(h_conc))
        
        hidden = h_conc + h_conc_linear1 + h_conc_linear2
        
        result = self.linear_out(hidden)
        aux_result = self.linear_aux_out(hidden)
        out = torch.cat([result, aux_result], 1)
        
        return out
        
# load raw comments without sentiments and Glove dictionary
dimension = 300
traindata = pd.read_csv('Data/train.csv')
testdata = pd.read_csv('Data/test_private.csv')
# keep the same test set as in multi_layer_bilstm_with_sentiment.py
_, _, _, test_index=pickle.load(open('Data/LSTM'+str(dimension)+'d.pkl',"rb"))
testdata = testdata[testdata['id'].isin(test_index)]
comments_train = traindata['comment_text']
comments_test = testdata['comment_text']
comments_test = pd.Series.tolist(comments_test)
emb_path = 'Data/Embedding/glove.6B.' + str(dimension) + 'd.txt'

# remove useless symbols in the raw comments, i.e., preprocessing, then generate the training and testing set respectivily
def remove_useless_symbols(text):
    symbols = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&' + '\n'
    for symbol in symbols:
        text = text.replace(symbol, ' ')
    return text

def generate_new_comments(raw_comments):   
    numsamples = len(raw_comments)
    comments_new = []
    for i in range(numsamples):
        comments_new.append(remove_useless_symbols(raw_comments[i]))
    return comments_new

x_train = generate_new_comments(comments_train)
y_train = traindata['target']
y_train = np.rint(np.array(y_train))
y_aux_train = traindata[['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']]
x_test = generate_new_comments(comments_test)
y_test = testdata['toxicity']
y_test = np.rint(np.array(y_test))

# Tokenize sentenses and index the words, padding each sequence to the same length 220
MAX_LEN = 220
tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(x_train + x_test)

x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)
x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)
x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)
print(len(tokenizer.word_index))

# build Glove dictionary
def load_emb_data(emb_path):
    emb = {}
    with open(emb_path, 'r', encoding='UTF-8') as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            tokens = line.split()
            if len(tokens)%10 == 1:
                emb[tokens[0]] = [float(i) for i in tokens[1:]]
    return emb

embedding = load_emb_data(emb_path)
# generate the embeering matrix
def build_matrix(word_index, dic):
    embedding_matrix = np.zeros((len(word_index) + 1, dimension))
    unknown_words = []
    
    for word, i in word_index.items():
        try:
            embedding_matrix[i] = dic[word]
        except KeyError:
            unknown_words.append(word)
    return embedding_matrix
embedding_matrix = build_matrix(tokenizer.word_index, embedding)

# transfer to tensor data and load by data.TensorDataset and cuda to prepare for training
x_train_torch = torch.tensor(x_train, dtype=torch.long).cuda()
x_test_torch = torch.tensor(x_test, dtype=torch.long).cuda()
y_train_torch = torch.tensor(np.hstack([y_train[:, np.newaxis], y_aux_train]), dtype=torch.float32).cuda()
train_dataset = data.TensorDataset(x_train_torch, y_train_torch)
test_dataset = data.TensorDataset(x_test_torch)
max_features = len(embedding_matrix)
# initialize the muber of model that will be trained, the hidden units of lstm and so on
all_test_preds = []
NUM_MODELS = 2
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS
# train 2 models and average their results as the final predictions of test data
for model_idx in range(NUM_MODELS):
    print('Model ', model_idx)
    seed_everything(1234 + model_idx)    
    model = NeuralNet(embedding_matrix, y_aux_train.shape[-1])
    model.cuda()   
    test_preds = train_model(model, train_dataset, test_dataset, output_dim=y_train_torch.shape[-1], 
                             loss_fn=nn.BCEWithLogitsLoss(reduction='mean'))
    all_test_preds.append(test_preds)
    print()
predictions = np.mean(all_test_preds, axis=0)[:, 0]
#pickle.dump((predictions, y_test, test_index), open('Data/multilabel' + str(dimension) + 'd_nosent.pkl','rb'))

# =============================================================================
# evaluation by AUCs
# =============================================================================

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
    return (weight * overall_auc) + ((1 - weight) * bias_score) # GAUC

# set the weights of the AUC 
def power_mean(series, p):
    total = sum(np.power(series, p))
    return np.power(total / len(series), 1 / p)

sub_groups = [
        'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
        'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
path = 'Data/test_private.csv'
test_data = load_with_subgroups(path, sub_groups)
test_data[sub_groups] = test_data[sub_groups].fillna(0)    
test_data = test_data[test_data['id'].isin(test_index)]        
GAUC = gauc(predictions, test_data)
print(GAUC)