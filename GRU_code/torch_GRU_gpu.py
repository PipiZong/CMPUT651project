import os
from GRUNet import *
import torch
import numpy as np
import pandas as pd
import _pickle as pickle
import time
from torch.autograd import Variable
from torch.utils import data
import random

# removes symbols so the program is not confused when identifying the words
def remove_useless_symbols(sent):
    symbols = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
    for symbol in symbols:
        for i in range(len(sent) - 1, -1, -1):
            if 'http:' in sent[i]:
                del sent[i]
            elif 'https:' in sent[i]:
                del sent[i]
            else:
                sent[i] = sent[i].replace(symbol, '')
    return sent



# split the given dataset such that the number of positive samples and negative samples have reasonable ratio(such as 1:1)
def sample_equal(x, y, n_p_ratio):
    n_p_ratio = int(n_p_ratio)
    x = np.array(x)
    #print('x0:\n' + str(x[0]))
    # print('sample training data so number of pos and neg cases are equal')
    total = len(y)
    total_pos = np.sum(y[:, 1])
    pos_x = x[(y == [0,1])[:, 0]]
    pos_x = np.repeat(pos_x,n_p_ratio,axis = 0)
    neg_x = x[(y == [1,0])[:, 0]]
    neg_x = neg_x[random.sample(range(total-total_pos), int(total_pos*n_p_ratio))]

    return np.concatenate((pos_x, neg_x)), np.concatenate((np.array([[0,1]] * (total_pos * n_p_ratio)), np.array([[1,0]] * (total_pos * n_p_ratio))))

# directory of pickled train dataset
train_dir = "Data/train_data.pkl"
# directory of pickled test dataset
test_dir = "Data/test_data.pkl"
# the maximum length for encoded sentences, the ones shorter are padded by 0s and longer ones are cut
max_sentence_length = 220

# the directory of pickled embedding file, dimension has 100,200 and 300 options
word2vec = "Data/emb/glove.6B.300d_sent.pkl"

# the sembedding size without sentiments. the number should match of word2vec
embedding_dim = 300
# embedding size for hidden layer in GRU
hidden_size = 128
# batch size when training and testing
batch_size = 2048
# learning rate for the model
learning_rate = 0.001
# dropout rate for GRU
drop_out_rate = 0.5

# display loss information after this number of batches
loss_interval = 100
# evaluate on test set every this number of epoches
eval_interval = 1
# maxinum number of epoches to train
max_epoch = 20

# the name of file that records acc on test set, and it is stored under 'results/'
log_file = 'results/acc.log'
# the name of file that records the predicted y from test set, used for calculating AUC
pred_y_file = 'results/pred_y_'
# these two are used for function sample_equal
sample = False
sample_n_p_ratio = 1


dummy = ' '
# use train dataset as test to see how fast the model is learning/overfitting
test_on_train = False
# flag that controls whether to use sentiment as a feature for training
has_sent = True
# flag that determines if y is a scalar or a vector of length 2(mainly for different loss functions)
binary_out = True

# if both are False, then unknown words have embedding of all 0s
# skip current word if it is not in glove embedding
skip_unknown = False
# generate a random vector for unknown word.
random_unknown = True

# save a copy of test dataset with all empty samples removed
save_empty_test = True

def train():
    # use GPU for faster training
    isCUDA = torch.cuda.is_available()
    if not isCUDA:
        raise Exception('Need Cuda to Run!')
    torch.cuda.set_device(1)

    # load embeddings
    print('load embedding')
    t = time.time()
    with open(word2vec, 'rb') as fin:
        emb = pickle.load(fin)
    print('took ' + str(time.time() - t) + ' sec')

    # load train dataset
    print('load train data')
    t = time.time()
    with open(train_dir, 'rb') as fin:
        train = pickle.load(fin)

    skip_empty = 0
    x_text = []
    y = []
    word_idx = 1

    word_dict = {dummy: 0} #mapping word to index
    weight_matrix = [] #mapps id to embedding
    # initialize with dummy value for padding
    if has_sent:
        weight_matrix.append([0] * (embedding_dim+1))
    else:
        weight_matrix.append([0] * embedding_dim)
        
    # process each sentence, convert words into integers, and keep a mapping between the integers and embedding matrix
    for row in train:
        #skip empty samples for training
        if row[1] == []:
            skip_empty += 1
            continue
        tmp = []
        for w in row[1]:
            if w in word_dict:
                tmp.append(word_dict[w])
            else:
                if w not in emb:
                    if not skip_unknown:
                        if random_unknown:
                            tmp_emb = [random.randrange(0, 1) for iter in range(embedding_dim)]
                            if has_sent:
                                tmp_emb.append(0)
                            tmp.append(word_idx)
                            word_dict[w] = word_idx
                            word_idx += 1
                            weight_matrix.append(tmp_emb)
                        else:
                            tmp.append(word_dict[dummy])
                else:
                    word_dict[w] = word_idx
                    tmp.append(word_idx)
                    if has_sent:
                        weight_matrix.append(emb[w])
                    else:
                        weight_matrix.append(emb[w][:-1])
                    word_idx += 1
        if len(tmp) > max_sentence_length:
            tmp = tmp[:max_sentence_length]
        elif len(tmp) < max_sentence_length:
            tmp = [0] * (max_sentence_length - len(tmp)) + tmp
        x_text.append(tmp)
        cy = row[2]
        if cy >= 0.5:
            if binary_out:
                cy = [0, 1]
            else:
                cy = [1]
        else:
            if binary_out:
                cy = [1, 0]
            else:
                cy = [0]
        y.append(cy)
    
    x_text = np.array(x_text)
    y = np.array(y)
    del train

    print('took ' + str(time.time() - t) + ' sec')

    # load test data
    print('load test data')
    t = time.time()
    with open(test_dir, 'rb') as fin:
        test = pickle.load(fin)

    x_test = []
    y_test = []
    test_id = []
    test_true = []
    empty_id = []
    for row in test:
        # x_test.append([word_dict[w] for w in row[1]])
        test_id.append(row[0])
        tmp = []
        if row[1]==[]:
            empty_id.append(row[0])
        for w in row[1]:
            if w in word_dict:
                tmp.append(word_dict[w])
            else:
                if w not in emb:
                    if not skip_unknown:
                        if random_unknown:
                            tmp_emb = [random.randrange(0, 1) for iter in range(embedding_dim)]
                            if has_sent:
                                tmp_emb.append(0)
                            tmp.append(word_idx)
                            word_dict[w] = word_idx
                            word_idx += 1
                            weight_matrix.append(tmp_emb)
                        else:
                            tmp.append(word_dict[dummy])
                else:
                    word_dict[w] = word_idx
                    tmp.append(word_idx)
                    if has_sent:
                        weight_matrix.append(emb[w])
                    else:
                        weight_matrix.append(emb[w][:-1])
                    word_idx += 1
        if len(tmp) > max_sentence_length:
            tmp = tmp[:max_sentence_length]
        elif len(tmp) < max_sentence_length:
            tmp = [0] * (max_sentence_length - len(tmp)) + tmp
        x_test.append(tmp)
        cy = row[2]
        if cy >= 0.5:
            test_true.append(1)
            if binary_out:
                cy = [0,1]
            else:
                cy = [1]
        else:
            test_true.append(0)
            if binary_out:
                cy = [1,0]
            else:
                cy = [0]
        y_test.append(cy)
    y_test = np.array(y_test)
    test_size = len(test)
    del test
    del emb
    weight_matrix = np.array(weight_matrix)
    print('took ' + str(time.time() - t) + ' sec')

    if save_empty_test:
        with open('Data/empty_test.pkl','wb') as fout:
            pickle.dump(empty_id,fout)
    
    if sample:
        x_text, y = sample_equal(x_text, y, sample_n_p_ratio)

    if test_on_train:
        x_test = np.copy(x_text)
        y_test = np.copy(y)
        
    print('each train set has ' + str(len(y)) + ' entries')

    print('create model')
    # generate the GRU model
    if binary_out:
        model = GRUNet(weight_matrix, hidden_size, 2, drop_out_rate).cuda().float()
    else:
        model = GRUNet(weight_matrix, hidden_size, 1, drop_out_rate).cuda().float()
    # loss function
    if binary_out:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.BCEWithLogitsLoss()
    
    # optimizer with Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_x = torch.tensor(x_text, dtype=torch.long)
    #train_x = torch.tensor(x_text)
    del x_text
    train_y = torch.tensor(y)
    #print('y t size '+str(y.shape))
    #print('y train size '+str(train_y.size()))
    del y

    train_dataset = data.TensorDataset(train_x, train_y)
    train_dataset = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    del train_x
    del train_y

    test_x = torch.tensor(x_test, dtype=torch.long)
    #test_x = torch.tensor(x_test)
    del x_test
    test_y = torch.tensor(y_test)
    del y_test

    test_id = torch.tensor(test_id)
    test_dataset = data.TensorDataset(test_x, test_y)
    test_dataset = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    del test_x
    del test_y

    print('start iterations')
    for epoch in range(max_epoch):
        print('epoch: ' + str(epoch))
        model.train()
        #train on each batch
        for i, (x_batch, y_batch) in enumerate(train_dataset, 1):
            hidden = model.init_hidden(x_batch.size(0)).cuda().float() #new hidden
            optimizer.zero_grad()
            y_pred, hidden = model(x_batch.cuda(), hidden, False)

            if binary_out:
                _,actual_y = torch.max(y_batch,1)
                loss = criterion(y_pred, actual_y.cuda())
            else:
                loss = criterion(y_pred, y_batch.float().cuda())

            # gradient descent
            loss.backward()
            optimizer.step()

            if i % loss_interval == 0:
                print('at batch ' + str(i) + ', loss: ' + str(loss.item()))

        # evaluate on test set
        if epoch % eval_interval == 0:
            print('eval at epoch: ' + str(epoch))

            model.eval()
            # variables to keep record of true positive, true negative, false positive and false negative
            TP, TN, FP, FN = 0, 0, 0, 0
            size = 0
            correct = 0
            
            y_out = []
            pos_pred = 0
            list_id = []
            
            # test set also divided into batches
            for k, (x_batch, y_batch) in enumerate(test_dataset, 1):
                
                hidden = model.init_hidden(x_batch.size(0)).cuda().float()
                y_pred, hidden = model(x_batch.cuda(), hidden , False)
                
                size += x_batch.size(0)
                y = y_batch.cpu().numpy()

                for j in range(x_batch.size(0)):

                    

                    if binary_out:
                        r = torch.argmax(y_pred[j]).item()
                        if y[j][0] < y[j][1]:
                            real = 1
                        else:
                            real = 0
                        
                    else:
                        r = y_pred[j].item()
                        real = y[j][0]
                    if r >= 0.5:
                        r = 1.0
                        pos_pred+=1
                    else:
                        r = 0.0
                    if r == real:
                        if r == 1.0:
                            TP += 1
                        else:
                            TN += 1
                    else:
                        if r == 1.0:
                            FP += 1
                        else:
                            FN += 1
                    y_out.append(r)
                            
            
            # calculate the accuracy, precision and recall
            acc = (TP + TN) / size
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            
            print('epoch:' + str(epoch) + ', acc: ' + str(acc) + ', precision: ' + str(precision) + ', recall: ' + str(recall) + '\n')
            # save the acc records
            with open(log_file, 'a') as fout:
                fout.write(str(epoch) + ', ' + str(acc)+'\n')
            # save the predict ys for AUC evaluation
            with open(pred_y_file + str(epoch) + '.pkl', 'wb') as fout:
                pickle.dump((y_out,test_true,[id.item() for id in test_id]), fout)
            


train()
