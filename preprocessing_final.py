# ============================================================================
# CMPUT 651 project
# Data preprocessing
# ============================================================================
# original negex is written by Peter Kang, a version for server
# peter.kang@roswellpark.org
# https://github.com/mongoose54/negex/tree/master/negex.python
# ============================================================================
import numpy as np
# import pickle
import pandas as pd
from lib.MNegex import *
import lib.MNegex as MNegex
import _pickle as pickle


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


# load data used in sentiment calculation
def load_negex_data(p_path, n_path, negex_path):
    sent = {}
    # ------------------------------------------------------------
    # now read sentiment data
    print("\rNegex Data Loading                     ", end='')
    with open(n_path, 'r') as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            if line != '' and line[0] != ';':
                sent[line.replace('\n', '')] = int(-1)
    with open(p_path, 'r') as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            if line != '' and line[0] != ';':
                sent[line.replace('\n', '')] = int(1)
    # ------------------------------------------------------------
    # now read rules for checking negation
    rfile = open(negex_path)
    irule = sortRules(rfile.readlines())
    print('  -----Complete!')
    return sent, irule
    # ------------------------------------------------------------


def load_emb_data(emb_path):
    # now read word embedding
    emb = {}
    print("\r   Glove.6B: Loading Embedding         ", end='')
    with open(emb_path, 'r', encoding='UTF-8') as fin:
        while True:
            # print(index)

            line = fin.readline()
            if not line:
                break
            tokens = line.split()
            if len(tokens)%10 == 1:
                emb[tokens[0]] = [float(i) for i in tokens[1:]]
    print('  -----Complete!')
    # ------------------------------------------------------------
    return emb


# load raw data
def load_data(path, type):
    # read the data
    print("\rData loading: %-25s" % type, end='')
    if type == 'train':
        target = 'target'
    elif type == 'test':
        target = 'toxicity'
    else:
        raise Exception('Wrong type!!!')
    data = pd.read_csv(path,
                       usecols=['id', target, 'comment_text'],
                       memory_map=True,
                       # nrows=20000
                       )

    print('  -----Complete!')
    return data


def tagging(data, sent, irule, type='train', round=True):
    print('Data Tagging: ' + type)
    save_path = 'Data/processed/' + type + '_data.pkl'
    tagged_data = []
    for index, row in data.iterrows():
        if index % 100 == 0 or index == len(data) - 1:
            print("\r   Tagging     %8s / %-13s" % (index + 1, data.shape[0]), end='')
        line = row['comment_text'].strip().lower().split()
        line = remove_useless_symbols(line)  # clean strings
        word_list = []
        while '' in line:
            line.remove('')
        # length = int(min(max(len(line)/2, 1), 6))
        length = int(6)
        for w in line:
            if w in sent:
                word_list.append(w)
        # tagging sent words
        tagger = MNegex.negTagger(sentence=line, phrases=word_list, rules=irule, negP=False, scope_len=length).getNegDict()
        sentence = []
        # generate new words with 1 or 2 to represent positive or negative
        for w in line:
            if w in tagger.keys():
                if tagger[w][0] == -1:
                    w_sent = -sent[w]
                else:
                    w_sent = sent[w]
                del (tagger[w][0])
                if w_sent == -1:
                    w_sent = 2
                w = w + str(int(w_sent))
            sentence.append(w)
        data_id = row['id']
        if 'target' in data.columns:
            target = row['target'] 
        else:
            target = row['toxicity']
        if round:
            target = int(np.rint(target))
        tagged_data.append([data_id, sentence, target])
    with open(save_path, 'wb') as f:
        pickle.dump(tagged_data, f)
    print('Data Dumped: ' + save_path)
    print('  -----Complete!')
    del tagged_data
    return


def new_emb(emb, sent):
# this is to expand the length of the dict to contain rewritten words such as "bad2"
    print('\r Adding Sent', end='')
    for w in sent:
        if w in emb:
            emb[w + '1'] = emb[w].copy()
            emb[w + '2'] = emb[w].copy()
            emb[w + '1'][-1] = 1
            emb[w + '2'][-1] = -1
    print('  -----Complete!')
    return emb


if __name__ == '__main__':
    negative_words_path = 'Data/negex/negative-words.txt'
    positive_words_path = 'Data/negex/positive-words.txt'
    emb_file = [
                'glove.6B.50d',
                'glove.6B.100d',
                'glove.6B.200d',
                'glove.6B.300d',
                # 'glove.840B.300d'
        ]
    train_data_path = 'Data/train.csv'
    test_data_path = 'Data/test_private.csv'
    negex_triggers_path = 'Data/negex/negex_triggers.txt'
    print('===============================')
    print('      Data Preprocessing')
    print('===============================')
    sent_words, irules = load_negex_data(positive_words_path, negative_words_path, negex_triggers_path)
    print("-------------------------------")
    round_target = True
    # round_target = False
    # tagging words in train data with sentiments
    train_data = load_data(train_data_path, type='train')
    tagging(train_data, sent_words, irules, type='train', round=round_target)
    del train_data
    # tagging words in test data with sentiments
    test_data = load_data(test_data_path, type='test')
    tagging(test_data, sent_words, irules, type='test', round=round_target)
    del test_data, irules
    print("-------------------------------")
    # create new embedding dict with additional d.
    for index, name in enumerate(emb_file):
        path = 'Data/Embedding/' + name + '.txt'
        print("Processing with " + name + ":")
        emb_matrix = load_emb_data(path)
        print('\r Padding Zero', end='')
        for item in emb_matrix:
            emb_matrix[item].append(float(0))
        print('  -----Complete!')
        emb_matrix = new_emb(emb_matrix, sent_words)
        save_path = 'Data/Embedding/processed/' + name + '_sent.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(emb_matrix, f)
        print('Data Dumped: ' + save_path)
    print('===============================')
    print('           Finished')
    print('===============================')
