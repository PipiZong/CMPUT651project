#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pickle
from sklearn import metrics
import pandas as pd
# change the dimension for differnet embedding dictionary


def listtoarr(y):   
    y_ = np.zeros(len(test_index))
    for i in range(len(y)):
        if i == len(y) - 1:
            y_[i*batch_size:] = y[i]
        else:
            y_[i*batch_size: (i+1)*batch_size] = y[i]
    return y_


def load_with_subgroups(path, subgroups):
    target = 'toxicity'
    data = pd.read_csv(path,
                       usecols=['id', target] + subgroups,
                       )
    return data



def get_auc(y_true, y_pred):
    try:
        return metrics.roc_auc_score(y_true, y_pred)
    except ValueError:
        return np.nan


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
    return (weight * overall_auc) + ((1 - weight) * bias_score)


def power_mean(series, p):
    total = sum(np.power(series, p))
    return np.power(total / len(series), 1 / p)

if __name__ == '__main__':
    
    batch_size = 2048
    dim = 300
    has_sent = False
    lr = 0.001
    epoch = 4
    remove_empty = True
    
    fdim = dim
    if has_sent:
        fdim+=1
    #test_pred, test_true, test_index = pickle.load(open('../final_results/lr_'+str(lr)+'/results_full_'+str(fdim)+'/pred_y_'+str(epoch)+'.pkl',"rb"))
    test_pred, test_true, test_index=pickle.load(open('../results/pred_y_'+str(epoch)+'.pkl',"rb"))

    #print('test_pred[6]: '+str(test_pred[6])+', test_true[6]: '+str(test_true[6])+', test_index[6]: '+str(test_index[6]))
    #test_predict = listtoarr(test_pred)
    #test_label = listtoarr(test_true)
    test_index = np.array(test_index)
    test_predict = np.array(test_pred)
    test_label = np.array(test_true)
    #print('test_predict[6]: '+str(test_predict[6])+', test_label[6]: '+str(test_label[6])+', test_index[6]: '+str(test_index[6]))
    if remove_empty:
        with open('../Data/empty_test.pkl','rb') as fin:
            remove = pickle.load(fin)
            
        for rm in remove:
            keep_id = (test_index!=rm)
            test_index = test_index[keep_id]
            test_predict = test_predict[keep_id]
            test_label = test_label[keep_id]
    
    print('total of '+str(len(test_index)) + ' test cases')
    
    sub_groups = [
            'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
            'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
    path = '../Data/test_private.csv'
    test_data = load_with_subgroups(path, sub_groups)
    test_data[sub_groups] = test_data[sub_groups].fillna(0)
    
    test_data = test_data[test_data['id'].isin(test_index)]
    
    fout = open('eval.csv','w')
    fout.write('id'+', '+'pred'+', '+'from_m'+', '+'from_test'+'\n')
        
    
    GAUC = gauc(test_predict, test_data)

