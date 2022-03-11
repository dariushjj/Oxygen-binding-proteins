from __future__ import print_function
import pickle
import os
import sys
import pandas as pd
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import re
import numpy as np
'''
items=[0.999999]
print(len(str(items[0])))
if len(str(items[0]))<15:
    zero_row = 15 - len(str(items[0]))
    while zero_row>0:
        items[0]=(str(items[0])+'0')
        zero_row-=1

print(items[0])
'''
class hot_dna:
    def __init__(self, seq):
        # get sequence into an array
        seq_array = array(list(seq))
        categories = 'auto'
        # integer encode the sequence
        label_encoder = LabelEncoder()
        integer_encoded_seq = label_encoder.fit_transform(seq_array)

        # one hot the sequence
        onehot_encoder = OneHotEncoder(sparse=False)
        # reshape because that's what OneHotEncoder likes
        integer_encoded_seq = integer_encoded_seq.reshape(len(integer_encoded_seq), 1)
        onehot_encoded_seq = onehot_encoder.fit_transform(integer_encoded_seq)

        # add the attributes to self

        self.sequence = seq
        self.integer = integer_encoded_seq
        self.onehot = onehot_encoded_seq

fR='C:\\Users\\黄家俊\\Desktop\\9 fasta\\kfold5_test.txt'#训练集
#fR='C:\\Users\\黄家俊\\Desktop\\9new_oxygen-binging_proteins\\sequence_label_id\\multitype_test\\all.txt'#训练集

def get_label():
    data=[]
    with open(fR,'r') as f:
        for line in f:
            items=line.strip().split(',')##strip去除items后面的空格内容
            data.append(items[21])
            print(items[21])

    return data

def get_feature():
    data=[]
    with open(fR,'r') as f:
        for line in f:
            items=line.split(',')
            '''
            for i in range(1,21):
                if len(str(items[i]))<15:
                    zero_row = 15 - len(str(items[i]))
                    while zero_row>0:
                        items[i]=(str(items[i])+'0')
                        zero_row-=1
            '''
            for i in range(1,21):
                items[i]=float(items[i])
            data.append((items[1:21]))
    return data




label=get_label()
print(label)
#print(label)
feature=get_feature()
print(id)
#print(id)
df=pd.DataFrame({'labels':label,'pdb_feature':feature})

df.to_pickle('C:\\Users\\黄家俊\\Desktop\\9 fasta\\kfold5\\all_test.pkl')
train_df=pd.read_pickle('C:\\Users\\黄家俊\\Desktop\\9 fasta\\kfold5\\all_test.pkl')
#df.to_pickle('C:\\Users\\黄家俊\\Desktop\\9new_oxygen-binging_proteins\\sequence_label_id\\multitype_test\\all.pkl')
#train_df=pd.read_pickle('C:\\Users\\黄家俊\\Desktop\\9new_oxygen-binging_proteins\\sequence_label_id\\multitype_test\\all.pkl')

print(train_df['labels'])
print(train_df['pdb_feature'])
print(train_df['labels'].values.shape)
print(train_df['pdb_feature'].values.shape)
print(train_df['pdb_feature'].values[0])

