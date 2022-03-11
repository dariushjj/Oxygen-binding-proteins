import os
import sys

#fR='C:\\Users\\黄家俊\\Desktop\\9 fasta\\csv\\cytoglobin.csv'#训练集
#fR='C:\\Users\\黄家俊\\Desktop\\9 fasta\\csv\\erythrocruorin.csv'#训练集
#fR='C:\\Users\\黄家俊\\Desktop\\9 fasta\\csv\\flavohemoprotein.csv'#训练集
#fR='C:\\Users\\黄家俊\\Desktop\\9 fasta\\csv\\hemerythrin.csv'#训练集
#fR='C:\\Users\\黄家俊\\Desktop\\9 fasta\\csv\\hemocyanin.csv'#训练集
#fR='C:\\Users\\黄家俊\\Desktop\\9 fasta\\csv\\hemoglobin.csv'#训练集
#fR='C:\\Users\\黄家俊\\Desktop\\9 fasta\\csv\\leghemoglobin.csv'#训练集
#fR='C:\\Users\\黄家俊\\Desktop\\9 fasta\\csv\\myoglobin.csv'#训练集
fR='C:\\Users\\黄家俊\\Desktop\\9 fasta\\csv\\neuroglobin.csv'#训练集


kfold1_train='C:\\Users\\黄家俊\\Desktop\\9 fasta\\kfold1_train.txt'
kfold2_train='C:\\Users\\黄家俊\\Desktop\\9 fasta\\kfold2_train.txt'
kfold3_train='C:\\Users\\黄家俊\\Desktop\\9 fasta\\kfold3_train.txt'
kfold4_train='C:\\Users\\黄家俊\\Desktop\\9 fasta\\kfold4_train.txt'
kfold5_train='C:\\Users\\黄家俊\\Desktop\\9 fasta\\kfold5_train.txt'

count = len(open(fR, 'r').readlines())
k1 = int(count / 5)
k2 = int(2 * count / 5)
k3 = int(3 * count / 5)
k4 = int(4 * count / 5)
num=1
with open(fR, 'r') as f:
    for line in f:
        if k1<num<=count:
            with open(kfold1_train, 'a+') as k11:
                k11.write(line)
        num=num+1
num=1
with open(fR, 'r') as f:
    for line in f:
        if num<=k1:
            with open(kfold2_train, 'a+') as k22:
                k22.write(line)
        if k2<num<=count:
            with open(kfold2_train, 'a+') as k22:
                k22.write(line)
        num=num+1
num=1
with open(fR, 'r') as f:
    for line in f:
        if num<=k2:
            with open(kfold3_train, 'a+') as k33:
                k33.write(line)
        if k3<num<=count:
            with open(kfold3_train, 'a+') as k33:
                k33.write(line)
        num=num+1
num=1
with open(fR, 'r') as f:
    for line in f:
        if num<=k3:
            with open(kfold4_train, 'a+') as k44:
                k44.write(line)
        if k4<num<=count:
            with open(kfold4_train, 'a+') as k44:
                k44.write(line)
        num=num+1
num=1
with open(fR, 'r') as f:
    for line in f:
        if num<=k4:
            with open(kfold5_train, 'a+') as k55:
                k55.write(line)
        num=num+1
print(count)
print(k1)