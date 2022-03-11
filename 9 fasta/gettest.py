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


kfold1_test='C:\\Users\\黄家俊\\Desktop\\9 fasta\\kfold1_test.txt'
kfold2_test='C:\\Users\\黄家俊\\Desktop\\9 fasta\\kfold2_test.txt'
kfold3_test='C:\\Users\\黄家俊\\Desktop\\9 fasta\\kfold3_test.txt'
kfold4_test='C:\\Users\\黄家俊\\Desktop\\9 fasta\\kfold4_test.txt'
kfold5_test='C:\\Users\\黄家俊\\Desktop\\9 fasta\\kfold5_test.txt'

count = len(open(fR, 'r').readlines())
k1 = int(count / 5)
k2 = int(2 * count / 5)
k3 = int(3 * count / 5)
k4 = int(4 * count / 5)
num=1
with open(fR, 'r') as f:
    for line in f:
        if num<=k1:
            with open(kfold1_test, 'a+') as k11:
                k11.write(line)
        if k1<num<=k2:
            with open(kfold2_test, 'a+') as k22:
                k22.write(line)
        if k2<num<=k3:
            with open(kfold3_test, 'a+') as k33:
                k33.write(line)
        if k3<num<=k4:
            with open(kfold4_test, 'a+') as k44:
                k44.write(line)
        if k4<num<=count:
            with open(kfold5_test, 'a+') as k55:
                k55.write(line)
        num=num+1
print(count)
print(k1)