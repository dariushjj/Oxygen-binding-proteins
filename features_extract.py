# coding=utf-8
# get the PSSM features and ACC and ss features,then concat these features
# PSSM feature:the first 50,then normalize it
# acc: transfer into 2 dim vector using ont-hot
# ss :transfer into 3 dim vector using onr-hot

import numpy as np
import math
import os
import re

file_PSSM='/home/wanglei/data/Oxygen/neg'#PSSM文件路径
dir='/home/wanglei/data/Oxygen/neg'#pssm文件路径
fR='C:\\Users\\黄家俊\\Desktop\\9new_oxygen-binging_proteins\\train\\erythrocruorin.txt'#训练集
fR_test='C:\\Users\\黄家俊\\Desktop\\9new_oxygen-binging_proteins\\test\\erythrocruorin.txt'#测试集
name=[]
def get_names_of_pssm():
    list = []
    for lists in os.listdir('/home/wanglei/data/Oxygen/neg'):
        pat = ".+\.(pssm)"  # 匹配文件名正则表达式
        pattern = re.findall(pat, lists)  # 进行匹配
        if pattern:
            items = lists.strip().split('.')
            list.append(items[0])
            #print(list)
    return list

'''
# to know if the seq has pssm,return the seq which has these features
def exist_judge():
    test_feature_extract=[]
    #get the seq which does not have PSSM and SSACC features
    for lists in os.listdir(dir):
        if lists[-4:]=='pssm':
            # seq=lists[:4].lower()+lists[4:6]
            # seq=lists[:4]+lists[4:6]
            seq=lists
            name.append(seq)

    with open(fR_test,'r') as f:
        for line in f:
            arr=line.strip().split(',')
            seq=arr[0]
            if seq not in name:
                test_feature_extract.append(seq)
                print(seq)
    return test_feature_extract

# turn the sequences into integer arrays
def sequence_to_indices(sequence,acid_letters):
    try:
        indices = [acid_letters.index(c) for c in list(sequence)]
        return indices
    except Exception:
        print(sequence)
        raise Exception
'''
# normalize the pssm scores into 0-1,return pssm features
def get_PSSM_normalize(pssm_file):
    pssm=[]
    with open(pssm_file) as f:#打开文件
        for lines in f:#f为文件的一行
            if len(lines)<5:#舍弃空行
                continue
            if not lines[4].isdigit():#舍弃最后的一些数据
                continue
            temp=lines.strip().split()#去除空格，得到单个数据
            for i in range(2,22):#20维数据
                temp[i]=float(temp[i])#归一化
            pssm.append(temp[2:22])#加入列表
    pssm = np.array(pssm)  # 转换成数组
    max = np.amax(pssm)  # 找出最大的元素
    min = np.amin(pssm)
    for i in range(0,pssm.shape[0]):
        for j in range(0,pssm.shape[1]):
            pssm[i][j] = (pssm[i][j] - min) / (max - min)
    return pssm
# get all sequences which has to extract pssm and ssacc features
'''
def get_seq_name(file_name):
    names=[]
    with open(file_name,'r') as f:
        for line in f:
            arr=line.strip().split(',')
            seq=arr[0][:4].upper()+arr[0][4:]
            names.append(seq)
    return names

# get names which is not dealt with ,that is which pdb does not have npy files
def get_names_without_npy():
    seqs=list()
    seqs_no=list()
    for lists in os.listdir('C:\\Users\\黄家俊\\Desktop\\9new_oxygen-binging_proteins\\xsxsxsxsxs'):
        seq_temp=lists[:6]
        seq=seq_temp[:4].upper()+seq_temp[4:]
        seqs.append(seq)
    with open('C:\\Users\\黄家俊\\Desktop\\9new_oxygen-binging_proteins\\train\\erythrocruorin.txt','r') as f:
        for line in f:
            items=line.strip().split(',')
            if items[0] not in seqs:
                seqs_no.append(items[0])
    return seqs_no
'''

if __name__=='__main__':
    # names=get_seq_name(fR_test)
    # print(len(names))
    '''
    names=get_names_without_npy()
    print(len(names))
    '''
    name_of_pssm=get_names_of_pssm()
    # print(names)
    # names=['1ASU_A']
    # extra_lists=exist_judge()
    # print(len(extra_lists))
    '''
    code below make files according to the seq names,which is used to store the ssacc features
    '''
       # file_ssacc_extra='/home/siqihuang/feature_extraction/test_data/ssacc/'
    # # file to store the sequence which has to extract features,for test dataset
    # extra_file='/home/siqihuang/feature_extraction/test_data/seqNameLists.txt'
    # with open(extra_file,'w') as f:
    #     f.truncate()
    #     for aa in extra_lists:
    #         f.write(aa+'\n')
    #
    # # make files which name is the element in the extra_list
    # for seq in extra_lists:
    #     seq_name=seq[:4]+seq[5:]
    #     file_temp=file_ssacc_extra+seq_name
    for seq in name_of_pssm:
    #for seq in names:
        file_pssm=file_PSSM+'/'+seq+'.pssm'
        #print(seq)
        #print(file_pssm)
        pssm=get_PSSM_normalize(file_pssm)
        print(pssm.shape)
    #        ssacc_name=seq[:4]+seq[5:]
        # format XXXX_X
        # ssacc_name=seq
    #       file_ss=file_SSACC+ssacc_name+'/'+ssacc_name+'.ss'
    #       file_acc=file_SSACC+ssacc_name+'/'+ssacc_name+'.acc'
    #       ss=np.array(get_ss(file_ss))
        # print(ss.shape)
    #       acc=np.array(get_acc(file_acc))
        # print(acc.shape)
    #        feature=np.concatenate([pssm,ss,acc],axis=1)#按行拼接三个特征
        # 将矩阵补0，补成700*25维数据
        zero_row=800-pssm.shape[0]
        # 这里要判断是否序列长度大于800，如果大于800，那么这个值为负，不能补0
        if zero_row<0:
            print(0)
            continue
        padding_zeros=np.zeros((zero_row,20),dtype='float32')#为不足700的行准备好对应行数的0
        last_feature=np.row_stack((pssm,padding_zeros))#按行进行添加
        # print(last_feature.shape)
        file_to_save=file_PSSM+'/'+seq+'.npy'
        txt_to_save = file_PSSM+'/'+seq+ '.txt'
        np.save(file_to_save,last_feature)
        np.savetxt(txt_to_save,last_feature)









