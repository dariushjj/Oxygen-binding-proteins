# coding=utf-8
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc,matthews_corrcoef,recall_score,precision_score,f1_score
from xml.etree import ElementTree as ET
from collections import deque
from keras import backend as K
import pandas as pd
#from go_relationship_dict import get_dict,get_pos_neg
import json
plt.switch_backend('agg')

MAX_LENGTH=700
GRAM_LENGTH=3
BIOLOGICAL_PROCESS = 'GO:0008150'
MOLECULAR_FUNCTION = 'GO:0003674'
CELLULAR_COMPONENT = 'GO:0005575'
FUNC_DICT = {
    'cc': CELLULAR_COMPONENT,
    'mf': MOLECULAR_FUNCTION,
    'bp': BIOLOGICAL_PROCESS}
EXP_CODES = ['EXP', 'IDA', 'IPI', 'IMP', 'IGI', 'IEP', 'TAS', 'IC']

def get_gene_ontology(filename='../data/go-basic.obo'):
    # Reading Gene Ontology from OBO Formatted file
    go = dict()
    obj = None
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line == '[Term]':
                if obj is not None:
                    go[obj['id']] = obj
                obj = dict()
                obj['is_a'] = list()
                obj['part_of'] = list()
                obj['regulates'] = list()
                obj['is_obsolete'] = False
                continue
            elif line == '[Typedef]':
                obj = None
            else:
                if obj is None:
                    continue
                l = line.split(": ")
                if l[0] == 'id':
                    obj['id'] = l[1]
                elif l[0] == 'is_a':
                    obj['is_a'].append(l[1].split(' ! ')[0])
                elif l[0] == 'name':
                    obj['name'] = l[1]
                elif l[0] == 'is_obsolete' and l[1] == 'true':
                    obj['is_obsolete'] = True
    if obj is not None:
        go[obj['id']] = obj
    for go_id in list(go.keys()):
        if go[go_id]['is_obsolete']:
            del go[go_id]
    # go_id is the children of the nodes in is_a
    for go_id, val in go.items():
        if 'children' not in val:
            val['children'] = set()
        for p_id in val['is_a']:
            if p_id in go:
                if 'children' not in go[p_id]:
                    go[p_id]['children'] = set()
                go[p_id]['children'].add(go_id)
    return go

'''
return all sequences length of which is lower than 700
'''
def get_sequence():
    data=list()
    with open('../data/cath_seqs_single_domain.txt','r') as f:
        for line in f:
            items=line.strip().split(',')
            if len(items[1])<=MAX_LENGTH:
                data.append(items[1])
    with open('../data/dataset2.txt','r') as f:
        for line in f:
            items=line.strip().split(',')
            if len(items[1])<=MAX_LENGTH:
                data.append(items[1])
    return data

def get_ngrams():
    seqs=get_sequence()
    ngrams=set()
    for seq in seqs:
        for i in range(len(seq)-GRAM_LENGTH+1):
            ngrams.add(seq[i:(i+GRAM_LENGTH)])
    ngrams=list(sorted(ngrams))
    df=pd.DataFrame({'ngrams':ngrams})
    df.to_pickle('../data/ngrams.pkl')

# transfer sequencese into 3grams,file_name means sequence
#  file,for binary classification
def sequence_ngram_index(functions_file,file_name,target_function,mode):
    with open(functions_file) as fn_file:
        protein_function_map = json.load(fn_file)
    ngram_df=pd.read_pickle('../data/ngrams.pkl')
    vocab={}
    for key,gram in enumerate(ngram_df['ngrams']):
        vocab[gram]=key+1
    gram_len=len(ngram_df['ngrams'][0])
    print('Vocabulary size:',len(vocab))
    sequences=list()
    ngrams=list()
    pdb_ids=list()
    labels=list()
    pos_examples,neg_examples=0,0
    go_ancestor=get_dict()
    with open(file_name,'r') as f:
        for line in f:
            items=line.strip().split(',')
            pdb_id=items[0]
            seq=items[1]
            if len(seq)>MAX_LENGTH:
                continue
            try:
                functions=[fn for fn in protein_function_map[pdb_id]]
                pdb_ids.append(pdb_id)
                count=0
                for func in functions:
                    if get_pos_neg(go_ancestor,target_function,func):
                        count+=1
                if count>0:
                    labels.append(1)
                    pos_examples+=1
                else:
                    labels.append(0)
                    neg_examples+=1

                sequences.append(seq)
                grams = np.zeros((len(seq) - gram_len + 1,), dtype='int32')
                for i in range(len(seq) - gram_len + 1):
                    grams[i] = vocab[seq[i:(i + gram_len)]]
                ngrams.append(grams)
            except KeyError:
                pass
    print("pos_examples:%d,neg_examples:%d"%(pos_examples,neg_examples))
    res_df=pd.DataFrame({
        'pdb_id':pdb_ids,
        'sequences':sequences,
        'ngrams':ngrams,
        'labels':labels
    })
    print(len(res_df))
    # res_df.to_pickle('../data/'+str(target_function)+'_'+str(mode)+'.pkl')
    return res_df

# make the y label a 2dim vector,using the code in the deepgo
def sequence_ngram_multilabel(mode,function):
    GO_ID=FUNC_DICT[function]
    go=get_gene_ontology('../data/go-basic.obo')
    FUNCTION=function
    func_df=pd.read_pickle('../data/'+str(FUNCTION)+'all1.pkl')
    functions=func_df['functions'].values
    func_set=get_go_set(go,GO_ID)
    print(len(functions))
    gos=list()
    proteins=list()
    labels=list()
    sequences=list()
    ngrams=list()

    ngram_df=pd.read_pickle('../data/ngrams.pkl')
    vocab={}

    # index all the ngrams,here is 3 gram
    go_indexes=dict()
    for ind,go_id in enumerate(functions):
        go_indexes[go_id]=ind

    for key,gram in enumerate(ngram_df['ngrams']):
        vocab[gram]=key+1
    gram_len=len(ngram_df['ngrams'][0])
    print('Vocabulary size:',len(vocab))
    count=0
    count1=0

    if mode=='train':
        seq_df=pd.read_pickle('../data/train_seq_go_all.pkl')
    elif mode=='test':
        seq_df=pd.read_pickle('../data/test_seq_go_all.pkl')
    for i,row in seq_df.iterrows():
        go_list=row['gos']
        go_set=set()
        for go_id in go_list:
            if go_id in func_set:
                go_set|=get_anchestors(go,go_id) # get the ancestor go term of the annotated sequence
        if not go_set or GO_ID not in go_set:
            count1+=1
            continue
        go_set.remove(GO_ID)
        seq = row['seqs']
        if len(seq)>MAX_LENGTH:
            count+=1
            continue
        gos.append(go_list)
        proteins.append(row['pdb_id'])
        sequences.append(row['seqs'])

        grams = np.zeros((len(seq) - gram_len + 1,), dtype='int32')
        for i in range(len(seq) - gram_len + 1):
            grams[i] = vocab[seq[i:(i + gram_len)]]
        ngrams.append(grams)

        label = np.zeros((len(functions),), dtype='int32')
        for go_id in go_set:
            if go_id in go_indexes:
                label[go_indexes[go_id]] = 1
        labels.append(label)

    res_df=pd.DataFrame({
        'pdb_id':proteins,
        'sequences':sequences,
        'ngrams':ngrams,
        'labels':labels,
        'gos':gos
    })
    print('length greater than 700:',count)
    print('not in go :',count1)
    print(len(res_df))
    res_df.to_pickle('../data/'+str(mode)+'/'+str(function)+'_all_seq_'+str(mode)+'.pkl')
    return res_df

def get_single_seq_anno(pdb_id,mode):
    if mode=='train':
        file_name='../data/cath_single_go.txt'
    else:
        file_name='../data/dataset2_gos.txt'
    with open(file_name) as f:
        function_map=json.load(f)
    annos=function_map[pdb_id]
    return annos

def get_extra_feature(pssm_file,pdb_id_name):
    #pdb_id=pdb_id_name[:4].lower()+pdb_id_name[4:]
    file_temp = pssm_file + '/'+pdb_id_name + '.npy'
    extra_feature = np.load(file_temp)
    return extra_feature

def get_anchestors(go, go_id):
    go_set = set()
    q = deque()
    q.append(go_id)
    while(len(q) > 0):
        g_id = q.popleft()
        go_set.add(g_id)
        for parent_id in go[g_id]['is_a']:
            if parent_id in go:
                q.append(parent_id)
    return go_set

def get_parents(go, go_id):
    go_set = set()
    for parent_id in go[go_id]['is_a']:
        if parent_id in go:
            go_set.add(parent_id)
    return go_set

def get_go_set(go, go_id):
    go_set = set()
    q = deque()
    q.append(go_id)
    while len(q) > 0:
        g_id = q.popleft()
        go_set.add(g_id)
        for ch_id in go[g_id]['children']:
            q.append(ch_id)
    return go_set

def f_score(labels, preds):
    preds = K.round(preds)
    tp = K.sum(labels * preds)
    fp = K.sum(preds) - tp
    fn = K.sum(labels) - tp
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    return 2 * p * r / (p + r)

def filter_specific(go, gos):
    go_set = set()
    for go_id in gos:
        go_set.add(go_id)
    for go_id in gos:
        anchestors = get_anchestors(go, go_id)
        anchestors.discard(go_id)
        go_set -= anchestors
    return list(go_set)

def read_fasta(lines):
    seqs = list()
    info = list()
    seq = ''
    inf = ''
    for line in lines:
        line = line.strip()
        if line.startswith('>'):
            if seq != '':
                seqs.append(seq)
                info.append(inf)
                seq = ''
            inf = line[1:]
        else:
            seq += line
    seqs.append(seq)
    info.append(inf)
    return info, seqs

def cal_all_mcc(y_true,y_pred,thresholds):
    '''
    :param y_true: the true label of the sample
    :param y_pred: the predicted label of the sample
    :param thresholds: a list of threshold
    :return: a list of mccs and the threshold for which the mcc is the maximum
    '''
    print('get all mccs according to all the thresholds')
    print('y_true.shape:',y_true.shape)
    print('y_pred.shape:',y_pred.shape)
    print('length of thresholds:',len(thresholds))

    Mccs = [] # all mccs
    max_thre = 0
    max_mcc = 0
    for threshold in thresholds:
        y_pred_temp=np.where(y_pred>threshold,1,0)
        MCC=matthews_corrcoef(y_true,y_pred_temp)
        # print('TP:%s,TN:%s,FP:%s,FN:%s'%(TP,TN,FP,FN))
        if MCC > max_mcc:
            max_thre = threshold
        Mccs.append(MCC)
    return Mccs, max_thre

def get_auc_bestthre(y_true,y_pred,save_file,fig_name):
    '''
    draw the roc curve of the predicted result
    :param y_true: the true label of the sample
    :param y_pred: the predicted label of the sample
    :param save_file: the file_name of the figure to save
    :param figname: the name of the figure
    :return: the auc value and the threshold of the max ks
    '''
    fpr, tpr, threshold = roc_curve(y_true, y_pred)
    auc_value = auc(fpr, tpr)
    ks_value_temp = tpr - fpr
    ks_value = ks_value_temp.tolist()
    # get the threshold when the ks is the maximum
    thre_index = ks_value.index(max(ks_value))
    thre = threshold[thre_index]
    print("the best threshold when the ks_value is the maximum:", thre)
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % auc_value)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('estimate the function of the model using ROC-AUC')
    plt.legend(loc="lower right")
    plt.savefig(save_file+fig_name + '.png')
    return (auc_value, thre)

def calc_confusion_matrix(y_true,y_pred,threshold=None):
    '''
    # calc the TN,TP,FN,FP value
    :param y_true: the true label of the sample
    :param y_pred: the predicted label of the sample
    :param threshold: a single value
    :return: TP,TN,FP,FN value
    '''
    TP, TN, FN, FP = 0, 0, 0, 0
    # get the y_pred value of the given threshold
    if threshold is None:
        for i in range(len(y_pred)):
            if y_true[i] == y_pred[i]:
                if y_true[i] == 1:
                    TP += 1
                else:
                    TN += 1
            else:
                if y_true[i] == 1:
                    FN += 1
                else:
                    FP += 1
    else:
        y_pred_temp = np.where(y_pred > threshold, 1, 0)
        for i in range(len(y_pred_temp)):
            if y_true[i] == y_pred_temp[i]:
                if y_true[i] == 1:
                    TP += 1
                else:
                    TN += 1
            else:
                if y_true[i] == 1:
                    FN += 1
                else:
                    FP += 1
    return (TP,TN,FN,FP)

def calc_acc(TP,TN,FN,FP):
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return accuracy

# def predict_on_thresholds(model,X_predict,y_true,batch_size,savefile,fig_name):
#     y_pred = model.predict(X_predict, batch_size=batch_size, verbose=1)
#     # y_eval = y_predict.astype(np.float)
#     print(y_true.shape)
#     print(y_pred.shape)
#
#     auc_value,threshold=get_auc_bestthre(y_true,y_pred,savefile,fig_name)
#     TP,TN,FN,FP=calc_confusion_matrix(y_true,y_pred,threshold)
#     print("threshold:",threshold)
#     print('TP:%5d'%TP)
#     print('FN:%5d'%FN)
#     print('TN:%5d'%TN)
#     print('FP:%5d'%FP)
#
#     print("auc_value:%f" % auc_value)
#     recall=calc_recall(TP,TN,FN,FP)
#     print("recall=%f"%recall)
#
#     accuracy=calc_acc(TP,TN,FN,FP)
#     print("accuracy=%f"%accuracy)
#
#     precision=calc_precision(TP,TN,FN,FP)
#     print("precision=%f"%precision)
#
#     F1_score=calc_F1_score(TP,TN,FN,FP)
#     print("F1_score=%f"%F1_score)
#
#     MCC = calc_MCC(TP,TN,FN,FP)
#     print("MCC=%f"%MCC)

def predict_on_thresholds_test(y_true,y_pred,savefile,fig_name):
    print(y_true.shape)
    print(y_pred.shape)
    thresholds=np.linspace(0.0001,1.,10000)
    # auc_value, threshold = get_auc_bestthre(y_true, y_pred, savefile, fig_name)

    # get the threshold when the mcc is maximum
    Mccs,threshold=cal_all_mcc(y_true,y_pred,thresholds)
    TP, TN, FN, FP = calc_confusion_matrix(y_true, y_pred, threshold)
    y_pred_temp=np.where(y_pred>threshold,1,0)
    print("threshold:", threshold)
    print('TP:%5d' % TP)
    print('FN:%5d' % FN)
    print('TN:%5d' % TN)
    print('FP:%5d' % FP)

    # print("auc_value:%f" % auc_value)
    recall = recall_score(y_true,y_pred_temp)
    print("recall=%f" % recall)

    accuracy = calc_acc(TP, TN, FN, FP)
    print("accuracy=%f" % accuracy)

    precision = precision_score(y_true,y_pred_temp)
    print("precision=%f" % precision)

    F1_score = f1_score(y_true,y_pred_temp)
    print("F1_score=%f" % F1_score)

    MCC = matthews_corrcoef(y_true,y_pred_temp)
    print("MCC=%f" % MCC)

# if __name__=='__main__':
#     # sequence_ngram_index('../data/cath_seqs_single_domain.txt')
#     sequence_ngram_multilabel('test','cc')