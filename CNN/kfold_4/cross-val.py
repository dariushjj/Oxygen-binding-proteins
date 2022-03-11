import numpy as np
from keras import Model
from keras.layers import Input,Embedding,concatenate,Conv1D,GRU,Activation
from keras.layers import Dropout,Dense,Flatten,BatchNormalization,TimeDistributed
from keras.layers.pooling import MaxPooling1D
import keras
from keras import regularizers
from keras.callbacks import EarlyStopping,ModelCheckpoint
import pandas as pd
from my_utils import(get_gene_ontology,get_go_set,get_anchestors,FUNC_DICT)
import logging
from sklearn.metrics import roc_curve,auc,matthews_corrcoef
from my_utils import *
from keras.preprocessing import sequence
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import os
from sklearn.preprocessing import LabelBinarizer
from keras.utils import to_categorical
from keras.utils import multi_gpu_model
from tensorflow.contrib.metrics import streaming_auc
from sklearn.metrics import matthews_corrcoef,recall_score,f1_score
#import seaborn as sns
from sklearn.metrics import confusion_matrix

#config=tf.ConfigProto()
#config.gpu_options.allow_growth=True
#set_session(tf.Session(config=config))
np.random.seed(1600)  # for reproducibility, needs to be the first 2 lines in a script
os.environ["CUDA_VISIBLE_DEVICES"]="3"

def prepare_data():
    max_sequence_size=800
    logging.debug("Target Gene Ontology :BP ")
    train_file = '/home/jiajunh/CNN/kfold_4/all_train.pkl'
    pssm_file = '/home/wanglei/data/Oxygen/npy'
    test_file = '/home/jiajunh/CNN/kfold_4/all_test.pkl'
    pssm_test_file = '/home/wanglei/data/Oxygen/npy'
    logging.info("Preparing train data ... ")

    features = []
    train_df=pd.read_pickle(train_file)

    def reshape(values):
        values=np.hstack(values).reshape(len(values),len(values[0]))
        return values

    for i,row in train_df.iterrows():
        pdb_id=row['pdb_id']
        extra_feature=get_extra_feature(pssm_file,pdb_id)
        features.append(extra_feature)

    logging.info('Padding sequences to %d ... ' % max_sequence_size)
    X=sequence.pad_sequences(train_df['ngrams'].values,maxlen=max_sequence_size)
    y=train_df['labels'].values
    lb = LabelBinarizer()

    y = lb.fit_transform(y)  # transfer label to binary value

    #y = to_categorical(y)  # transfer binary label to one-hot. IMPORTANT
    X_all=np.array(X)
    y_all=np.array(y)
    pssm_ssacc=np.array(features)
    logging.info("Input shape: %s" % str(X_all.shape))
    logging.info("Output shape: %s" % str(y_all.shape))
    logging.info("pssm and ssacc shape:%s" % str(pssm_ssacc.shape))
    n = X_all.shape[0]

    # randomize to shuffle first
    randomize = np.arange(n)
    np.random.shuffle(randomize)

    X_train = X_all[randomize]
    y_train_temp = y_all[randomize]
    y_train=reshape(y_train_temp)
    X_extra = pssm_ssacc[randomize]

    logging.info('------------------------------------------------')
    logging.info('preparing the test data...')
    features_test = []
    test_df = pd.read_pickle(test_file)
    for i, row in test_df.iterrows():
        pdb_id = row['pdb_id']
        extra_feature = get_extra_feature(pssm_test_file, pdb_id)
        features_test.append(extra_feature)
    logging.info('Padding sequences to %d ... ' % max_sequence_size)
    X_temp = sequence.pad_sequences(test_df['ngrams'].values, maxlen=max_sequence_size)
    y_temp = test_df['labels'].values
    X_test = np.array(X_temp)
    lb = LabelBinarizer()

    y_temp = lb.fit_transform(y_temp )

    y_test_temp = np.array(y_temp)
    y_test = reshape(y_test_temp)
    pssm_ssacc_test = np.array(features_test)
    logging.info("Input shape: %s" % str(X_test.shape))
    logging.info("Output shape: %s" % str(y_test.shape))
    logging.info("pssm and ssacc shape:%s" % str(pssm_ssacc_test.shape))
    return (X_train, X_extra, y_train, X_test, y_test, pssm_ssacc_test)

def save_hist(hist,function):
    model_snapshot_directory = "/home/jiajunh/CNN/kfold_4"
    h1 = hist.history
    epoch = hist.epoch
    acc_ = np.asarray(h1['acc'])
    loss_ = np.asarray((h1['loss']))
    val_acc = np.asarray(h1['val_acc'])
    val_loss = np.asarray(h1['val_loss'])
    acc_and_loss = np.column_stack((epoch, acc_, loss_, val_acc, val_loss))
    save_file_mlp = model_snapshot_directory + '/mlp_run_2' + '.txt'
    with open(save_file_mlp, 'w') as f:
        np.savetxt(save_file_mlp, acc_and_loss, delimiter=" ")

def compute_roc(preds, labels):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    return roc_auc

def compute_mcc(preds, labels):
    # Compute ROC curve and ROC area for each class
    mcc = matthews_corrcoef(labels.flatten(), preds.flatten())
    return mcc

def Get_Average(list):
    sum = 0
    for item in list:
        sum += item
    return sum / len(list)

def create_model_3CNN():
    max_features = 20
    # model = get_cnn_model(num_amino_acids, 2000, 1, target_function)
    # target_function='bp'
    # func_df=pd.read_pickle('../data/'+str(target_function)+'_all.pkl')
    # functions=func_df['functions'].values
    output_node = 9

    input_embed = Input(shape=(800,), name='input_embed')
    input_extra = Input(shape=(800, 20,), name='input_extra')
    embedded = Embedding(max_features, 128, input_length=800)(input_embed)
    x1 = concatenate([embedded, input_extra], axis=2)
    x1 = Conv1D(filters=64, kernel_size=5,activation='relu', strides=1, name='conv1')(x1)
    x1 = Dropout( 0.33923, name='drop1')(x1)
    x1 = Conv1D(filters=128, kernel_size=5,activation='relu', strides=1, name='conv2')(x1)
    x1 = Dropout(0.22174, name='drop2')(x1)

    x1 = Conv1D(filters=32, kernel_size=5,
                activation='relu', strides=1, name='conv3')(x1)
    x1 = Dropout(0.07212, name='drop3')(x1)
    x1 = MaxPooling1D(pool_size=8, strides=32, name='maxpooling')(x1)
    
    x1 = Flatten()(x1)
    x1 = Dense(units=256, activation='relu',kernel_regularizer=regularizers.l1_l2(0.00001), name='dense1')(x1)
    x1 = BatchNormalization()(x1)
    output = Dense(output_node, activation='softmax')(x1)
    model = Model(inputs=[input_embed, input_extra], outputs=output)
    print(model.summary())
    return model

if __name__=='__main__':
    k = 5
    model_snapshot_directory = "/home/jiajunh/CNN/kfold_4"
    X_train, X_extra, y_train, X_test, y_test, pssm_ssacc_test = prepare_data()
    model = create_model_3CNN()
    loss_fn = 'categorical_crossentropy'
    adam = keras.optimizers.Adam(lr=10 ** -3, clipnorm=1.)

    #n = len(X_train) // k
    precisions = list()
    recalls = list()
    #auc_values = list()
    f1scores=list()
    #f1score_best=0
    #precisions2 = list()
    #recalls2 = list()
    #f1scores2=list()
    accuracys=list()

    best_model_path = "/home/jiajunh/CNN/kfold_4/best_cross_model.h5"
    # best_model_path = '../results/single_BP/best_cross_model.h5'

    for i in range(1):
        filepath = model_snapshot_directory + '/'+ '_weights.hdf5'
        # filepath = '../results/single_BP/cross' + str(i) + '_BP.h5'
        model.compile(loss=loss_fn, optimizer=adam, metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, mode='min')
        checkpointer = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1,
                                       save_best_only=True, mode='min')
        callback_list = [early_stopping, checkpointer]

        hist = model.fit(x={'input_embed': X_train, 'input_extra': X_extra},
                         y=y_train,
                         epochs=100, batch_size=64,
                         verbose=1, callbacks=callback_list,
                         validation_split=0.2)
        h1 = hist.history
        epoch=hist.epoch
        preds_validate = model.predict(x={'input_embed': X_test, 'input_extra': pssm_ssacc_test})
        # preds_validate = model.predict(x=X_validate)
        #roc_auc = compute_roc(preds_validate, y_validate)

        #print('ROC AUC: \t %f ' % (roc_auc,))
        #auc_values.append(roc_auc)
        print(y_test)
        print(preds_validate)
        y_validate = [np.argmax(l)+1 for l in y_test]
        preds_validate=[np.argmax(l)+1 for l in preds_validate]
        print(y_validate)
        print(preds_validate)

        save_file = model_snapshot_directory + '/'
        #fig_name = i
        #auc_value, thre = get_auc_bestthre(y_validate, preds_validate, save_file, fig_name)
        TP=list()
        FP=list()
        FN=list()
        TN=list()
        FPRs=list()
        TPRs=list()
        AUCs=list()

        for i in range(9):

                a=[1,2,3,4,5,6,7,8,9]
                a.remove(i+1)


                tp = np.sum(np.logical_and(np.equal(y_validate, i+1), np.equal(preds_validate, i+1)))
                fn = np.sum(np.logical_and(np.equal(y_validate, i+1), np.logical_or(np.logical_or(np.logical_or(np.equal(preds_validate, a[0]),np.equal(preds_validate, a[1])),np.logical_or(np.equal(preds_validate, a[2]),np.equal(preds_validate, a[3]))),np.logical_or(np.logical_or(np.equal(preds_validate, a[4]),np.equal(preds_validate, a[5])),np.logical_or(np.equal(preds_validate, a[6]),np.equal(preds_validate, a[7]))))))
                tn = np.sum(np.logical_and(np.logical_or(np.logical_or(np.logical_or(np.equal(y_validate, a[0]),np.equal(y_validate, a[1])),np.logical_or(np.equal(y_validate, a[2]),np.equal(y_validate, a[3]))),np.logical_or(np.logical_or(np.equal(y_validate, a[4]),np.equal(y_validate, a[5])),np.logical_or(np.equal(y_validate, a[6]),np.equal(y_validate, a[7])))), np.logical_or(np.logical_or(np.logical_or(np.equal(preds_validate, a[0]),np.equal(preds_validate, a[1])),np.logical_or(np.equal(preds_validate, a[2]),np.equal(preds_validate, a[3]))),np.logical_or(np.logical_or(np.equal(preds_validate, a[4]),np.equal(preds_validate, a[5])),np.logical_or(np.equal(preds_validate, a[6]),np.equal(preds_validate, a[7]))))))
                fp = np.sum(np.logical_and(np.logical_or(np.logical_or(np.logical_or(np.equal(y_validate, a[0]),np.equal(y_validate, a[1])),np.logical_or(np.equal(y_validate, a[2]),np.equal(y_validate, a[3]))),np.logical_or(np.logical_or(np.equal(y_validate, a[4]),np.equal(y_validate, a[5])),np.logical_or(np.equal(y_validate, a[6]),np.equal(y_validate, a[7])))), np.equal(preds_validate, i+1)))
                if (tp + fn)==0:
                        recall=0
                        recalls.append(recall)
                else:
                        recall = tp / (tp + fn)
                        recalls.append(recall)
                if (tp + fp)==0:
                        precision =0
                        precisions.append(precision )
                else:
                        precision = tp / (tp + fp)
                        precisions.append(precision )
                accuracy=(tp+tn)/(tp+fp+tn+fn)
                TP.append(tp)
                FP.append(fp)
                FN.append(fn)
                TN.append(tn)
                #FPR=fp/(fp+tn)
                #TPR=tp/(tp+fn)
                #FPRs.append(FPR)
                #TPRs.append(TPR)
                #AUC=auc(FPR,TPR)
                #AUCs.append(AUC)
                accuracys.append(accuracy)
                if (precision + recall)==0:
                        f1score=0
                        f1scores.append(f1score)
                else:
                        f1score = (2 * precision * recall) / (precision + recall)
                        f1scores.append(f1score)
                print('f1score:', f1score)
                print('recall:', recall)
                print('precision:', precision)                
        #FPR=Get_Average(FPRs)
        #TPR=Get_Average(TPRs)
        #AUC=Get_Average(AUCs)
        accuracy= Get_Average(accuracys)
        precision=Get_Average(precisions)
        recall=Get_Average(recalls)
        f1score=Get_Average(f1scores)
        print('f1score:', f1score)
        print('recall:', recall)
        print('precision:', precision)
        print('acc:', accuracy)
    #auc_value = sum(auc_values) / len(auc_values)
    #mcc_value = sum(mcc_values) / len(mcc_values)
    print('value for precision:%.3f ,recall:%.3f  , f1score:%.3f, acc:%.3f' %
          (precision_score(y_validate, preds_validate, average='macro'), recall_score(y_validate, preds_validate,average='macro'), f1_score(y_validate, preds_validate, average='macro'),accuracy))
