# -*-coding:utf-8-*-
from __future__ import print_function
import numpy as np
from utils import *
import math
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation,GRU
from keras.layers import Embedding,concatenate,Flatten
from keras.layers import Conv1D
from keras.layers.pooling import GlobalAveragePooling1D,GlobalMaxPooling1D,MaxPooling1D
from keras.layers import Input
from keras.layers.normalization import BatchNormalization
config = tf.ConfigProto(allow_soft_placement=True)
set_session(tf.Session(config=config))
import logging
import json
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping,ModelCheckpoint
from hyperas.distributions import uniform,choice
from hyperopt import Trials,STATUS_OK,tpe
from hyperas import optim
import keras
from keras import regularizers
#from go_relationship_dict import get_dict,get_pos_neg
#import globalvars
from keras.utils import multi_gpu_model
#from alt_model_checkpoint import AltModelCheckpoint
from tensorflow.contrib.metrics import streaming_auc
from sklearn.metrics import matthews_corrcoef,recall_score,f1_score
from my_utils import *
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from adabound import AdaBound
from sklearn.preprocessing import LabelBinarizer
from keras.utils import to_categorical
import matplotlib.pyplot as plt

#config=tf.ConfigProto()
#config.gpu_options.allow_growth=True
#set_session(tf.Session(config=config))
np.random.seed(1600)  # for reproducibility, needs to be the first 2 lines in a script
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def prepare_data():
    max_sequence_size=800
    logging.debug("Target Gene Ontology :BP ")
    train_file = '/home/jiajunh/CNN/kfold_3/all_train.pkl'
    pssm_file = '/home/wanglei/data/Oxygen/npy'
    test_file = '/home/jiajunh/CNN/kfold_3/all_test.pkl'
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
    for i,row in test_df.iterrows():
        pdb_id = row['pdb_id']
        extra_feature = get_extra_feature(pssm_test_file, pdb_id)
        features_test.append(extra_feature)
    logging.info('Padding sequences to %d ... ' % max_sequence_size)
    X_temp = sequence.pad_sequences(test_df['ngrams'].values, maxlen=max_sequence_size)
    y_temp = test_df['labels'].values
    X_test=np.array(X_temp)
    y_test_temp=np.array(y_temp)
    y_test=reshape(y_test_temp)
    pssm_ssacc_test=np.array(features_test)
    logging.info("Input shape: %s" % str(X_test.shape))
    logging.info("Output shape: %s" % str(y_test.shape))
    logging.info("pssm and ssacc shape:%s" % str(pssm_ssacc_test.shape))
    return (X_train,X_extra,y_train,X_test,y_test,pssm_ssacc_test)

def train(X_train,X_extra,y_train,X_test,y_test,pssm_ssacc_test):
    # num_amino_acids = 22  # alphabet of proteins +1 for padding
    model_snapshot_directory = "/home/jiajunh/CNN/kfold_3"
    max_features = 20
    # model = get_cnn_model(num_amino_acids, 2000, 1, target_function)
    #target_function='bp'
    #func_df=pd.read_pickle('../data/'+str(target_function)+'_all.pkl')
    #functions=func_df['functions'].values
    output_node=9

    input_embed = Input(shape=(800,), name='input_embed')
    input_extra = Input(shape=(800, 20,), name='input_extra')
    embedded = Embedding(max_features, 128, input_length=800)(input_embed)
    x1 = concatenate([embedded, input_extra], axis=2)
    x1 = Conv1D(filters={{choice([32,64, 128])}}, kernel_size={{choice([3,5,7, 11, 15, 19, 21])}},
                activation='relu', strides=1,name='conv1')(x1)
    x1 = Dropout({{uniform(0, 1)}},name='drop1')(x1)
    x1 = Conv1D(filters={{choice([32,64, 128])}}, kernel_size={{choice([3,5,7, 11, 15, 19, 21])}},
                activation='relu', strides=1,name='conv2')(x1)
    x1 = Dropout({{uniform(0, 1)}},name='drop2')(x1)

    x1 = Conv1D(filters={{choice([32,64, 128])}}, kernel_size={{choice([3,5,7, 11, 15, 19, 21])}},
                activation='relu', strides=1,name='conv3')(x1)
    x1 = Dropout({{uniform(0, 1)}},name='drop3')(x1)
    x1 = MaxPooling1D(pool_size={{choice([4,8,16, 32])}}, strides={{choice([4,8,16, 32, 64])}},name='maxpooling')(x1)
    x1 = BatchNormalization()(x1)

    x = Flatten()(x1)
    x = Dense(units={{choice([64,128,256])}}, activation='relu',
              kernel_regularizer=regularizers.l1_l2({{choice([0.001, 0.0001, 0.00001])}}),name='dense1')(x)    
    x = BatchNormalization()(x)
    output = Dense(output_node, activation='softmax')(x)
    model = Model(inputs=[input_embed, input_extra], outputs=output)
    print(model.summary())

    loss_fn = 'categorical_crossentropy'
    adam = keras.optimizers.Adam(lr=10 ** -3,clipnorm=1.)
    optim = AdaBound(lr={{choice([1e-03,1e-04])}}, final_lr=0.1, gamma=1e-03, weight_decay=0, amsbound=False)

    filepath = model_snapshot_directory + '/'+ '_weights.hdf5'
    gpu_model = multi_gpu_model(model, 2)
    # model.compile(loss=loss_fn, optimizer=adam, metrics=['accuracy'])
    gpu_model.compile(loss=loss_fn, optimizer=optim, metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, mode='min')
    checkpointer = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=1,
                                   save_best_only=True, save_weights_only=True, mode='min')
    callback_list = [early_stopping, checkpointer]
    hist = gpu_model.fit(x={'input_embed': X_train, 'input_extra': X_extra},
                         y=y_train,
                         epochs={{choice([100, 150, 200])}}, batch_size={{choice([32,64,128,256])}},
                         verbose=1, callbacks=callback_list,
                         validation_split=0.2)
    score, acc =gpu_model.evaluate(x={'input_embed':X_train,'input_extra':X_extra},y=y_train)
    h1 = hist.history
    epoch=hist.epoch
    acc_ = np.asarray(h1['acc'])
    loss_ = np.asarray((h1['loss']))
    val_acc = np.asarray(h1['val_acc'])
    val_loss = np.asarray(h1['val_loss'])
    acc_and_loss = np.column_stack((epoch,acc_, loss_, val_acc, val_loss))

    plt.plot(h1['loss'])
    plt.plot(h1['val_loss'])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train","val"],loc="upper left")
    plt.show()
    save_file_mlp = model_snapshot_directory + '/mlp_run_' + '.txt'
    with open(save_file_mlp, 'w') as f:
        np.savetxt(save_file_mlp, acc_and_loss, delimiter=" ")
    print('train accuracy:', acc)
    print('train score:',score)
    print("Test:------------------------------------------------------------")
    # score_test,acc_test = gpu_model.evaluate(x={'input_embed': X_test, 'input_extra': pssm_ssacc_test}, y=y_test)
    # y_preds=gpu_model.predict(x={'input_embed':X_test,'input_extra':pssm_ssacc_test})

    # get the best threshold according to mcc value
    return {'loss': -acc, 'status': STATUS_OK, 'model': gpu_model}

if __name__ == "__main__":
    # set up logging
    results_dir = "/home/jiajunh/CNN/kfold_3"
    '''
    exp_id = get_experiment_id(results_dir)
    set_logging_params(results_dir, exp_id)
    logging.info("Experiment ID: %s" % exp_id)
    print("Experiment ID: " + exp_id)
    '''
    # let's also capture console output
    import sys
    oldStdout = sys.stdout
    file = open("/home/jiajunh/CNN/kfold_3/" +  "console.txt", 'w')
    sys.stdout = file
    best_run,best_model=optim.minimize(model=train,data=prepare_data,algo=tpe.suggest,
                                       max_evals=100,trials=Trials())
    X_train, X_extra, y_train, X_test, y_test, pssm_ssacc_test=prepare_data()
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
    json.dump(best_run, open("/home/jiajunh/CNN/kfold_3/model_hyper.txt", 'w'))
    sys.stdout = oldStdout
    print("Done.")