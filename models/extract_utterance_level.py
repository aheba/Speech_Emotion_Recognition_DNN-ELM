"""
Created on Tue May 23 17:24:02 2017

@author: eesungkim
"""
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from sklearn import mixture
from six.moves import urllib
import collections
from sklearn import svm
from sklearn.metrics import accuracy_score
import itertools
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
from utils import infolog, normalize_Zscore, normalize_MinMax, dense_to_one_hot, batch_creator
from utils import makedirs
import tflearn
import pickle
from models.elm.elm import ELMClassifier, GenELMClassifier
from models.elm.random_layer import RandomLayer, MLPRandomLayer, RBFRandomLayer, GRBFRandomLayer


log = infolog.log

n_hidden_ELM = 120  
rbf_width = 0.1
n_iter_optimiz_elm=300

# Parameters
training_epoch = 50
batch_size = 128
display_step = 5

dim_hidden_1 = 256
dim_hidden_2 = 256
dim_hidden_3 = 256


def extract_utterance_level_features(idx, path_save):
    tf.reset_default_graph()    
    tf.set_random_seed(371)
    filename = "%s/CV_0%s/Frame_CV_0%s.pickle"  %(path_save,idx,idx)
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)

    X_Frame_train   = data['X_Frame_train'].astype('float32')
    y_Frame_train   = data['y_Frame_train']
    idx_Frame_train = data['idx_Frame_train']
    X_Frame_test    = data['X_Frame_test'].astype('float32')
    y_Frame_test    = data['y_Frame_test']
    idx_Frame_test  = data['idx_Frame_test']
    y_train         = data['y_Utter_train']
    y_test          = data['y_Utter_test']

    n_classes=y_test.max()+1
    y_train_onehot=dense_to_one_hot(y_Frame_train,n_classes)
    y_test_onehot=dense_to_one_hot(y_Frame_test,n_classes)


    log ('Training Deep Neural Network............................')
    for d in ['/gpu:%s'%0]:
        with tf.device(d):

            X = tf.placeholder(tf.float32, [None, X_Frame_train.shape[1]])
            Y = tf.placeholder(tf.float32, [None, n_classes])

            weights  = {
                    #he init : stddev=np.sqrt(2 / dim_input)
                    'encoder_h1': tf.Variable(tf.random_normal([X_Frame_train.shape[1], dim_hidden_1],stddev=np.sqrt(2 / X_Frame_train.shape[1]))),
                    'encoder_h2': tf.Variable(tf.random_normal([dim_hidden_1, dim_hidden_2],stddev=np.sqrt(2 / dim_hidden_1))),
                    'encoder_h3': tf.Variable(tf.random_normal([dim_hidden_2, dim_hidden_3],stddev=np.sqrt(2 / dim_hidden_2))),
                    'encoder_output': tf.Variable(tf.random_normal([dim_hidden_3, n_classes],stddev=np.sqrt(2 / dim_hidden_3)))
            }
            biases = {
                    'encoder_b1': tf.Variable(tf.zeros([dim_hidden_1])),
                    'encoder_b2': tf.Variable(tf.zeros([dim_hidden_2])),
                    'encoder_b3': tf.Variable(tf.zeros([dim_hidden_3])),
                    'encoder_output': tf.Variable(tf.zeros([n_classes]))
            }

            Z1 = tf.matmul( X, weights['encoder_h1'] ) + biases['encoder_b1']
            l1 = tf.nn.relu(Z1)

            Z2 = tf.matmul(l1, weights['encoder_h2']) + biases['encoder_b2']
            l2 = tf.nn.relu(Z2)

            Z3 = tf.matmul(l2, weights['encoder_h3']) + biases['encoder_b3']
            l3 = tf.nn.relu(Z3)    

            output       = tf.matmul(l3, weights['encoder_output']) + biases['encoder_output']
            pred         = tf.nn.softmax(output)     

            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y))
            optimizer = tf.train.AdamOptimizer().minimize(cost)
            
            correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(Y,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    with tf.Session(config = tf.ConfigProto(allow_soft_placement = True,device_count={'GPU':1},gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        log ('Training CV %s Deep Neural Network............................'%idx)
        for epoch in range(training_epoch):
                _, c = sess.run([optimizer,cost], feed_dict = {X: X_Frame_train, Y: y_train_onehot})
                if (epoch % display_step) == 0:
                    train_accuracy=sess.run([accuracy], feed_dict={X: X_Frame_train, Y: y_train_onehot})
                    test_accuracy=sess.run([accuracy], feed_dict={X: X_Frame_test, Y: y_test_onehot})
                    log("Epoch: {:02d} \t\tCost={:.2f} \tTrainAcc: {:.2f} \tTestAcc: {:.2f}".format(epoch,c, train_accuracy[0],test_accuracy[0]))

        X_Frame_DNN_train = sess.run(pred,feed_dict={X: X_Frame_train})  
        X_Frame_DNN_test  = sess.run(pred,feed_dict={X: X_Frame_test})   


    X_Frame_DNN_train,X_Frame_DNN_test=normalize_Zscore(X_Frame_DNN_train,X_Frame_DNN_test)

    utterFeatList = []
    for i in range(idx_Frame_train.max() + 1):
        frames = X_Frame_DNN_train[idx_Frame_train == i, :]
        if frames.size != 0:
            utter_feat = np.hstack([np.amax(frames, axis=0), np.amin(frames, axis=0), np.mean(frames, axis=0), np.mean(frames > 0.2, axis=0) ])
            utterFeatList.append(utter_feat)
        X_train = np.vstack(utterFeatList)
    
    utterFeatList_test = []
    for i in range(idx_Frame_test.max() + 1):
        frames = X_Frame_DNN_test[idx_Frame_test == i, :]
        if frames.size != 0:
            utter_feat = np.hstack([np.amax(frames, axis=0), np.amin(frames, axis=0), np.mean(frames, axis=0), np.mean(frames > 0.2, axis=0) ])
            utterFeatList_test.append(utter_feat)
        X_test = np.vstack(utterFeatList_test)
    log("Utterance-Level-Features are extracted.")

    log ("Classifying Speech Emotions using Utter-Level features............................")

    """Extreme Learning Machine"""
    rhl = RBFRandomLayer(n_hidden=200, rbf_width=0.1)
    elmr = GenELMClassifier(hidden_layer=rhl)
    elmr.fit(X_train, y_train)
    y_pred=elmr.predict(X_test)

    uar=0
    cnf_matrix = confusion_matrix(y_test, y_pred)
    diag=np.diagonal(cnf_matrix)
    for index,i in enumerate(diag):
        uar+=i/collections.Counter(y_test)[index]
    test_unweighted_accuracy=uar/len(cnf_matrix)
    accuracy=[]
    accuracy.append(test_weighted_accuracy*100)
    accuracy.append(test_unweighted_accuracy*100)
    
    a = ['Ang', 'Hap','Neu','Sad']
    # Compute confusion matrix
    cnf_matrix = np.transpose(cnf_matrix)
    cnf_matrix = cnf_matrix*100 / cnf_matrix.astype(np.int).sum(axis=0)
    cnf_matrix = np.transpose(cnf_matrix).astype(float)
    cnf_matrix = np.around(cnf_matrix, decimals=1)

    #accuracy per class 
    conf_mat = (cnf_matrix.diagonal()*100)/cnf_matrix.sum(axis=1)
    conf_mat = np.around(conf_mat, decimals=2)

    log('==============[ [%s] ]=============='%idx)
    log('Feature Dimension: %d'%X_train.shape[1])
    log('Confusion Matrix:\n%s'%cnf_matrix)
    log('Accuracy per classes:\n%s'%conf_mat)
    log("WAR\t\t\t:\t%.2f %%" %(test_weighted_accuracy*100))
    log("UAR\t\t\t:\t%.2f %%" %(test_unweighted_accuracy*100))

    return np.around(np.array(accuracy),decimals=1)


def main(ROOT_PATH, MODEL_NAME, nFolders):
    MODEL_PATH = os.path.join(ROOT_PATH, "datasets/IEMOCAP" ,MODEL_NAME)
    makedirs(MODEL_PATH)

    logFolderName='exp/log/%s'%MODEL_NAME
    logFileName='%s/Extract_Uttrance_Level_Feats.log'%logFolderName
    log_path = os.path.join(ROOT_PATH, logFolderName)
    makedirs(log_path)
    log_path = os.path.join(ROOT_PATH, logFileName)
    infolog.init(log_path, ROOT_PATH)
    
    
    acc_stat1=np.zeros(2)
    for idx in range(nFolders):    
        acc_stat1 += extract_utterance_level_features(idx, MODEL_PATH)
        
    log('='*50)
    log('Total Accuracy[ SVM ][ WAR UAR ]')
    log('[XX][ %s ]'%(acc_stat1/nFolders))
    log('='*50)
    
if __name__ == '__main__':
    main()
