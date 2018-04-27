"""
Created on Tue May 23 17:24:02 2017

@author: eesungkim
"""
import numpy as np
import os
import scipy
from time import localtime
import os
import resampy
import librosa
from sklearn import mixture
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from utils import infolog, normalize_Zscore, normalize_MinMax
from utils import makedirs
from scipy.special import expit
import librosa
from utils.speech_features import mfcc, pitch_based_features
import pickle
log = infolog.log

def convert_segments(featMat, segSize):
    sideSegSize=segSize//2
    resultMat = featMat
    result=np.zeros(resultMat.shape[1]*segSize)
    for idx in range(featMat.shape[0]-segSize+1):
        result =np.vstack([result,np.hstack(np.vstack([featMat[idx:idx+sideSegSize, :], featMat[idx+sideSegSize, :] ,featMat[idx+sideSegSize+1:idx+2*sideSegSize+1, :]]))])
    return result[1:,:]

def extract_lowlevel_features(audio_path, segmentNum):
    (fs,signal) = scipy.io.wavfile.read(audio_path)
    feats_mfcc = mfcc(y=signal, sr=fs, n_mfcc=13, n_fft=512, win_length=400, hop_length=160)
    pitch_based=pitch_based_features(signal, fs, win_length=400, hop_length=160)
    #zcr = librosa.feature.zero_crossing_rate(signal, frame_length=400, hop_length=160, center=True)
    feats=np.concatenate((feats_mfcc.T, pitch_based.T), axis=1)
    feats_delta = librosa.feature.delta(feats)
    feats_all_data=np.concatenate((feats, feats_delta), axis=1)
    featSegment = convert_segments(feats_all_data,segmentNum)  
    return featSegment,featSegment.shape[0]  

def extract_segment_level_features(path_DB_dir, path_save, idx, segmentNum):
    all_dataFolder=os.path.join(path_save, 'all_data')
    makedirs(all_dataFolder)        

    uttrFeatsList = []
    uttrLabelList = []
    uttrIdxList = []
    frameLabelList =[]
    emotionLabels = {}
    tmpIdx,labelnum = 0,0
    subdirs = [x[0] for x in os.walk(path_DB_dir)]
    for subdir in subdirs:
        if subdir != path_DB_dir:
            label = subdir[-3:]
            emotionLabels[label] = labelnum
            labelnum += 1
            for fn in os.listdir(subdir):
                audio_path = os.path.join(subdir, fn)
                if fn.endswith("n.wav"):
                    featSegment,segNum = extract_lowlevel_features(audio_path, segmentNum)

                    uttrFeatsList.append(featSegment)
                    idices = np.zeros(segNum) + tmpIdx
                    uttrIdxList.append(idices)
                    tmpIdx += 1
                    uttrLabelList.append(emotionLabels[label])
                    for i in range(segNum):
                        frameLabelList.append(emotionLabels[label])  
    uttrFeatures = np.vstack(uttrFeatsList).astype('float32')
    uttrIdx = np.hstack(uttrIdxList).astype('int')
    uttrLabel = np.hstack(uttrLabelList).astype('int')
    frameLabel = np.hstack(frameLabelList).astype('int')

    mu = np.mean(uttrFeatures, axis = 0)
    sigma = np.std(uttrFeatures, axis = 0)
    uttrFeatures = (uttrFeatures - mu) / sigma   
    
    data={'X_frame':uttrFeatures,'idx_frame':uttrIdx,'y_uttr':uttrLabel,'y_frame':frameLabel}

    filename ="%s/all_data/session_%s.pickle" %(path_save,idx)
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    log('[0%s]Completed to extract segment level features sucessfully. %s'%(idx,path_DB_dir))   


def save_cross_validation_5folder(idx, nFolders, path_save):
    stacked_X_Frame_test = []
    stacked_y_Utter_test = []
    stacked_y_Frame_test = []
    stacked_X_Frame_train = []
    stacked_y_Utter_train = []
    stacked_y_Frame_train = []
    idx_Frame_train       = np.array([0])
    idx_Frame_test       = np.array([0])
    path_folder="%s/CV_0%s/"%(path_save,idx)
    makedirs(path_folder)
    print ("|---------------- Processing [ Cross Validation 0%s ] ----------------|"%idx)   
    for j in range(nFolders):
        if (idx == j):  # move i th data to i th folder as Test data

            filename ="%s/all_data/session_%s.pickle" %(path_save,j)
            with open(filename, 'rb') as handle:
                data = pickle.load(handle)
            tmp_X_Frame_test   = data['X_frame']
            tmp_y_Utter_test   = data['y_uttr']
            tmp_y_Frame_test   = data['y_frame']
            tmp_idx_Frame_test = data['idx_frame']

            stacked_X_Frame_test.append(tmp_X_Frame_test)
            stacked_y_Utter_test.append(tmp_y_Utter_test)
            stacked_y_Frame_test.append(tmp_y_Frame_test)

            if idx_Frame_test[-1] == 0:
                tmp_idx_Frame_test =idx_Frame_test[-1]+tmp_idx_Frame_test
                idx_Frame_test     =np.hstack(tmp_idx_Frame_test).astype('int')
            else:
                tmp_idx_Frame_test =idx_Frame_test[-1]+1+tmp_idx_Frame_test
                idx_Frame_test     =np.hstack((idx_Frame_test,tmp_idx_Frame_test)).astype('int')
        else : # move remainder datas to i th folder as Train data

            filename2 ="%s/all_data/session_%s.pickle" %(path_save,j)
            with open(filename2, 'rb') as handle:
                data2 = pickle.load(handle)

            tmp_X_Frame_train   = data2['X_frame']
            tmp_y_Utter_train   = data2['y_uttr']
            tmp_y_Frame_train   = data2['y_frame']
            tmp_idx_Frame_train = data2['idx_frame']

            stacked_X_Frame_train.append(tmp_X_Frame_train)
            stacked_y_Utter_train.append(tmp_y_Utter_train)
            stacked_y_Frame_train.append(tmp_y_Frame_train)

            if idx_Frame_train[-1] == 0:
                tmp_idx_Frame_train =idx_Frame_train[-1]+tmp_idx_Frame_train
                idx_Frame_train     =np.hstack(tmp_idx_Frame_train).astype('int')
            else:
                tmp_idx_Frame_train =idx_Frame_train[-1]+1+tmp_idx_Frame_train
                idx_Frame_train     =np.hstack((idx_Frame_train,tmp_idx_Frame_train)).astype('int')

    X_Frame_test   =np.vstack(stacked_X_Frame_test).astype('float32')  
    y_Utter_test   =np.hstack(stacked_y_Utter_test).astype('int')         
    y_Frame_test   =np.hstack(stacked_y_Frame_test).astype('int')     

    X_Frame_train   =np.vstack(stacked_X_Frame_train).astype('float32')  
    y_Utter_train   =np.hstack(stacked_y_Utter_train).astype('int')         
    y_Frame_train   =np.hstack(stacked_y_Frame_train).astype('int')      

    log('Improvised Number of the utterance Set: %s'%(idx_Frame_train[-1]+idx_Frame_test[-1]+2))
    log("Frame-Level-Features are extracted.")
    log("X_Frame_train:(%s, %s)"  %X_Frame_train.shape)
    log("y_Frame_train:%s"        %y_Frame_train.shape)
    log("idx_Frame_train: %s"     %y_Frame_train.shape)
    log("X_Frame_test: (%s, %s)"  %X_Frame_test.shape)
    log("y_Frame_test: %s"        %y_Frame_test.shape)
    log("idx_Frame_test: %s"      %idx_Frame_test.shape)

    data={'X_Frame_train':X_Frame_train, 'y_Frame_train':y_Frame_train, 'idx_Frame_train':idx_Frame_train,
                'X_Frame_test':X_Frame_test, 'y_Frame_test':y_Frame_test, 'idx_Frame_test':idx_Frame_test,
                'y_Utter_train':y_Utter_train, 'y_Utter_test':y_Utter_test}

    filename  = "%s/CV_0%s/Frame_CV_0%s.pickle"  %(path_save,idx,idx)
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)



def save_cross_validation_folder(idx, nFolders, path_save):
    stacked_X_Frame_train = []
    stacked_y_Utter_train = []
    stacked_y_Frame_train = []
    idx_Frame_train       = np.array([0])
    path_folder="%s/CV_0%s/"%(path_save,idx)
    makedirs(path_folder)

    log ("|---------------- Processing [ Cross Validation 0%s ] ----------------|"%idx)   
    for j in range(nFolders):
        if idx == j :  
            filename ="%s/all_data/session_%s.pickle" %(path_save,j)
            with open(filename, 'rb') as handle:
                data = pickle.load(handle)
            X_Frame_test   = data['X_frame']
            y_Utter_test   = data['y_uttr']
            y_Frame_test   = data['y_frame']
            idx_Frame_test = data['idx_frame']
        else : 
            filename2 ="%s/all_data/session_%s.pickle" %(path_save,j)
            with open(filename2, 'rb') as handle:
                data2 = pickle.load(handle)
            tmp_X_Frame_train   = data2['X_frame']
            tmp_y_Utter_train   = data2['y_uttr']
            tmp_y_Frame_train   = data2['y_frame']
            tmp_idx_Frame_train = data2['idx_frame']

            stacked_X_Frame_train.append(tmp_X_Frame_train)
            stacked_y_Utter_train.append(tmp_y_Utter_train)
            stacked_y_Frame_train.append(tmp_y_Frame_train)

            if idx_Frame_train[-1] == 0:
                tmp_idx_Frame_train =idx_Frame_train[-1]+tmp_idx_Frame_train
                idx_Frame_train     =np.hstack(tmp_idx_Frame_train).astype('int')
            else:
                tmp_idx_Frame_train =idx_Frame_train[-1]+1+tmp_idx_Frame_train
                idx_Frame_train     =np.hstack((idx_Frame_train,tmp_idx_Frame_train)).astype('int')
    X_Frame_train   =np.vstack(stacked_X_Frame_train).astype('float32')  
    y_Utter_train   =np.hstack(stacked_y_Utter_train).astype('int')         
    y_Frame_train   =np.hstack(stacked_y_Frame_train).astype('int')      

    log('Improvised Number of the utterance Set: %s'%(idx_Frame_train[-1]+idx_Frame_test[-1]+2))
    log("Frame-Level-Features are extracted.")
    log("X_Frame_train:(%s, %s)"  %X_Frame_train.shape)
    log("y_Frame_train:%s"        %y_Frame_train.shape)
    log("idx_Frame_train: %s"     %y_Frame_train.shape)
    log("X_Frame_test: (%s, %s)"  %X_Frame_test.shape)
    log("y_Frame_test: %s"        %y_Frame_test.shape)
    log("idx_Frame_test: %s"      %idx_Frame_test.shape)

    data={'X_Frame_train':X_Frame_train, 'y_Frame_train':y_Frame_train, 'idx_Frame_train':idx_Frame_train,
                'X_Frame_test':X_Frame_test, 'y_Frame_test':y_Frame_test, 'idx_Frame_test':idx_Frame_test,
                'y_Utter_train':y_Utter_train, 'y_Utter_test':y_Utter_test}

    filename  = "%s/CV_0%s/Frame_CV_0%s.pickle"  %(path_save,idx,idx)
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)




def main(ROOT_PATH,MODIFIED_DATASETS_PATH, MODEL_NAME, nFolders, segmentNum):
    MODEL_PATH = os.path.join(ROOT_PATH, "datasets/IEMOCAP" ,MODEL_NAME)
    makedirs(MODEL_PATH)

    logFolderName='exp/log/%s'%MODEL_NAME
    logFileName='%s/Extract_Segment_Level_Feats.log'%logFolderName
    log_path = os.path.join(ROOT_PATH, logFolderName)
    makedirs(log_path)
    log_path = os.path.join(ROOT_PATH, logFileName)
    infolog.init(log_path, ROOT_PATH)

    log ("Extracting segment-level features......................")
    for idx, subdir in enumerate(os.listdir(MODIFIED_DATASETS_PATH)):
        subdir=os.path.join(MODIFIED_DATASETS_PATH, subdir)
        extract_segment_level_features(subdir, MODEL_PATH, idx, segmentNum)

    log ("Saving datasets for Cross Validation...................")
    for idx in range(nFolders):
        save_cross_validation_5folder(idx, nFolders, MODEL_PATH)  
        # save_cross_validation_folder(idx, nFolders, MODEL_PATH)            

if __name__ == '__main__':
    main()
