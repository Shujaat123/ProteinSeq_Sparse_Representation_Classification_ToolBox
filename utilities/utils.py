import sys, os, re, gc
import numpy as np
import pandas as pd
from random import sample

## 
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

## Models
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Dropout
from keras import metrics
from keras import optimizers
from keras.utils.np_utils import to_categorical

import numpy.linalg as LA
from sklearn.model_selection import train_test_split

## Perfmetrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score, matthews_corrcoef, balanced_accuracy_score, precision_recall_fscore_support
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve,roc_auc_score


## utilities
from matplotlib import pyplot as plt
from scipy.io import savemat
from sklearn.model_selection import KFold, StratifiedKFold
# !pip install wget
import wget


## pre-processing
from sklearn.preprocessing import normalize, Normalizer
from sklearn.decomposition import KernelPCA,PCA
from imblearn.over_sampling import ADASYN, SMOTE, SVMSMOTE, KMeansSMOTE, BorderlineSMOTE




#####################-FILE DOWNLOADER-#######################
def download_file(input_img_path, input_file):
  if (os.path.exists(input_file)):
    print('File already exist.')
    # os.remove(input_file)
    # wget.download(input_img_path, input_file)
  else:
    wget.download(input_img_path, input_file)
    print('DONE.')

#####################-DOWNLOAD SOLVER-#######################
input_img_path = 'https://raw.githubusercontent.com/NLPrinceton/sparse_recovery/master/solvers.py'
input_file = 'solvers.py'
download_file(input_img_path, input_file)

from solvers import *

########################### -- EVALUATION METRICS -- ########################

def yoden_index(y, y_pred):
  epsilon = 1e-30
  tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0,1]).ravel()
  j = (tp/(tp + fn + epsilon)) + (tn/(tn+fp + epsilon)) - 1
  return j

def pmeasure(y, y_pred):
    epsilon = 1e-30
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0,1]).ravel()
    sensitivity = tp / (tp + fn + epsilon)
    specificity = tn / (tn + fp + epsilon)
    f1score = (2 * tp) / (2 * tp + fp + fn + epsilon)
    return ({'Sensitivity': sensitivity, 'Specificity': specificity, 'F1-Score': f1score})
    
def Calculate_Stats(y_actual,y_pred):
  acc = accuracy_score(y_actual, y_pred)
  sen = pmeasure(y_actual, y_pred)['Sensitivity']
  spe = pmeasure(y_actual, y_pred)['Specificity']
  f1 = pmeasure(y_actual, y_pred)['F1-Score']
  mcc = matthews_corrcoef(y_actual, y_pred)
  bacc = balanced_accuracy_score(y_actual, y_pred)
  yi = yoden_index(y_actual, y_pred)
  
  return acc, sen, spe, f1, mcc, bacc, yi


#################################################################
############## -- SPARSE REPRESENTATION CLASSIFIER ##############
#################################################################

#### RECONSTRUCTION FUNCTION ####
def Test_SRC(A,delta_y,DATA,LABEL,solver='BP',verbose=0, x0=None, ATinvAAT=None, nnz=None, positive=False, tol=1E-4, niter=100, biter=32):
  import time
  LABEL_PRED = []
  SCORE_PRED=[]
  count = 0
  time_ellapsed = []
  for ind in range(0,DATA.shape[1]):
    start_time = time.time()
    b = DATA[:,ind]
    if(solver=='BP'):     
      x = BasisPursuit(A, b, x0=x0, ATinvAAT=ATinvAAT, positive=positive, tol=tol, niter=niter, biter=biter)
    elif(solver=='MP'):      
      x = MatchingPursuit(A, b, tol=tol, nnz=nnz, positive=positive)
    elif (solver=="OP"):
      x = OrthogonalMP(A, b, tol=tol, nnz=nnz, positive=positive)
 
    label_out, score_out = delta_rule(A,delta_y,x,b)
    time_ellapsed.append(time.time()-start_time)
    if (verbose):
      check = label_out==LABEL[ind]
      if (check):
        count = count + 1
      accuracy = 100*count/(ind+1)
      print(ind+1, count, accuracy, LABEL[ind], label_out, check)
    LABEL_PRED.append(label_out)
    SCORE_PRED.append(score_out)

  return np.array(LABEL_PRED), np.array(SCORE_PRED),np.array(time_ellapsed )

#### DECISION RULE FUNCTION ####
def delta_rule(A,delta_y,x,b):
  delta1 = 0*x
  delta2 = 0*x
  delta1[delta_y==1]=x[delta_y==1]
  delta2[delta_y==0]=x[delta_y==0]
  y1 = np.matmul(A,delta1)
  y2 = np.matmul(A,delta2)
  r1 = np.linalg.norm(y1-b)
  r2 = np.linalg.norm(y2-b)

  if(r1<r2):
    label = 1
  else:
    label = 0
  score=(r2)/(r1+r2)

  return label, score


#################################################################################
########################## FEATURE EXTRACTION FUNCTIONS #########################
#################################################################################

## CKS-AAP Features ##
def Convert_Seq2CKSAAP(train_seq, gap=8):
  cksaapfea = []
  seq_label = []
  for sseq in train_seq:
    temp= CKSAAP([sseq], gap=8)
    cksaapfea.append(temp[1][1:])
    seq_label.append(sseq[0])

  x = np.array(cksaapfea)
  y = np.array(seq_label)
  y[y=='ACP']=1
  y[y=='non-ACP']=0
  y = to_categorical(y)
  # print('num pos:', sum(y[:,0]==1), 'num neg:', sum(y[:,0]==0))
  return x,y

def minSequenceLength(fastas):
    minLen = 10000
    for i in fastas:
        if minLen > len(i[1]):
            minLen = len(i[1])
    return minLen

def CKSAAP(fastas, gap=5, **kw):
    if gap < 0:
        print('Error: the gap should be equal or greater than zero' + '\n\n')
        return 0

    if minSequenceLength(fastas) < gap+2:
        print('Error: all the sequence length should be larger than the (gap value) + 2 = ' + str(gap+2) + '\n' + 'Current sequence length ='  + str(minSequenceLength(fastas)) + '\n\n')
        return 0

    AA = 'ACDEFGHIKLMNPQRSTVWY'
    encodings = []
    aaPairs = []
    for aa1 in AA:
        for aa2 in AA:
            aaPairs.append(aa1 + aa2)
    header = ['#']
    for g in range(gap+1):
        for aa in aaPairs:
            header.append(aa + '.gap' + str(g))
    encodings.append(header)
    for i in fastas:
        name, sequence = i[0], i[1]
        code = [name]
        for g in range(gap+1):
            myDict = {}
            for pair in aaPairs:
                myDict[pair] = 0
            sum = 0
            for index1 in range(len(sequence)):
                index2 = index1 + g + 1
                if index1 < len(sequence) and index2 < len(sequence) and sequence[index1] in AA and sequence[index2] in AA:
                    myDict[sequence[index1] + sequence[index2]] = myDict[sequence[index1] + sequence[index2]] + 1
                    sum = sum + 1
            for pair in aaPairs:
                code.append(myDict[pair] / sum)
        encodings.append(code)
    return encodings


#################################################################################
############ SEQUENCE READING FUNCTION FOR ACP 740 DATASET ######################
#################################################################################
def prepare_feature_acp740():
  ############### DOWNLOAD DATASET ###############
    input_img_path = 'http://www.cczubio.top/static/ACP-check/datasets/ACP-DL/acp740.txt'
    input_file = 'acp740.txt'
    download_file(input_img_path, input_file)

  ################################
    path = r"acp740.txt"
    new_list=[]
    seq_list=[]
    label = []
    lis = []
    lx=[]
    interaction_pair = {}
    RNA_seq_dict = {}
    protein_seq_dict = {}
    protein_index = 0
    with open(path, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                values = line[1:].strip().split('|')
                label_temp = values[1]            
                proteinName = values[0]
                proteinName_1=proteinName.split("_")
                new_list.append(proteinName_1[0])   

                if label_temp == '1':
                    label.append(1)
                else:
                    label.append(0)
            else:
                seq = line[:-1]
                seq_list.append(seq)
        for i, item in enumerate(new_list):
            lis.append([item, seq_list[i]])
        for i in lis:
            if len(i[1])>60:
                x=([i[0],i[1][0:60]])
                lx.append(x)
            else:
                lx.append(i)        
    return lx 
