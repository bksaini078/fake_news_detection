
import os
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

def loading_data(args):
    if not os.path.isfile(args.inputPath + args.xTest):
        # print(args.inputPath + args.xTest)
        print("Please clean the data first or the location of npy file is incorrect, Checking function Loading_data")
        quit( )
    else:
        x_tr = np.load(args.inputPath + args.xTrain,allow_pickle=True)
        x_te = np.load(args.inputPath + args.xTest ,allow_pickle=True)
        y_tr = np.load(args.inputPath + args.yTrain,allow_pickle=True)
        y_te = np.load(args.inputPath + args.yTest,allow_pickle=True)
        x_un = np.load(args.inputPath + args.xUnlabel,allow_pickle=True)

    return x_tr, y_tr, x_te, y_te, x_un

def Kfold_crossvalidation(args,x_train,y_train,x_test,y_test):
    '''this function is for k_fold crossvalidation implementation'''
    data= np.append(x_train, x_test, axis=0)
    label= np.append(y_train,y_test, axis=0)
    column_size= np.shape(data)[1]
    # combining whole data
    whole_data=np.append(data,label.astype(int),axis=1)
    whole_data= np.random.permutation(whole_data)
    whole_label= whole_data[:][:,-1:]
    whole_data= whole_data[:][:,:column_size]
    x_tr, x_te, y_tr, y_te = train_test_split(whole_data, whole_label, test_size=0.33)

    return x_tr, y_tr, x_te, y_te

def add_params_data(parser):
    parser.add_argument('--inputPath',type=str, default=os.path.abspath(os.getcwd())+"/Data/Input2/")
    parser.add_argument('--outputPath',type=str, default=os.path.abspath(os.getcwd())+"/Data/Output2/")
    parser.add_argument('--xTrain', type=str,default="xtr_shuffled.npy")
    parser.add_argument('--xTest', type=str,default="xte_shuffled.npy")
    parser.add_argument('--yTrain', type=str,default="ytr_shuffled.npy")
    parser.add_argument('--yTest', type=str,default="yte_shuffled.npy")
    parser.add_argument('--xUnlabel', type=str,default="xun_shuffled.npy")

    return parser

