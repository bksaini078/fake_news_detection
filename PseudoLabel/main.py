from tensorflow.python.keras.preprocessing.text import Tokenizer
import pandas as pd
from tensorflow.keras.losses import binary_crossentropy,kullback_leibler_divergence, categorical_crossentropy
import tensorflow.keras.backend as kb
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, concatenate, LSTM, Embedding, Bidirectional, Dropout, GaussianNoise
from tensorflow.keras.models import Model
import numpy as np
import tensorflow.keras
from pathlib import Path
from tensorflow.keras.preprocessing import sequence
import datetime
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_curve
from matplotlib import pyplot
import argparse
from process_data import process_data,add_params_process_data
from predict_pseudo import model,prec_rec_f1score,plot_roc,report_writing,predict_pseudo,add_params_prdict_pseduo
import os
from pseudo_main import pseudo_main

def main(args):


  #loading preprocessed dataset which is already lemmatized
  x_train = np.load(args.inputPath + args.xTrain,allow_pickle=True)
  x_test = np.load(args.inputPath + args.xTest,allow_pickle=True)
  y_train = np.load(args.inputPath + args.yTrain,allow_pickle=True)
  y_test = np.load(args.inputPath + args.yTest,allow_pickle=True)
  x_unlabelled = np.load(args.inputPath + args.xUnlabel,allow_pickle=True)
  #preprocessing data which includes tokenization and word dropout.
  x_train_labelled, y_train_labelled, x_test_labelled, y_test_labelled, x_unlabelled,vocab_size = process_data(args,x_train,y_train,x_test,y_test,x_unlabelled)

  #df to keep record over iteration
  supervised_df = pd.DataFrame(columns=["Training Acc", "Accuracy", "precision_true", "precision_fake", "recall_true", "recall_fake", "f1score_true", "f1score_fake","binary_loss"])
  semisupervised_df = pd.DataFrame(columns=["Model","Training Acc", "Accuracy", "precision_true", "precision_fake", "recall_true", "recall_fake", "f1score_true", "f1score_fake","binary_loss"])

  

  for ep in range(args.n):

    #running supervised model and saving results in supervised_df
    df = predict_pseudo(args,x_train_labelled,y_train_labelled,x_test_labelled,y_test_labelled,vocab_size, args.batch_size, args.lr, args.Epochs, 'Supervised-BDATA 0.90 confident 5WordDrop and 5 word drop to pseudo')
    supervised_df = supervised_df.append(df)
    #running pseudo label semisupervised model and saving result in semisupervised_df
    df,m1,m2 = pseudo_main(args,x_train_labelled,y_train_labelled,x_test_labelled, y_test_labelled,x_unlabelled, vocab_size,args.batch_size,args.lr,args.Epochs,args.Thresh, 'tri learning-BDATA 0.90 confident 5WordDrop')
    semisupervised_df = semisupervised_df.append(df)
  #taking and saving mean and std of matrcies in report
  supervised_mean = supervised_df.mean()
  semisupervised_mean = semisupervised_df.mean()
  print('Mean value of Supervised model for ',args.n,' iteration')
  print(supervised_mean)
  print('Mean value of Pesudo Label model for ',args.n,' iteration')
  print(semisupervised_mean)
  report_writing(args,'Supervised Mean', args.lr, args.batch_size,args.Epochs, supervised_mean[0], supervised_mean[1], supervised_mean[2], supervised_mean[3], supervised_mean[4], supervised_mean[5], supervised_mean[6], supervised_mean[7],supervised_mean[8], 'Mean for supervised')
  report_writing(args,'LP Mean', args.lr, args.batch_size,args.Epochs, semisupervised_mean[0], semisupervised_mean[1], semisupervised_mean[2], semisupervised_mean[3], semisupervised_mean[4], semisupervised_mean[5], semisupervised_mean[6], semisupervised_mean[7],semisupervised_mean[8], 'Mean for LP')
  supervised_std = supervised_df.std()
  semisupervised_std = semisupervised_df.std()
  print('SD of Supervised model for ',args.n,' iteration')
  print(supervised_std)
  print('SD of Pesudo Label model for ',args.n,' iteration')
  print(semisupervised_std)
  report_writing(args,'Supervised STD', args.lr, args.batch_size,args.Epochs, supervised_std[0], supervised_std[1], supervised_std[2], supervised_std[3], supervised_std[4], supervised_std[5], supervised_std[6], supervised_std[7],supervised_std[8], 'STD for supervised')
  report_writing(args,'LP STD', args.lr, args.batch_size,args.Epochs, semisupervised_std[0], semisupervised_std[1], semisupervised_std[2], semisupervised_std[3], semisupervised_std[4], semisupervised_std[5], semisupervised_std[6], semisupervised_std[7],semisupervised_std[8], 'STD for LP')
  mod = semisupervised_df.groupby(['Model']).count().sort_values('precision_true', ascending=False).reset_index().Model[0]
  
  if mod=='m1':
    print(m1)
    y_pred = m1.predict(x_test_labelled)
    y_pred = tf.argmax(y_pred, 1).numpy()
    fpr_, tpr_, _ = roc_curve(y_test_labelled, y_pred)
    plot_roc(args,fpr_,tpr_,'Pseudo Label')
    return m1

  if mod == 'm2':
    print(m2)
    y_pred = m2.predict(x_test_labelled)
    y_pred = tf.argmax(y_pred, 1).numpy()
    fpr_, tpr_, _ = roc_curve(y_test_labelled, y_pred)
    plot_roc(args,fpr_,tpr_,'Pseudo Label')
    return m2

if __name__ == '__main__':


        # parameters from arugument parser 
    parser = argparse.ArgumentParser()
    
    # pseduolabel 
    #User parameters lr, batch_size, Epochs, n= no of iteration need to get mean result, Thresh = Confidence Threshold between 0 and 1 to select Pseudo Labels
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--Epochs', default=14, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--n', default=10, type=int)
    parser.add_argument('--Thresh', default=0.90, type=float)
    #filename and path for input and output file
    parser.add_argument('--inputPath',type=str, default=os.path.abspath(os.getcwd())+"\\Data\\Input2\\")
    parser.add_argument('--outputPath',type=str, default=os.path.abspath(os.getcwd())+"\\Data\\Output2\\")
    parser.add_argument('--xUnlabel', type=str,default="xun_shuffled.npy")
    parser.add_argument('--xTrain', type=str,default="xtr_shuffled.npy")
    parser.add_argument('--xTest', type=str,default="xte_shuffled.npy")
    parser.add_argument('--yTrain', type=str,default="ytr_shuffled.npy")
    parser.add_argument('--yTest', type=str,default="yte_shuffled.npy")

    #parser.add_argument('--threashold', type=int, default=0.51)

    # Adding parameters from other python files
    parser = add_params_process_data(parser)
    parser =add_params_prdict_pseduo(parser)
    args = parser.parse_args()

    main(args)
