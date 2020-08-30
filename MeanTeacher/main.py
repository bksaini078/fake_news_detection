import argparse
from load_data import loading_data,Kfold_crossvalidation, add_params_data
from tokenization import tokenization
from MeanTeacher_syn_unlabel_diff import train_MeanTeacher_syn_unlabel_diff
from MeanTeacher_syn_noise_diff import train_MeanTeacher_syn_noise_diff
from MeanTeacher_dropout import train_MeanTeacher_dropout
from MeanTeacher_syn_dropout import train_MeanTeacher_syn_dropout
from report_writing import add_params_report
from noise_creater import add_params_noise
import numpy as np
import tensorflow as tf


if __name__ == '__main__':

    # parameters from arugument parser 
    parser = argparse.ArgumentParser()
    
    # k fold function calling 
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--meanteacher', default='MT_syn_unlabel_diff', type=str)
    #for mean teacher 
    parser.add_argument('--ratio', default=0.5, type=int)
    parser.add_argument('--alpha', type=int, default=0.99)
    parser.add_argument('--maxlen', type=int, default=100)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--synonym_noise_b1', type=float, default=0.5)
    parser.add_argument('--synonym_noise_b2', type=float, default=0.3)
    #parser.add_argument('--threashold', type=int, default=0.51)

    # Adding parameters from other python files
  #  parser = add_parms_supervised(parser)
    parser = add_params_data(parser)
    parser = add_params_report(parser)
    parser = add_params_noise(parser)
  #  parser = add_params_mt(parser)
    args = parser.parse_args()
 

    # lr=0.0001
    # epochs=30
    # batch_size= 64
    # #for mean teacher 
    # ratio =0.5
    # alpha=0.99 #(0.90-0.99)
    # maxlen=100
    x_train, y_train, x_test, y_test, x_unlabel = loading_data(args)
    x_train, x_test, x_unlabel, vocab_size, tokenizer = tokenization(args,x_train,x_test, x_unlabel, args.maxlen)
 
    for i in range(0,1):

        x_train, y_train, x_test, y_test = Kfold_crossvalidation(args,x_train,y_train,x_test,y_test)

        print("train Data_Size:",  np.shape(x_train))
        print("test Data_Size:",  np.shape(x_test))
        print('Train Label count: True, Fake', np.count_nonzero(y_train==1),np.count_nonzero(y_train==0))
        print('Test Label count : True, Fake', np.count_nonzero(y_test==1),np.count_nonzero(y_test==0))
        # train_supervised(epochs, batch_size, lr,x_train, y_train, x_test, y_test,maxlen,vocab_size)
        if (args.meanteacher == 'MT_syn_unlabel_diff'):
            train_MeanTeacher_syn_unlabel_diff(args,args.epochs, args.batch_size, args.alpha, args.lr, args.ratio,x_train, y_train, x_test, y_test, x_unlabel,vocab_size, tokenizer,args.maxlen)
        elif (args.meanteacher == 'MT_syn_noise_diff'):
            train_MeanTeacher_syn_noise_diff(args,args.epochs, args.batch_size, args.alpha, args.lr, args.ratio,x_train, y_train, x_test, y_test, x_unlabel,vocab_size, tokenizer,args.maxlen)
        elif (args.meanteacher == 'MT_dropout'):
            train_MeanTeacher_dropout(args,args.epochs, args.batch_size, args.alpha, args.lr, args.ratio,x_train, y_train, x_test, y_test, x_unlabel,vocab_size, tokenizer,args.maxlen)
        elif (args.meanteacher == 'MT_syn_dropout'):
            train_MeanTeacher_syn_dropout(args,args.epochs, args.batch_size, args.alpha, args.lr, args.ratio,x_train, y_train, x_test, y_test, x_unlabel,vocab_size, tokenizer,args.maxlen)
        else :
            print("No Mean teacher for given argument")
        # get_ipython().magic('reset -sf')
        # resetting the environment
        tf.keras.backend.clear_session()
    print('finished')
