import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.losses import binary_crossentropy,kullback_leibler_divergence
import tensorflow.keras.backend as kb
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, concatenate, LSTM, Embedding, Bidirectional, Dropout, GaussianNoise
from tensorflow.keras.models import Model
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
import datetime
from tensorflow.keras.losses import binary_crossentropy,kullback_leibler_divergence, categorical_crossentropy
tf.keras.losses.BinaryCrossentropy()
tf.keras.losses.CategoricalCrossentropy()
from pathlib import Path

#model architecture
#retuns object of model function
def model(args,vocab_size):
  print('Build model...')
  Embedding_input = Input(shape=(args.max_len, ))
  Embedding_out = Embedding(vocab_size, 128, input_length=None)(Embedding_input) 
  blstm = Bidirectional(LSTM(128))(Embedding_out)   
  noise = Dense(2)(blstm)
  output = Dense(2,activation='softmax')(noise)
  blstmmodel = Model(Embedding_input, output)
  return blstmmodel

#evaluation metrices
#arguments label, attributes and trained model
#returns accuracy, binary loss and per class pericsion, recall and F1
def prec_rec_f1score(y_true,x_test,model):
    bce = tf.keras.losses.BinaryCrossentropy()
    cce = tf.keras.losses.CategoricalCrossentropy()
    y_hat= model.predict(x_test)
    y_pred=tf.argmax(y_hat, 1).numpy().astype(int).flatten()
    pr_re_f1score_perclass= precision_recall_fscore_support(y_true, y_pred, average=None)
    pr_re_f1score_average=precision_recall_fscore_support(y_true, y_pred, average='micro')
    accuracy= accuracy_score(y_true,y_pred)
    #per class
    precision_true=pr_re_f1score_perclass[0][1]
    precision_fake=pr_re_f1score_perclass[0][0]
    recall_true=pr_re_f1score_perclass[1][1]
    recall_fake=pr_re_f1score_perclass[1][0]
    f1score_true= pr_re_f1score_perclass[2][1]
    f1score_fake= pr_re_f1score_perclass[2][0]
    metrices_name=['accuracy','precision_true','precision_fake','recall_true','recall_fake','f1score_true','f1score_fake']
    metrices_value=[accuracy, precision_true, precision_fake, recall_true, recall_fake, f1score_true, f1score_fake]
    i=0
    for item in metrices_name:
        print(item +':',metrices_value[i])
        i+=1
    #adding one hot encoding for categorical cross entropy
    y_true = tf.one_hot(y_true, depth=2)
    binary_loss= cce(y_true, y_hat).numpy()
    print('Binary_loss',binary_loss)
    return accuracy, precision_true, precision_fake, recall_true, recall_fake, f1score_true, f1score_fake,binary_loss

#function to plot roc
def plot_roc(args,fpr,tpr,label):
  from matplotlib import pyplot
  fig = pyplot.figure()
  #  plot the roc curve for the model
  pyplot.plot(fpr, tpr, linestyle='--', label=label)
  pyplot.xlabel('False Positive Rate')
  pyplot.ylabel('True Positive Rate')
  # show the legend
  pyplot.legend()
  # show the plot 
  #pyplot.show()
  fig.savefig(args.outputPath + args.rocfn)
  return

#function to writing report to drive
def report_writing( args,Model,lr,Batch_Size, Epoch, train_accuracy,test_accuracy,precision_true,precision_fake,recall_true,recall_fake,f1score_true,f1score_fake,Classification_Loss,comment=''):
    x = datetime.datetime.now()
    report_df = pd.DataFrame(columns=['Date', 'Model','Batch_Size', 'Epoch','Train_Accuracy',
                                      'Test_Accuracy', 'Precision_True','Precision_Fake','Recall_True','Recall_Fake','F1_Score_True','F1_Score_Fake','Classification_Loss',
                                      'comment'])
    report_df = report_df.append({'Date' : x.strftime("%c"), 'Model' :Model,'Batch_Size' : Batch_Size, 'Epoch': Epoch,'Train_Accuracy': train_accuracy,
                                  'Test_Accuracy': test_accuracy, 'Precision_True': precision_true,'Precision_Fake': precision_fake,'Recall_True': recall_true,'Recall_Fake': recall_fake,'F1_Score_True': f1score_true,'F1_Score_Fake': f1score_fake,'Classification_Loss':Classification_Loss,'comment': comment}, ignore_index=True)
    my_file = Path(args.outputPath + args.bitrain)

    if my_file.exists():
        report_df.to_csv(args.outputPath + args.bitrain ,mode='a', header= False , index = False)
    else:
        report_df.to_csv(args.outputPath + args.bitrain ,mode='w', header= True , index= False) 
    return 
# trainig and evaluating supervised model
def predict_pseudo(args,x_train,y_train,x_test, y_test ,vocab_size,batch_size, lr, epochs, comment=''):
  #One Hot for categorical cross entropy 
  y_train = tf.one_hot(y_train, depth=2)
  y_test = tf.one_hot(y_test, depth=2)
  lr = lr
  optimizerS= tf.keras.optimizers.Adam(learning_rate= lr)
  lp = model(args,vocab_size)
  lp.compile(optimizer=optimizerS, loss =categorical_crossentropy, metrics=['accuracy'])
  lp.fit(x_train,y_train, batch_size = batch_size, epochs = epochs)
  print('Train Eval \n')
  tr_result = lp.evaluate(x_train, y_train, batch_size = batch_size)
  print('Test Eval \n')
  results = lp.evaluate(x_test, y_test , batch_size = batch_size)
  print(results)
  #removing one hot encoding
  y_test = tf.argmax(y_test, 1).numpy().flatten()
  accuracy, precision_true, precision_fake, recall_true, recall_fake, f1score_true, f1score_fake,binary_loss = prec_rec_f1score(y_test,x_test, lp)
  report_writing(args,'Supervised_LP', lr, batch_size,epochs, tr_result[1], accuracy, precision_true, precision_fake, recall_true, recall_fake, f1score_true, f1score_fake,binary_loss,comment)
  df = pd.DataFrame(columns=["Accuracy", "precision_true", "precision_fake", "recall_true", "recall_fake", "f1score_true", "f1score_fake","binary_loss"])
  df.loc[0] = [accuracy, precision_true, precision_fake, recall_true, recall_fake, f1score_true, f1score_fake,binary_loss]
  return df

def add_params_prdict_pseduo(parser):
    parser.add_argument('--rocfn', type=str,default="Bi_Train_PL_ROC.png")
    parser.add_argument('--bitrain', type=str,default="Bi_Train_PL_27AUG.csv")


    return parser

