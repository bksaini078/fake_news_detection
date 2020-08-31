import numpy as np
import tensorflow as tf
import pandas as pd
from predict_pseudo import model,prec_rec_f1score,plot_roc,report_writing,predict_pseudo,add_params_prdict_pseduo
import os
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import datetime
from tensorflow.keras.losses import binary_crossentropy,kullback_leibler_divergence, categorical_crossentropy
tf.keras.losses.BinaryCrossentropy()
tf.keras.losses.CategoricalCrossentropy()
#Semi-supervised model
#arguments: x_train, y_train,x_test ,y_test, x_unlabel,batch_size, lr, epochs, comment, T = confidence Threshold to select pseudo labels
#returns df with result on validation set i.e. accuracy, precision_true, precision_fake, recall_true, recall_fake, f1score_true, f1score_fake,binary_loss
def pseudo_main(args,x_train, y_train,x_test ,y_test, x_unlabel,vocab_size,batch_size, lr, epochs, Thresh, comment=''):
  
  x_train, y_train,x_test ,y_test, x_unlabel = x_train, y_train,x_test ,y_test, x_unlabel
  #One Hot for categorical cross entropy 
  y_train = tf.one_hot(y_train, depth=2)
  y_test = tf.one_hot(y_test, depth=2)
  
  m1 = model(args,vocab_size)
  m2 = model(args,vocab_size)
  lr = lr
  optimizerS= tf.keras.optimizers.Adam(learning_rate= lr)
  epochs=epochs
  T = Thresh

  for epoc in range(epochs):
    print("\n *************",epoc,"************ \n")
    batch_size = batch_size
    totalSamps = x_train.shape[0]
    batches = totalSamps // batch_size
    if totalSamps % batch_size > 0:
      batches+=1
    unlabel_batch_size = x_unlabel.shape[0] //  batches
    if (unlabel_batch_size > 0) and (epochs-epoc < 2):
      if x_unlabel.shape[0] % unlabel_batch_size > 0:
        E = x_unlabel.shape[0] % unlabel_batch_size
        x_unlabel = x_unlabel[:-E]
    new_unlabel = np.empty([0,args.max_len])
    print('Len of Unlabel', len(x_unlabel))
    print('Len of X_train ', len(x_train))
    print('Len of Y_train ', len(y_train))
    
    for batch in range(batches):
      #Training predictors using labelled data
      with tf.GradientTape() as tape:
        with tf.GradientTape() as tape2:
          section = slice(batch*batch_size,(batch+1)*batch_size)
          unlabel_selection = slice(batch*unlabel_batch_size,(batch+1)*unlabel_batch_size)
          label_pred_m1 = m1(tf.convert_to_tensor(x_train[section]), training=True)
          label_loss_m1 = categorical_crossentropy(y_train[section],label_pred_m1)
          label_pred_m2 = m2(tf.convert_to_tensor(x_train[section]), training=True)
          label_loss_m2 = categorical_crossentropy(y_train[section],label_pred_m2)
          grads_m1= tape2.gradient(label_loss_m1, m1.trainable_weights)
        grads_m2= tape.gradient(label_loss_m2, m2.trainable_weights)
      optimizerS.apply_gradients(zip(grads_m1, m1.trainable_weights))
      optimizerS.apply_gradients(zip(grads_m2, m2.trainable_weights))

      #predicting and selecting Pseudo Labels and adding them to training set and removing them from unlabelled set
      if (len(x_unlabel) > 0) :  
        m1_prob = m1(tf.convert_to_tensor(x_unlabel[unlabel_selection]))
        m1_pseudo = tf.argmax(m1_prob, 1).numpy().astype(int).flatten()
        m2_prob = m2(tf.convert_to_tensor(x_unlabel[unlabel_selection]))
        m2_pseudo = tf.argmax(m2_prob, 1).numpy().astype(int).flatten()
        m1_prob_selection = m1_prob[np.array(m1_pseudo)==np.array(m2_pseudo)]
        m2_prob_selection = m2_prob[np.array(m1_pseudo)==np.array(m2_pseudo)]
        lab_selection = m1_pseudo[np.array(m1_pseudo)==np.array(m2_pseudo)]
        x_unlabel_data_selection = x_unlabel[unlabel_selection][np.array(m1_pseudo)==np.array(m2_pseudo)]
        selection = []
        for i in range(len(lab_selection)):
          selection.append(m1_prob_selection[i][lab_selection[i]].numpy()>= T and m2_prob_selection[i][lab_selection[i]].numpy()>= T)  #confidence score exceeding threshold 
        x_train = np.concatenate((x_train,x_unlabel_data_selection[selection]))
        y_train = np.concatenate((y_train,tf.one_hot(lab_selection[selection], depth=2)))
        new_unlabel = np.concatenate((new_unlabel,x_unlabel[unlabel_selection][m1_pseudo != m2_pseudo]))
        new_unlabel = np.concatenate((new_unlabel, x_unlabel_data_selection[np.where(np.array(selection) == False)]))
    if len(new_unlabel)>0:
      x_unlabel = new_unlabel
    

  #Compiling and Evaluating on both predictors and selecting one with best performance
  #m1.compile(optimizer=optimizerS, loss = categorical_crossentropy, metrics=['accuracy'])
  #print('M1 Train')
  #tr_result = m1.evaluate(x_train, y_train, batch_size = batch_size)
  #print(tr_result)
  #print('M1 Test Evaluation \n')
  #M1_result = m1.evaluate(x_test, y_test, batch_size = batch_size)
  #print(M1_result)

  #m2.compile(optimizer=optimizerS, loss = categorical_crossentropy, metrics=['accuracy'])
  #print('M2 Train')
  #tr_result = m2.evaluate(x_train, y_train, batch_size = batch_size)
  #print('M2 Test Evaluation \n')
  #M2_result = m2.evaluate(x_test, y_test, batch_size = batch_size)
  #print(M2_result)

  y_test = tf.argmax(y_test, 1).numpy()
  m1_pred = m1.predict(x_test)
  m1_pred = tf.argmax(m1_pred, 1).numpy()
  m1_acc = accuracy_score(y_test, m1_pred)
  print('M1 ACC :',m1_acc)
  m2_pred = m2.predict(x_test)
  m2_pred = tf.argmax(m2_pred, 1).numpy()
  m2_acc = accuracy_score(y_test, m2_pred)
  print('M2 ACC :',m2_acc)
  df = pd.DataFrame(columns=["Model","Accuracy", "precision_true", "precision_fake", "recall_true", "recall_fake", "f1score_true", "f1score_fake","binary_loss"])
  if m1_acc > m2_acc:
    print('m1 \n')
    accuracy_m1, precision_true_m1, precision_fake_m1, recall_true_m1, recall_fake_m1, f1score_true_m1, f1score_fake_m1,binary_loss_m1 = prec_rec_f1score(y_test,x_test, m1)
    report_writing(args,'LP M1', lr, batch_size,epochs,'NA', accuracy_m1, precision_true_m1, precision_fake_m1, recall_true_m1, recall_fake_m1, f1score_true_m1, f1score_fake_m1,binary_loss_m1, comment)
    df.loc[0] = ['m1',accuracy_m1, precision_true_m1, precision_fake_m1, recall_true_m1, recall_fake_m1, f1score_true_m1, f1score_fake_m1,binary_loss_m1]
  else:
    print('m2 \n')
    accuracy_m2, precision_true_m2, precision_fake_m2, recall_true_m2, recall_fake_m2, f1score_true_m2, f1score_fake_m2,binary_loss_m2 = prec_rec_f1score(y_test,x_test, m2)
    report_writing(args,'LP M2', lr, batch_size,epochs, 'NA' , accuracy_m2, precision_true_m2, precision_fake_m2, recall_true_m2, recall_fake_m2, f1score_true_m2, f1score_fake_m2,binary_loss_m2, comment)
    df.loc[0] = ['m2',accuracy_m2, precision_true_m2, precision_fake_m2, recall_true_m2, recall_fake_m2, f1score_true_m2, f1score_fake_m2,binary_loss_m2]
  
  return df,m1,m2
