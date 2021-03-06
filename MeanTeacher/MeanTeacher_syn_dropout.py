import tensorflow as tf 
import tensorflow.keras as tfk
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Lambda
from tensorflow.keras.models import Model
import numpy as np
import datetime
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras import regularizers
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
#this is to enable eager execution
tf.compat.v1.enable_eager_execution()

#calling functions

from costfunction import classification_costs,Overall_Cost,Consistency_Cost,ema
from report_writing import report_writing
from bilstm import BiLstmModel
from noise_creater import instant_noise,embedding_creation,synonym_noise,drop_out
from evaluation import Confusion_matrix,prec_rec_f1score,scatter_plot,model_evaluation

def train_MeanTeacher_syn_dropout(args,epochs, batch_size, alpha, lr, ratio,x_train, y_train, x_test, y_test, x_unlabel_tar,vocab_size, tokenizer,maxlen):
    #splitting training data 
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25)


    #preparing the training dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    # Prepare the validation dataset.
    # val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    # val_dataset = val_dataset.batch(batch_size)

    #preparing the target dataset 
    tar_dataset =  tf.data.Dataset.from_tensor_slices(x_unlabel_tar)
    tar_dataset = tar_dataset.shuffle(buffer_size=1024).batch(batch_size)

    #declaring optimiser
    optimizer= tf.keras.optimizers.Adam(learning_rate= lr ) #trying changing learning rate , sometimes it gives good result 
    train_metrics = tf.keras.metrics.BinaryAccuracy(name='Binary_Accuracy')
    val_acc_metric = tf.keras.metrics.BinaryAccuracy(name="Binary_Acc")
    teacher_acc_metric = tf.keras.metrics.BinaryAccuracy(name="Binary_Acc_teacher") 
    # Creating model
    student = BiLstmModel(maxlen, vocab_size)
    teacher = BiLstmModel(maxlen, vocab_size)


    # collecting costs
    #this one for collecting the costs
    # consistency=[]
    # overall=[]
    # classification=[]
    train_accuracy=[]
    steps=[]

    # iterator_unlabel = iter(tar_dataset)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    val_dataset = val_dataset.batch(batch_size)

    #training teacher with one epoch 
   
    #this I am doing to get all steps details in epoch
    i=0
    print('Train Mean teacher Model...')
    teacher.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),loss=tf.keras.losses.BinaryCrossentropy(),metrics=['accuracy'])
    teacher.fit(x_train,y_train, batch_size=batch_size, epochs=1)

    acc_t=0
    x_unlabel_tar= tf.convert_to_tensor(x_unlabel_tar)
    for epoch in range(1,epochs+1):  
        print(*"*****************")
        print('Start of epoch %d' % (epoch,))
        print(*"*****************")
        #iteration over batches 
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
             with tf.GradientTape() as tape:
         
                # adding instant noise
                iterator_unlabel = iter(tar_dataset)
                x_batch_unlabel = iterator_unlabel.get_next()

                
                '''this is related to change with synonyms in articles'''
                x_batch_sn= synonym_noise(args,x_batch_train.numpy(),maxlen,tokenizer)
                x_batch_dp= drop_out(x_batch_sn,args.dropout)

                '''this is one method of adding -1 label using unlable data'''
                # x_train_n,y_train_n= instant_noise(x_batch_train,y_batch_train,x_batch_unlabel,0.2)

                # Run the forward pass of the layer
                logits= student(x_batch_dp, training= True)  
                # logits_acc =  student(x_batch_sn, training= False) 

                # TODO:this  metrics also have to right 
                train_metrics(y_batch_train,logits)  

                #Calculating classification cost 
                classification_cost = classification_costs(logits,y_batch_train)
                # classification.append(classification_cost)
         


                x_batch_sn1= synonym_noise(args,x_batch_train.numpy(),maxlen,tokenizer)
                x_batch_dp1= drop_out(x_batch_sn1,args.dropout)
                
                # tar_student= student(x_batch_dp1)
                tar_teacher = teacher(x_batch_dp1) #x_batch_train
                #  tar_student= student(x_train_n)
                consistency_cost= Consistency_Cost(tar_teacher,logits) 
                # consistency.append(consistency_cost)

                overall_cost= Overall_Cost(classification_cost, consistency_cost, ratio=0.5)
                # overall.append(overall_cost)
                #  consistency_cost = consistency_cost #this is ratio 
                #adding loss to student model 
             grads= tape.gradient(overall_cost, student.trainable_weights)
             i=i+1
             steps.append(i)
   
             # the value of the variables to minimize the loss.
             optimizer.apply_gradients(zip(grads, student.trainable_weights))
             teacher= ema(student, teacher, alpha=alpha)
            #  if step % 10==0:
            #      print('alpha:', alpha)
            #      print('Training loss:- Binary Cross entropy at step %s: %s' % (step, float(classification_cost)))
            #      print("Consistency Cost: %s" % (float(consistency_cost)))
            #      print('Seen so far: %s samples' % ((step + 1) * batch_size))
            #      print("Overall Cost: %s" % (float(overall_cost)))
            #      print("--------------- step, batch %s: %s ---------------------"% (step,((step + 1) * batch_size)))
        # ramp up from 0.99 to 0.999
        # if alpha <=0.99:
        #     alpha= alpha+0.0025
        train_acc = train_metrics.result()
        print(alpha)
   
        #appending training accuracy
        train_accuracy.append(train_acc)

        # print('Training acc over epoch: %s' % (float(train_acc)*100,))
            # Reset training metrics at the end of each epoch
        train_metrics.reset_states()
   
        # Run a validation loop at the end of each epoch.
        print('*******STUDENT*************')
        prec_rec_f1score(y_val,x_val,student)
        print('*******TEACHER*************')
        prec_rec_f1score(y_val,x_val,teacher)

        # acc_s = model_evaluation(student, x_test, y_test, name= 'student')

        # acc_t = model_evaluation(teacher, x_test, y_test, name= 'teacher')
        if epoch >= 10 and epoch % 5 ==0 :
            print('---------------------------STUDENT--------------------------')
            test_accuracy,precision_true,precision_fake,recall_true,recall_fake,f1score_true,f1score_fake,binary_loss = prec_rec_f1score(y_test,x_test,student)
            report_writing(args,'Student',lr, batch_size,epoch,alpha,ratio, train_acc.numpy(), 
                           test_accuracy, precision_true, precision_fake, recall_true, recall_fake, 
                           f1score_true, f1score_fake,binary_loss,'Noise-using-dropout')  
            print('-----------------------------------------------------------------')
    
            print('---------------------------TEACHER---------------------------------')
        #    cm, test_accuracy, precision, recall, f1_score =Confusion_matrix(teacher,x_test,y_test,0.51, 'Teacher model')
            test_accuracy,precision_true,precision_fake,recall_true,recall_fake,f1score_true,f1score_fake,binary_loss = prec_rec_f1score(y_test,x_test,teacher)
            report_writing(args,'Teacher',lr, batch_size,epoch,alpha,ratio, train_acc.numpy(), 
                           test_accuracy, precision_true, precision_fake, recall_true, recall_fake, 
                           f1score_true, f1score_fake,binary_loss,'Noise-using-dropout') 
            print('-----------------------------------------------------------------')
    tf.keras.backend.clear_session()
    return


    