
from tokenization import tokenization
import matplotlib.pyplot as plt
from bilstm import BiLstmModel
import tensorflow as tf
from evaluation import Confusion_matrix,prec_rec_f1score
from report_writing import report_writing


def train_supervised(args,epochs, batch_size, lr, x_train, y_train, x_test, y_test,maxlen,vocab_size):

    model_supervised = BiLstmModel(maxlen, vocab_size)
    model_supervised.compile(optimizer= tf.keras.optimizers.Adam(learning_rate= lr ),loss= 'binary_crossentropy', metrics=['accuracy'])
    print('Training supervised Model...')
    history=model_supervised.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,validation_split=0.25)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()  

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    # evaluation
    train_accuracy=history.history['accuracy'][len(history.epoch)-1]
       
    test_accuracy,precision_true,precision_fake,recall_true,recall_fake,f1score_true,f1score_fake,binary_loss = prec_rec_f1score(y_test,x_test,model_supervised)
    cm, test_accuracy, precision, recall, f1_score =Confusion_matrix(model_supervised,x_test,y_test,0.51, 'Supervised model')
    report_writing(args,'Supervised_BILSTM',lr, batch_size,len(history.epoch),'NaN','NaN', train_accuracy, 
                   test_accuracy, precision_true, precision_fake, recall_true, recall_fake, 
                   f1score_true, f1score_fake,binary_loss,'Baseline')    
