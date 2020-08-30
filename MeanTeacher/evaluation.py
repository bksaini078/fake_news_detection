from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score as f1
from sklearn.metrics import accuracy_score
import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt

#Function to create confusion matrix 
def Confusion_matrix(model,x_test,y_true, threshold, caption='Confusion matrix'):
    '''this function will create confusion matrix with predicted value and true label'''
    y_hat= model.predict(x_test)
    y_pred=(np.greater_equal(y_hat,threshold)).astype(int)
    cm=confusion_matrix(y_true,y_pred)
    # print(cm)
    # calculating recall , precision and f1 score 
    tp_and_fp=np.sum(cm[:,1])
    tn_and_fp=np.sum(cm[0,:])
    tp_and_fn = np.sum(cm[1, : ])
    tp_and_tn= np.trace(cm)
    tp=(tp_and_fp-tn_and_fp+tp_and_tn)/2
    '''handling with divide by zero is pending'''
    #TODO: handling of divide by zero 
    precision=tp/tp_and_fp 
    recall = tp/tp_and_fn
    accuracy= np.trace(cm)/np.sum(cm)
    # f1_score=sklearn.metrics.f1_score(y_true, y_pred)
    f1_score= (2*precision*recall)/(precision+recall)
    print('Precision:', precision)
    print('Recall:', recall)
    print('f1 Score:', f1_score)
    print('Accuracy:', accuracy)

    # import matplotlib.pyplot as plt
    # figure = plt.figure(figsize=(8, 8))
    # # cm=np.around(cm.astype(int))
    # # con_mat_norm = np.around(cm, decimals=4)
    # con_mat_norm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    # sns.heatmap(con_mat_norm, annot=True,cmap=plt.cm.Oranges)
    # plt.tight_layout()
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    # plt.title(caption)
    
    # plt.show()
    return cm, accuracy, precision, recall, f1_score
def prec_rec_f1score(y_true,x_test,model):

    bce = tf.keras.losses.BinaryCrossentropy()
    y_hat= model.predict(x_test)
    y_pred=(np.greater_equal(y_hat,0.51)).astype(int)
    pr_re_f1score_perclass= precision_recall_fscore_support(y_true, y_pred, average=None)
    pr_re_f1score_average=precision_recall_fscore_support(y_true, y_pred, average='micro')
    precision=precision_score(y_true,y_pred,average=None)
    recall = recall_score(y_true,y_pred,average=None)
    accuracy= accuracy_score(y_true,y_pred)
    f1_score=f1(y_true,y_pred)
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
        print(item +':' ,metrices_value[i])
        i+=1
    binary_loss= bce(y_true, y_hat).numpy()
    print('Binary_loss',binary_loss)

    return accuracy, precision_true, precision_fake, recall_true, recall_fake, f1score_true, f1score_fake,binary_loss 

def scatter_plot(logits, y_t, title):
    marker_size=20
    figure = plt.figure(figsize=(20, 6))
    plt.scatter(logits,logits, marker_size, c=y_t)
    plt.title(title)
    plt.xlabel("Predicted Probability")
    plt.ylabel("Predicted Probability")
    cbar= plt.colorbar()
    cbar.set_label("Probability", labelpad=+1)
    plt.show()
    return

# accuracy 
def model_evaluation(model, x_test, y_true, name):
    y_hat= model(x_test)
    y_pred=(np.greater(y_hat,0.505)).astype(int)
    cm=confusion_matrix(y_true,y_pred)
    
    accuracy= np.trace(cm)/np.sum(cm)
    
    print(name+ ":")
  
    print(' Accuracy:', accuracy)

    # this will plot the result 
    # scatter_plot(y_hat,y_true, title=name)

    return accuracy #precision, recall, f1_score, accuracy