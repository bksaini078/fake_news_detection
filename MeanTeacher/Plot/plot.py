import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

def scatter_plot(args,epoch,logits, y_t, title):
    marker_size=20
    figure = plt.figure(figsize=(20, 6))
    plt.scatter(logits,logits, marker_size, c=y_t)
    plt.title(title)
    plt.xlabel("Predicted Probability")
    plt.ylabel("Predicted Probability")
    cbar= plt.colorbar()
    cbar.set_label("Probability", labelpad=+1)
    #plt.show()
    plt.savefig(args.outputPath +title+'_'+str(epoch)+'.png')
    return

# accuracy 
def model_evaluation(args,epoch,model, x_test, y_true, name):
    y_hat= model(x_test)
    y_pred=(np.greater(y_hat,0.505)).astype(int)
    cm=confusion_matrix(y_true,y_pred)
    
    accuracy= np.trace(cm)/np.sum(cm)
    
    print(name+ ":")
  
    print(' Accuracy:', accuracy)

    # this will plot the result 
    scatter_plot(args,epoch,y_hat,y_true, title=name)

    return accuracy #precision, recall, f1_score, accuracy