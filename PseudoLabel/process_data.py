import numpy as np
from tokenization import tokenization,unison_shuffled_copies,word_dropout


  

#Preprocesses the data which calls tokenization and word dropout function in it
#arguments x_train,y_train,x_test,y_test,x_unlabelled datsets
#returns preprocessed x_train,y_train,x_test,y_test,x_unlabelled 
def process_data(args,x_train,y_train,x_test,y_test,x_unlabelled):
  #applying tokenization
  x_train, x_test , x_unlabelled, vocab_size, tokenizer = tokenization(x_train, x_test, x_unlabelled,args.max_len)
  y_train = y_train.flatten()
  y_test = y_test.flatten()
  #Taking true and false data in same ration for train(200) and test(500) and unlablled(2000)
  x_train_labelled = np.concatenate((x_train[y_train==1][0:100], x_train[y_train==0][0:100])) 
  y_train_labelled = np.concatenate((y_train[y_train==1][0:100], y_train[y_train==0][0:100]))
  x_train_labelled,y_train_labelled = unison_shuffled_copies(x_train_labelled,y_train_labelled)
  x_test_labelled = np.concatenate((x_test[y_test==1][0:500], x_test[y_test==0][0:500]))
  y_test_labelled = np.concatenate((y_test[y_test==1][0:500], y_test[y_test==0][0:500]))
  x_test_labelled,y_test_labelled = unison_shuffled_copies(x_test_labelled,y_test_labelled)
  x_unlabelled = x_unlabelled[0:2000]
  #Adding word dropout for 5 words
  x_train_labelled = np.apply_along_axis(word_dropout, 1, x_train_labelled,5)
  return x_train_labelled, y_train_labelled, x_test_labelled,y_test_labelled, x_unlabelled,vocab_size

def add_params_process_data(parser):
    parser.add_argument('--max_len',type=int, default=128)

    return parser