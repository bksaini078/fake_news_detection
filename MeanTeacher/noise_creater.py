import numpy as np 
from tensorflow.keras.preprocessing import sequence
import nltk
from nltk import WordPunctTokenizer
from gensim.models.fasttext import FastText
import tensorflow as tf 
import os

def instant_noise(x_train, y_train, x_unlabel, n_ratio ):
    '''this function introduce noise in the training data for mean teacher model , 
    this function is used in calculating classification cost, user have to provide 
    amount of noise, want to add(ratio) in train data and test train split ratio too'''
    #amount of noise need to add in x_train data 
    noise=int(np.shape(x_train)[0]*n_ratio)
 
    # taking column of x_train, need it later 
    x_column = np.shape(x_train)[1]

    if noise <= int(np.shape(x_unlabel)[0]):

        #taking number of noise from unlabel data 
        ratio_noise = x_unlabel[:noise]

        # creating -1 label for noise data 
        y_unlabel=np.full((np.shape(ratio_noise)[0], 1), -1)

        # adding noise in train data 
        x = np.append(x_train, ratio_noise, axis=0)
        # print(np.shape(x))
        y = np.append(y_train, y_unlabel, axis=0)
        x = np.append(x,y, axis=1)
        row = np.shape(x)[0]

        # shufflin data 
        x =np.random.permutation(x)
        # print(np.shape(x))

        #seperating label from x 
        y_train_n=np.reshape(x[:,x_column],(row,1))
        x_train_n=x[0:len(x),0:x_column]
        # y_train_n= np.reshape(y[:len(x),0],(train_split,1))

        
    else :
        print('error: Insufficient unlabel data available !')

    return x_train_n, y_train_n




def embedding_creation(args,full_article):
    word_punctuation_tokenizer = nltk.WordPunctTokenizer()
    word_tokenized_corpus = [word_punctuation_tokenizer.tokenize(sent) for sent in full_article]

    embedd_model = FastText(word_tokenized_corpus,
                      size=args.embedding_size,
                      window=args.window_size,
                      min_count=args.min_word,
                      sample=args.down_sampling,
                      sg=0,
                      iter=50)
    print('Finished and saving model at location', args.embeddingPath)
    embedd_model.save(args.embeddingPath+ args.embedding_Model)
    return 


def synonym_noise(args,x_batch,maxlen,tokenizer):
    articles = tokenizer.sequences_to_texts(x_batch)
    changed_articles=[]
    model_embedd= FastText.load(args.embeddingPath+ 'embedding.model')
    for article in articles:
        word_array= article.split(' ')
        sent1=[]
        '''toss and taking random decision on data'''
        if np.random.binomial(1, args.synonym_noise_b1):
            for word in word_array:
                if word in model_embedd.wv.vocab:
                    most_similar=model_embedd.wv.most_similar(word)
                    # print(most_similar[0][0])
                    #flipping coin to decide to change or not if head change word and if tails dont change
                    #change p value for reducing or increasing the edit 
                    if np.random.binomial(1, args.synonym_noise_b2):
                        sent1.append(most_similar[0][0])
                    else:
                        sent1.append(word)
                else:
                    sent1.append(word)
            joined_text = ' '.join(sent1)
        else:
            joined_text=' '.join(word_array)
        changed_articles.append(joined_text)
    x_train_seq_n = tokenizer.texts_to_sequences(changed_articles)
    x_train_seq_n = sequence.pad_sequences(x_train_seq_n,maxlen=maxlen)
    x_train_seq_n=tf.convert_to_tensor(x_train_seq_n)
    return x_train_seq_n
    
def drop_out(x_batch,probability):
    # print(type(x_batch))
    for i in range(len(x_batch)):
        for j in range(len(x_batch[i])):
            if np.random.binomial(1, probability):
                x_batch[i][j]=0
            else:
                continue
    x_batch_1=tf.convert_to_tensor(x_batch)
    return x_batch_1

def add_params_noise(parser):
    parser.add_argument('--embeddingPath',type=str, default=os.path.abspath(os.getcwd())+"\\Data\\Embedding\\")
    parser.add_argument('--embedding_Model', type=str,default="embedding_Model.model")
    parser.add_argument('--embedding_size', type=int ,default=60)
    parser.add_argument('--window_size', type=int ,default=40)
    parser.add_argument('--min_word', type=int ,default=5)
    parser.add_argument('--down_sampling', type=float ,default=1e-2)


    return parser
