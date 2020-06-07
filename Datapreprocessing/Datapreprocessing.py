import argparse
import numpy as np
import re
import os
import nltk
import contractions
import string
import pandas as pd
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim import utils 
from gensim.test.utils import get_tmpfile
from nltk.corpus import stopwords
import nltk
import tensorflow as tf
import en_core_web_sm
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
nlp = en_core_web_sm.load()
nltk.download('stopwords')
nltk.download('punkt')
porter=PorterStemmer()

REGEX = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
             '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

NEWLINE_REGEX = ('[\n\r\t]')
def replace_contractions(text):
    """Replace contractions in string of text"""
    return contractions.fix(text)


def replace_ne(text: str, nlp) -> str:
    '''
    This function extracts the name entities, and then replace them with the ne labels.
    :param text:
    :type text:
    :return:
    :rtype:
    '''
    doc = nlp(text)
    for ent in doc.ents:
        text = text.replace(ent.text,ent.label_ )
    return text
def textClean(text):
    """
    Get rid of the non-letter and non-number characters
    """
    text = re.sub(REGEX, " ", text)
    text = re.sub(NEWLINE_REGEX, " ", text)
    text=re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", text)
    text = text.lower().split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    return (text)

def stemSentence(sentence):
    token_words=word_tokenize(sentence)
    token_words
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)

def cleanup(text):
    text = textClean(text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = replace_ne(text,nlp)
    text= replace_contractions(text)

    text= stemSentence(text)
    return text

def data_preprocessing_label(data):
    #removing missing rows 
    missing_rows=[]
    data=data.dropna()
    data['Label']= data['Label'].astype(int)

    data['Article']= data['Article'].astype(str)

   #cleaning data 
    for i in range(len(data)):
        data.loc[i, 'Article'] = cleanup(data.loc[i,'Article'])

    train_size = int(0.75 * len(data))
    test_size = len(data) - train_size
    x_train= data.loc[:train_size,'Article'].values
    y_train= data.loc[:train_size,'Label'].values
    x_test= data.loc[(train_size+1):,'Article'].values
    y_test= data.loc[(train_size+1):,'Label'].values

    #converting into np arrray and hot one encoding 
    y_train = tf.one_hot(y_train,1) # np.array(y_train)
    y_test = tf.one_hot(y_test,1) #np.array(y_test)
    np.save(os.path.abspath(os.getcwd())+"\\Data\\Input\\xtr_shuffled.npy",x_train)
    np.save(os.path.abspath(os.getcwd())+"\\Data\\Input\\xte_shuffled.npy",x_test)
    np.save(os.path.abspath(os.getcwd())+"\\Data\\Input\\ytr_shuffled.npy",y_train)
    np.save(os.path.abspath(os.getcwd())+"\\Data\\Input\\yte_shuffled.npy",y_test)
    return

def data_preprocessing_unlabel(data):
    df_unlabel=data
    for i in range(len(df_unlabel)):
        df_unlabel.loc[i,'Article'] = cleanup(df_unlabel.loc[i,'Article'])
        
#shuffling 
    df_unlabel= df_unlabel.sample(frac=1).reset_index(drop=True)

    x_unlabelled= df_unlabel['Article'].values
    np.save(os.path.abspath(os.getcwd())+"\\Data\\Input\\xun_shuffled.npy",x_unlabelled)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--labeldata', default=os.path.abspath(os.getcwd())+"\\Data\\Preprocessing\\label.csv", type=str)
    parser.add_argument('--unlabeldata', default=os.path.abspath(os.getcwd())+"\\Data\\Preprocessing\\unlabel.csv", type=str)
    #path_labelled = os.path.dirname(os.path.realpath(__file__))+"\\Data\\Testing_label.csv";
    #path_unlabelled=os.path.dirname(os.path.realpath(__file__))+"\\Data\\Testing_unlabel.csv";
    args = parser.parse_args()
    data_label = pd.read_csv(args.labeldata)
    data_unlabel=pd.read_csv(args.unlabeldata)
    
    data_preprocessing_label(data_label)
    data_preprocessing_unlabel(data_unlabel)
    