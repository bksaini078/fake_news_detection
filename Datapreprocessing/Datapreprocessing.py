# please install contractions
# import pdb
# pdb.set_trace()
import argparse
import contractions
import en_core_web_sm
import nltk
import nltk
import numpy as np
import os
import pandas as pd
import random
import re
import spacy
import string
import tensorflow as tf
import time
from gensim import utils
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.test.utils import get_tmpfile
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

nlp = en_core_web_sm.load()
nltk.download('stopwords')
nltk.download('punkt')
porter = PorterStemmer()
# import unidecode
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim import utils
from gensim.test.utils import get_tmpfile
from nltk.corpus import stopwords
import nltk
import en_core_web_sm
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
import tensorflow as tf
import matplotlib.pyplot as plt

nlp = en_core_web_sm.load()
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
porter = PorterStemmer()
lancaster = LancasterStemmer()

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
        text = text.replace(ent.text, ent.label_)
    return text


def textClean(text):
    # """
    # Get rid of the non-letter and non-number characters
    # """
    # text = re.sub(REGEX, " ", text)
    # text = re.sub(NEWLINE_REGEX, " ", text)
    # text = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", text)
    text = re.sub('[^A-Za-z0-9]+', ' ', text)
    re.sub(r'@\w+', '', text)

    # text = unidecode.unidecode(text)

    text = text.lower().split()
    stops = set(stopwords.words("english"))
    text = [t for t in text if t.isalpha()]
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(t) for t in text]
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    return (text)


def stemSentence(sentence):
    token_words = word_tokenize(sentence)
    token_words
    stem_sentence = []
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)


def cleanup(text):
    text = replace_contractions(text)
    text = textClean(text)
    # text = text.translate(str.maketrans("", "", string.punctuation))
    # text = replace_ne(text,nlp)
    text = stemSentence(text)
    return text


def data_preprocessing_label(args, data):
    # removing missing rows
    missing_rows = []
    data = data.dropna()
    data = data.reset_index(drop=True)
    # data=data.Article.dropna()
    # data.Label= data.Label.astype(int)
    # data.Article= data.Article.astype(str)

    # cleaning data
    for i in range(len(data)):
        # print(data.loc[i,'Article'])
        data.loc[i, 'Article'] = cleanup(data.loc[i, 'Article'])

    train_size = int(0.75 * len(data))
    test_size = len(data) - train_size
    x_train = data.loc[:train_size, 'Article'].values
    y_train = data.loc[:train_size, 'Label'].values
    x_test = data.loc[(train_size + 1):, 'Article'].values
    y_test = data.loc[(train_size + 1):, 'Label'].values

    # converting into np arrray and hot one encoding
    y_train = tf.one_hot(y_train, 1)  # np.array(y_train)
    y_test = tf.one_hot(y_test, 1)  # np.array(y_test)

    np.save(args.outputPath + args.xTrain, x_train)
    np.save(args.outputPath + args.xTest, x_test)
    np.save(args.outputPath + args.yTrain, y_train)
    np.save(args.outputPath + args.yTest, y_test)
    return


def data_preprocessing_unlabel(args, data):
    df_unlabel = data
    for i in range(len(df_unlabel)):
        df_unlabel.loc[i, 'Article'] = cleanup(df_unlabel.loc[i, 'Article'])

    # shuffling
    df_unlabel = df_unlabel.sample(frac=1).reset_index(drop=True)

    x_unlabelled = df_unlabel['Article'].values
    np.save(args.outputPath + args.xUnlabel, x_unlabelled)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputPath', type=str,
                        default=os.path.abspath(os.getcwd()) + "\\Data\\\Preprocessing\\Input1\\")
    parser.add_argument('--outputPath', type=str, default=os.path.abspath(os.getcwd()) + "\\Data\\Input1\\")
    parser.add_argument('--labeldataname', default="label.csv", type=str)
    parser.add_argument('--unlabeldataname', default="unlabel.csv", type=str)
    parser.add_argument('--xTrain', type=str, default="xtr_shuffled.npy")
    parser.add_argument('--xTest', type=str, default="xte_shuffled.npy")
    parser.add_argument('--yTrain', type=str, default="ytr_shuffled.npy")
    parser.add_argument('--yTest', type=str, default="yte_shuffled.npy")
    parser.add_argument('--xUnlabel', type=str, default="xun_shuffled.npy")
    # path_labelled = os.path.dirname(os.path.realpath(__file__))+"\\Data\\Testing_label.csv";
    # path_unlabelled=os.path.dirname(os.path.realpath(__file__))+"\\Data\\Testing_unlabel.csv";
    args = parser.parse_args()
    data_label = pd.read_csv(args.inputPath + args.labeldataname)
    data_label = data_label.reset_index(drop=True)
    data_unlabel = pd.read_csv(args.inputPath + args.unlabeldataname)
    data_unlabel = data_unlabel.reset_index(drop=True)
    data_preprocessing_label(args, data_label)
    data_preprocessing_unlabel(args, data_unlabel)
