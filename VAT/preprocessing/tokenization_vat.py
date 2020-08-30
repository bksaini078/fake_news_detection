from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import pickle
import argparse
import numpy as np

cache = {} #Initilizing cache

def clearcache():
    """Clear the cache entirely."""
    cache.clear()

clearcache()
parser = argparse.ArgumentParser()
parser.add_argument('--inputPath',type=str, default=os.path.abspath(os.getcwd())+"\\VAT\\Input1\\processed\\")
parser.add_argument('--outputPath',type=str, default=os.path.abspath(os.getcwd())+"\\VAT\\Input1\\temp\\")
parser.add_argument('--pickelPath',type=str, default=os.path.abspath(os.getcwd())+"\\VAT\\Input1\\meta\\")
parser.add_argument('--filename', type=str,default="lable.csv")
parser.add_argument('--xUnlabel', type=str,default="xun_shuffled.npy")
parser.add_argument('--xTrain', type=str,default="x_train.npy")
parser.add_argument('--xTest', type=str,default="x_test.npy")
parser.add_argument('--yTrain', type=str,default="y_train.npy")
parser.add_argument('--yTest', type=str,default="y_test.npy")
args = parser.parse_args()

# cwd = os.getcwd()
# print('Working Directory: ', cwd)
# path = cwd + '/Fake News Spacy/data'

x_train = np.load(args.inputPath + args.xTrain, allow_pickle=True)
x_test = np.load(args.inputPath + args.xTest , allow_pickle=True)
y_train = np.load(args.inputPath + args.yTrain, allow_pickle=True).astype('int')
y_test = np.load(args.inputPath + args.yTest,  allow_pickle=True).astype('int')
unlabelled = np.load(args.inputPath + args.xUnlabel,  allow_pickle=True)



MAX_VOCAB_SIZE = 1000000
MAX_DOC_LENGTH = 100

corpus = np.concatenate((x_train, x_test, unlabelled))
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token='UNK')
tokenizer.fit_on_texts(corpus)
word_index = tokenizer.word_index
print('Vocabulary size :', len(word_index))


def encode(data):
    data = tokenizer.texts_to_sequences(data)
    data = pad_sequences(data, padding='post', maxlen=MAX_DOC_LENGTH)
    return data

x_train = encode(x_train)
x_test = encode(x_test)
unlabelled = encode(unlabelled)

encoded_docs = np.concatenate((x_train, x_test, unlabelled))

#len_list = [len(row) for row in unlabelled]

# print('Mean length of corpus in terms of words: ', np.mean(len_list))
# print('Max length of corpus in terms of words: ', np.max(len_list))
# print('Min length of corpus in terms of words: ', np.min(len_list))
# print('Median length of corpus in terms of words: ', np.median(len_list))


pickle_out = open(args.pickelPath + '/word_index.pickle','wb')
pickle.dump(word_index, pickle_out)
pickle_out.close()

print('Word Index saved locally')


np.save(args.outputPath + args.xTrain, x_train)
np.save(args.outputPath + args.xTest, x_test)
np.save(args.outputPath + args.xUnlabel, unlabelled)
np.save(args.outputPath + args.yTrain, y_train)
np.save(args.outputPath + args.yTest, y_test)

print('Features and labels saved locally in npy format')
clearcache()