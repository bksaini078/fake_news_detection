from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
import numpy as np

#function to tokenize preprocessed dattaset
#arguments x_train, x_test, x_unlabel dataset and max_len for pad_sequence
#returns tokenized and padded dataset with vocabSize and tokenizer
def tokenization(x_train, x_test, x_unlabel, maxlen):

    tokenizer = Tokenizer(num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ',
                          char_level=False, oov_token=None, document_count=0)
    full_article = np.hstack((x_train, x_test, x_unlabel))
    tokenizer.fit_on_texts(full_article)
    x_train_token = tokenizer.texts_to_sequences(x_train)
    x_test_token = tokenizer.texts_to_sequences(x_test)
    x_unlabel_token = tokenizer.texts_to_sequences(x_unlabel)
    x_train_seq = sequence.pad_sequences(x_train_token, maxlen=maxlen,padding='post')
    x_test_seq = sequence.pad_sequences(x_test_token, maxlen=maxlen,padding='post')
    x_unlabel_tar= sequence.pad_sequences(x_unlabel_token, maxlen=maxlen,padding='post')
    # defining vocalbury size
    vocab_size = len(tokenizer.word_index) + 1

    x_train = x_train_seq
    x_test = x_test_seq
    return x_train, x_test , x_unlabel_tar, vocab_size, tokenizer

#unison Shuffling of dataset
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    #print(p)
    return a[p], b[p]

#word dropout function for introducing noise in model
#arguments **tokenized** article and no.of words to be dropped
#returns article with K tokens replced with zero
def word_dropout(article, K):
  from random import randint
  n = len(article)
  i = 0
  while i<K:
    if np.count_nonzero(article)<K:
      #print('Less')
      break
    rn = randint(0,n-1)
    if article[rn] != 0:
      article[rn] = 0
      i+= 1
  return article