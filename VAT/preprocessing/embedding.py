import os
import time
import pickle
import numpy as np
import argparse

def load_word_index(path):
  word_index = open(path + '/word_index.pickle','rb')
  word_index = pickle.load(word_index)
  print('Word Index Pickle load successful')
  return word_index

def load_glove(path, EMBEDDING_DIM):
  GLOVE_6B_DIR = path + '/glove.6B.' + str(EMBEDDING_DIM) + 'd.txt'
  start = time.time()
  print('Loading Glove in RAM...')
  def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')
  embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(GLOVE_6B_DIR, encoding="utf8"))
  stop = time.time()
  print('Time Elapsed: {} s'.format(round((stop-start),1)))
  return embeddings_index

def save_unknown(not_found_str, path):
  """
  :param not_found_str: all words which was not found t=in the glove matrix;
  :return: saves them in a txt file just for reference
  """
  unknowns = open(path + '/unknown_words.txt', 'w')
  unknowns.write(not_found_str)
  unknowns.close()

def create_emb_matrix(args,word_index, embeddings_index, path):
  """
  :param word_index: word index
  :param embeddings_index: glove embedding index for each word
  :return: embedding matrix; all words not found are give zero embedding such that they always represent UNKNOWN
  """
  print('Preparing embedding matrix for the Word Index...')
  start = time.time()
  vocab_size = len(word_index)
  embedding_matrix = np.zeros((vocab_size + 1, args.EMBEDDING_DIM))
  count = 0
  not_found_str = ''
  for word, index in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
      embedding_matrix[index] = embedding_vector
    else:
      not_found_str = not_found_str + word + ' '
      count += 1
  print('Words not found in glove:', count)
  save_unknown(not_found_str, path)
  stop = time.time()
  print('Time Elapsed: {} s'.format(round((stop-start),1)))
  return embedding_matrix


def normalize_embedding(path, embedding_matrix):

  corpus = np.load(path[:-5] + '/feature_tokens.npy', allow_pickle=True)
  print('Creating a Normalized embedding based on the present dataset')
  unique_tokens, frequency = np.unique(corpus, return_counts=True)
  freq_dict = {token: freq for token, freq in zip(unique_tokens, frequency)}
  vector_dict = {token: embedding_matrix[token] for token in unique_tokens}

  print('Calculating Mean of vectors in the whole corpus...')
  Ev = np.sum([freq_dict[token] * vector_dict[token] for token in unique_tokens], axis=0)
  print('Calculating Variance and standard deviation of vectors in the whole corpus...')
  variance = np.sum([freq_dict[token] * np.square(vector_dict[token] - Ev) for token in unique_tokens], axis=0)
  std_dev = np.sqrt(variance)
  norm_embedding_matrix = [(vector - Ev) / std_dev for vector in embedding_matrix]
  return norm_embedding_matrix, corpus

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--pickelPath',type=str, default=os.path.abspath(os.getcwd())+"\\VAT\\Input1\\meta")
  parser.add_argument('--glovePath',type=str, default=os.path.abspath(os.getcwd())+"\\VAT")
  parser.add_argument('--EMBEDDING_DIM', default=300, type=int)

  args = parser.parse_args()
  """
  Embedding matrix will have one row extra than word_index
  Reason: 0th row is by default set to zero because word_index.pickle starts from index value 1
          1st row is again zero; belongs to keyword UNK
          all words not found in the glove matrix belongs to value of zero again.
  """
  # cwd = os.getcwd()
  # print('Working Directory: ', cwd)
  # EMBEDDING_DIM = 300
  # path = cwd + '/Fake News Spacy/data/meta'

  word_index = load_word_index(args.pickelPath)
  embeddings_index = load_glove(args.glovePath, args.EMBEDDING_DIM)
  embedding_matrix = create_emb_matrix(args,word_index, embeddings_index, args.pickelPath)
  del embeddings_index

  np.save(args.pickelPath + '/emb_matrix.npy', embedding_matrix)
  print('Embedding matrix saved locally in npy format')

  # norm_embedding_matrix, corpus = normalize_embedding(path, embedding_matrix)
  # np.save(path + '/norm_emb_matrix.npy', norm_embedding_matrix)
  # print('Normalised embedding matrix saved locally in npy format at: \n', path)