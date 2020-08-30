import tensorflow as tf 
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Lambda,Bidirectional
from tensorflow.keras.models import Model
from tensorflow import keras

def BiLstmModel(maxlen, vocab_size):
  tf.keras.backend.clear_session()
  inputs = keras.Input(shape=(maxlen,))
  x =Embedding(vocab_size, 128, input_length=None)(inputs)
  x =Bidirectional(LSTM(128))(x)
#   x = Dropout(0.2)(x)
#   x =Dense(64,activation='relu')(x)
  x =Dense(2)(x)
  x =Dense(1, activation='sigmoid')(x)
  return Model(inputs,x)