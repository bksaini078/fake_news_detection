import numpy as np
import pandas as pd
import random
import os
import argparse



# for spacy
parser = argparse.ArgumentParser()
parser.add_argument('--Path',type=str, default=os.path.abspath(os.getcwd())+"\\VAT\\Input1\\processed\\")
parser.add_argument('--filename', type=str,default="lable.csv")
parser.add_argument('--xUnlabel', type=str,default="xun_shuffled.npy")
parser.add_argument('--xTrain', type=str,default="x_train.npy")
parser.add_argument('--xTest', type=str,default="x_test.npy")
parser.add_argument('--yTrain', type=str,default="y_train.npy")
parser.add_argument('--yTest', type=str,default="y_test.npy")
args = parser.parse_args()
# cwd = os.getcwd()
# print('Current working directory: ', cwd)
# path = cwd + '/Fake News Spacy/data'


l_df = pd.read_csv(args.Path + args.filename)



features, labels = l_df.Article, l_df.Label

split_ratio = 0.25

inds = np.arange(features.shape[0])
random.Random(1).shuffle(inds)
features = features[inds]
labels = labels[inds]
features = features
labels = labels

num_test_samples = int(split_ratio * features.shape[0])
print('Split ratio {}/{}:'.format(num_test_samples, features.shape[0]))

x_train = features[:-num_test_samples]
y_train = labels[:-num_test_samples]
x_test = features[-num_test_samples:]
y_test = labels[-num_test_samples:]
print("Training size:", x_train.shape, y_train.shape)
print("Testing size:", x_test.shape, y_test.shape)

np.save(args.Path + args.xTrain, x_train)
np.save(args.Path + args.yTrain, y_train)
np.save(args.Path + args.xTest , x_test)
np.save(args.Path + args.yTest, y_test)
print('Done')