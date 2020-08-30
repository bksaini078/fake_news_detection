import os
import re
import time
import spacy
import contractions
import numpy as np
import pandas as pd
import argparse
import numpy as np


# for spacy
parser = argparse.ArgumentParser()
parser.add_argument('--inputPath',type=str, default=os.path.abspath(os.getcwd())+"\\VAT\\Input1\\raw\\")
parser.add_argument('--labelfilename', type=str,default='label.csv')
parser.add_argument('--unlabelfilename', type=str,default='unlabel.csv')
parser.add_argument('--outputPath',type=str, default=os.path.abspath(os.getcwd())+"\\VAT\\Input1\\processed\\")
parser.add_argument('--Remain', type=str,default="lable.csv")
parser.add_argument('--xUnlabel', type=str,default="xun_shuffled.npy")
args = parser.parse_args()




# cwd = os.getcwd()
# print('Working Directory: ', cwd)
# path = cwd + '/Fake News Spacy/data'

l_df = pd.read_csv(args.inputPath + args.labelfilename)
ul_df = pd.read_csv(args.inputPath + args.unlabelfilename)


l_df.dropna(inplace=True)
ul_df.dropna(inplace=True)

nlp = spacy.load('en_core_web_lg')

def clean_corpus(row):
    # add your custom rules here; also do not change the order
    row = contractions.fix(row)
    row = row.lower()
    row = re.sub(r'<br />', '', row) # only for imdb dataset
    row = re.sub(r'[^a-z]+', ' ', row) # keeps only alphabets
    row = re.sub(r"\b[a-zA-Z]\b", "", row) # removes single characters
    row = re.sub(' +', ' ', row) # removes extra spaces
    return row

def stopwords(tokens):
    if tokens.is_stop == False:
        return tokens.lemma_

def spacy_tokenize(row):
    tokens = pd.Series(nlp(row))
    tokens = tokens.apply(stopwords)
    tokens = tokens[~tokens.isnull()].T
    tokens = tokens.str.cat(sep = ' ')
    return tokens

features_l = [l_df.Article, ul_df.Article]
processed_l = []
for features in features_l:

    start = time.time()
    print('Cleaning dataset...')
    features = features.apply(clean_corpus)
    features = features.apply(spacy_tokenize)
    stop = time.time()
    print('Time elapsed: {} s'.format(round((stop-start),1)))
    processed_l.append(features)


l_df = pd.DataFrame([processed_l[0], l_df.Label]).T
ul_df = processed_l[1]
l_df.to_csv(args.outputPath + args.Remain, index=False)
np.save(args.outputPath + args.xUnlabel, ul_df)

x_train = l_df

