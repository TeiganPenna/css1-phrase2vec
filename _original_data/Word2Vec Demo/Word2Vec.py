import re
import logging
import numpy as np
import pandas as pd
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import word2vec, Word2Vec


'''
Model Building and Training
'''

# Read Text Input, each line contains the text of a paper's title and abstract
Raw = ' '.join([line.strip('\n').lower() for line in open('1.Raw Input.txt', 'r', encoding= 'UTF-8').readlines()])
print('1.Model Training - Input Read.')

# Remove all the punctuations in the Text
Raw = re.sub(r"[{}]+".format(punctuation), '', Raw)

# Remove all the pure number tokens
# Raw = re.sub(r'\b[0-9]*\b', '', Raw)

# Tokenization
Word_Tokens = word_tokenize(Raw)

# Remove all the stop word tokens
StopWords = set(stopwords.words('english'))
Word_Tokens = [token for token in Word_Tokens if token not in StopWords]
print('2.Model Training - Pre-processing Done.')

# Model Training
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
model = word2vec.Word2Vec([Word_Tokens], size=100, window=5, min_count=1, workers=5)
model.save('Word2Vec Model')
print('3.Model Training - Model Trained.')

# Output the Vocab
f = open('2.Model Vocab.txt', 'w+', encoding='UTF-8')
vocab = list(model.wv.vocab)
for word in vocab:
    f.write(str(word) + '\n')
f.close()
print('4.Model Training - Vocab Saved.')
