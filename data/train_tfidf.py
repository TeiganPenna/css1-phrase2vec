from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import re
from string import punctuation

# get the raw words
raw_text = ' '.join([line.strip('\n').lower() for line in open('1.Raw Input.txt', 'r', encoding= 'UTF-8').readlines()])
raw_text = re.sub(r"[{}]+".format(punctuation), '', raw_text)
#raw_text = re.sub(r'\b[0-9]*\b', '', raw_text)
word_tokens = word_tokenize(raw_text)
stopwords = set(stopwords.words('english'))
word_tokens = [token for token in word_tokens if token not in stopwords]

# create tf-idf map
text_length = len(word_tokens)
print('text length is: ' + str(text_length))
word_counts = {}
for token in word_tokens:
  if token in word_counts:
    word_counts[token] += 1
  else:
    word_counts[token] = 1
print('unique word count is: ' + str(len(word_counts)))
word_weights = {}
for word in word_counts.keys():
  word_weights[word] = word_counts[word] / text_length

# write out tf-idf map
dataframe = pd.DataFrame.from_dict(word_weights, orient='index')
dataframe.to_csv('word_weights.csv')

