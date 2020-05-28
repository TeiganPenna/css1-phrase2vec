from annoy import AnnoyIndex
from gensim.models import word2vec, Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, MWETokenizer
import numpy as np
import pandas as pd
from pathlib import Path
import re
from string import punctuation

def get_token_vectors_from_xlsx(filename):
  dataframe = pd.read_excel(filename, index_col=0, header=0)
  tdataframe = dataframe.transpose() # seems the easiest way to get the information out of the dataframe. Pandas likes columns a lot
  return tdataframe.to_dict('list')
 
def get_phrase_vectors_from_raw_phrases(word_vectors, entry_phrases, MeSH_phrases):
  # assume that the files are the same length
  phrase_vectors = {}
  for entry_phrase in entry_phrases:
    word_tokens = word_tokenize(entry_phrase)
    entry_phrase_vector = np.mean([word_vectors[word_token] for word_token in word_tokens], axis=0)
    phrase_vectors[entry_phrase] = entry_phrase_vector
  for MeSH_phrase in MeSH_phrases:
    word_tokens = word_tokenize(MeSH_phrase)
    MeSH_phrase_vector = np.mean([word_vectors[word_token] for word_token in word_tokens], axis=0)
    phrase_vectors[MeSH_phrase] = MeSH_phrase_vector
  return phrase_vectors

def get_word_weights():
  dataframe = pd.read_csv(Path('data/word_weights.csv'), index_col=0)
  d = dataframe.to_dict('split')
  d = dict(zip(d['index'], d['data']))
  # oh no null STILL isn't in the dataframe
  d['null'] = float('9.84991624674156e-06')
  return d

def get_weighted_phrase_vectors_from_raw_phrases(word_vectors, word_weights, entry_phrases, MeSH_phrases):
  # assume that the files are the same length
  phrase_vectors = {}
  for entry_phrase in entry_phrases:
    word_tokens = word_tokenize(entry_phrase)
    entry_phrase_vector = np.mean([np.multiply(word_vectors[word_token], word_weights[word_token]) for word_token in word_tokens], axis=0)
    phrase_vectors[entry_phrase] = entry_phrase_vector
  for MeSH_phrase in MeSH_phrases:
    word_tokens = word_tokenize(MeSH_phrase)
    MeSH_phrase_vector = np.mean([np.multiply(word_vectors[word_token], word_weights[word_token]) for word_token in word_tokens], axis=0)
    phrase_vectors[MeSH_phrase] = MeSH_phrase_vector
  return phrase_vectors

def get_annoy_index(phrase_vectors, metric):
  dimensions = len(list(phrase_vectors.values())[0]) # assume that the dimensions of the vectors are all equal to the first
  index = AnnoyIndex(dimensions, metric)
  keys_to_phrases = {} # must keep my own map because Annoy doesn't do it for me
  phrases_to_keys = {} # ditto
  current_index = 1
  for phrase in phrase_vectors:
    keys_to_phrases[current_index] = phrase
    phrases_to_keys[phrase] = current_index
    index.add_item(current_index, phrase_vectors[phrase])
    current_index += 1
  index.build(10) # 10 Trees is apparently a good amount for accuracy
  return index, keys_to_phrases, phrases_to_keys

def train_phrase_model_to_xlsx(corpus_file, out_file, phrases):
  raw = ' '.join([line.strip('\n').lower() for line in open(corpus_file, 'r', encoding='UTF-8').readlines()])
  raw = re.sub(r"[{}]+".format(punctuation), '', raw)
  word_tokens = word_tokenize(raw)

  tokenizer = MWETokenizer([word_tokenize(phrase) for phrase in phrases], separator = '_')
  word_tokens = tokenizer.tokenize(word_tokens)

  stop_words = set(stopwords.words('english'))
  word_tokens = [token for token in word_tokens if token not in stop_words]

  model = word2vec.Word2Vec([word_tokens], size=100, window=5, min_count=1, workers=5)
  df = pd.DataFrame([model.wv.get_vector(word) for word in model.wv.vocab], index=model.wv.vocab)
  df.to_excel(out_file)

def phrase_to_token(phrase):
  return re.sub(r"\s", '_', phrase)

