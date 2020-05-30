import Phrase2Vec as p2v
import experiments.e1.cosine
import experiments.e1.euclidean
import experiments.e1.sklearn
import experiments.e2.cosine
import experiments.e2.euclidean
import experiments.e2.sklearn
import experiments.e3.cosine
import experiments.e3.euclidean
import experiments.e3.sklearn

from pathlib import Path
import re
from string import punctuation

word_vectors = p2v.get_token_vectors_from_xlsx(Path('data/3.Filtered_word_vectors.xlsx'))

entry_phrases = [re.sub(r"[{}]+".format(punctuation), '', line.strip('\n').lower()) for line in open(Path('data/Train_Set_Entry_Terms.txt'), 'r', encoding='UTF-8').readlines()]
MeSH_phrases = [re.sub(r"[{}]+".format(punctuation), '', line.strip('\n').lower()) for line in open(Path('data/Train_Set_MeSH_Terms.txt'), 'r', encoding='UTF-8').readlines()]

# use this to create the data for experiment 3
#combined_phrases = entry_phrases + MeSH_phrases
#p2v.train_phrase_model_to_xlsx(Path('data/1.Raw Input.txt'), Path('data/experiment3_vectors.xlsx'), combined_phrases)

e3_vectors = p2v.get_token_vectors_from_xlsx(Path('data/experiment3_vectors.xlsx'))
corpus_entry_phrases, corpus_MeSH_phrases = p2v.get_phrases_in_corpus(e3_vectors, entry_phrases, MeSH_phrases)

phrase_vectors = p2v.get_phrase_vectors_from_raw_phrases(word_vectors, entry_phrases, MeSH_phrases)
experiments.e1.cosine.run(phrase_vectors, entry_phrases, MeSH_phrases)
experiments.e1.euclidean.run(phrase_vectors, entry_phrases, MeSH_phrases)
experiments.e1.sklearn.run(phrase_vectors, entry_phrases, MeSH_phrases)

word_weights = p2v.get_word_weights()
weighted_phrase_vectors = p2v.get_weighted_phrase_vectors_from_raw_phrases(word_vectors, word_weights, entry_phrases, MeSH_phrases)
experiments.e2.cosine.run(weighted_phrase_vectors, entry_phrases, MeSH_phrases)
experiments.e2.euclidean.run(weighted_phrase_vectors, entry_phrases, MeSH_phrases)
experiments.e2.sklearn.run(weighted_phrase_vectors, entry_phrases, MeSH_phrases)

experiments.e3.cosine.run(e3_vectors, corpus_entry_phrases, corpus_MeSH_phrases)
experiments.e3.euclidean.run(e3_vectors, corpus_entry_phrases, corpus_MeSH_phrases)
experiments.e3.sklearn.run(e3_vectors, corpus_entry_phrases, corpus_MeSH_phrases)

