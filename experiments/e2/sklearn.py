import numpy as np
import sklearn.metrics.pairwise as sk

def _distance(coord1, coord2):
  return sk.cosine_distances([coord1], [coord2])[0][0] # sklearn wraps the result in a 2D array

def _evaluate_cosine_similarity(phrase_vectors, entry_phrases, MeSH_phrases):
  entry_vectors = [phrase_vectors[entry_phrase] for entry_phrase in entry_phrases]
  MeSH_vectors = [phrase_vectors[MeSH_phrase] for MeSH_phrase in MeSH_phrases]
  similarity = sk.cosine_similarity(entry_vectors, MeSH_vectors)
  # Unable to allocate 63.6 GiB for an array with shape (92423, 92423) and data type float64
  print('Cosine Similarity: ')
  print(similarity)

def _evaluate_distance_average_precision(phrase_vectors, entry_phrases, MeSH_phrases):
  all_distances = [] # distances between all Entry and MeSH terms
  for i in range(len(entry_phrases)):
    all_distances.append(_distance(phrase_vectors[entry_phrases[i]], phrase_vectors[MeSH_phrases[i]]))
  print('Average Distance: ' + str(np.mean(all_distances)))


def run(phrase_vectors, entry_phrases, MeSH_phrases):
  print('Results for Experiment 2 (Sklearn)')
  #_evaluate_cosine_similarity(phrase_vectors, entry_phrases, MeSH_phrases)
  _evaluate_distance_average_precision(phrase_vectors, entry_phrases, MeSH_phrases)
  print()

