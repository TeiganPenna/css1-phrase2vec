from Phrase2Vec import get_annoy_index, phrase_to_token
import numpy as np

def _distance(index, phrases_to_keys, phrase1, phrase2):
  return index.get_distance(phrases_to_keys[phrase_to_token(phrase1)], phrases_to_keys[phrase_to_token(phrase2)])

def _closest(index, keys_to_phrases, coord, n=2):
  identifiers = index.get_nns_by_vector(coord, n)
  return [keys_to_phrases[i] for i in identifiers]

def _evaluate_true_precision(token_vectors, index, keys_to_phrases, entry_phrases, MeSH_phrases):
  # Evaluate precision
  true_precision = 0
  total_records = len(entry_phrases)

  for i, entry_phrase in enumerate(entry_phrases):
    closest_synonyms = _closest(index, keys_to_phrases, token_vectors[phrase_to_token(entry_phrase)])
    closest_synonym = closest_synonyms[0]
    if closest_synonym == entry_phrase:
      # The closest synonym is itself. If the closest synonym is NOT itself it is because
      # the vectors of the first and second synonym are the same
      closest_synonym = closest_synonyms[1]

    if closest_synonym == phrase_to_token(MeSH_phrases[i]):
      true_precision += 1

  print('True Precision: ' + str((true_precision / total_records) * 100) + '%')

def _evaluate_distance_average_precision(index, phrases_to_keys, entry_phrases, MeSH_phrases):
  all_distances = [] # distances between all Entry and MeSH terms
  for i in range(len(entry_phrases)):
    all_distances.append(_distance(index, phrases_to_keys, entry_phrases[i], MeSH_phrases[i]))
  print('Average Distance: ' + str(np.mean(all_distances)))


def run(token_vectors, entry_phrases, MeSH_phrases):
  index, keys_to_phrases, phrases_to_keys = get_annoy_index(token_vectors, 'angular')

  print('Results for Experiment 3 (Cosine)')
  _evaluate_true_precision(token_vectors, index, keys_to_phrases, entry_phrases, MeSH_phrases)
  _evaluate_distance_average_precision(index, phrases_to_keys, entry_phrases, MeSH_phrases)
  print()

