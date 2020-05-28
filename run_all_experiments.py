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
# oh no null doesn't work for some reason
word_vectors['null'] = [-0.000586311, -0.004722272, -0.002906141, -0.002670823, 0.002514862, 0.001657518, -0.004780879, 0.001601622, 0.004698421, 0.004740661, 0.000152729, -0.0022193, 0.002339627, 0.003865311, 0.001690057, 0.002568416, 0.003261337, -0.001853965, 0.001590472, 0.001772919, -9.46451E-05, 0.004367359, 0.00119876, 0.003234746, -0.000257252, -0.003920502, 0.00465125, 0.002703287, -0.004533189, 0.000935598, -0.001213048, -0.000595104, -0.0021956, -0.001821463, -0.004125374, 0.001823117, 0.004232661, -0.00401916, 0.001707869, -0.001182369, -0.003271647, -0.003505243, -0.00197784, 0.002007429, 0.003653463, -0.001127671, -0.000143878, -0.002803235, -0.004280376, -0.001616072, 0.001502794, -0.003097014, 0.000809615, -0.001676369, 0.002434121, 0.001218057, -0.000411161, 0.00285347, 0.002694165, 0.004046326, 0.003195991, 0.001733728, -0.001453032, -0.004054314, 0.000815504, 0.004497233, -0.000321392, 0.001979934, 0.000896316, -0.001518743, -0.001924197, -0.004466464, 0.001888455, 0.00367528, 0.000310995, 0.001513249, 0.003939437, -0.001089512, 0.002640106, -0.00034135, -0.003495516, -0.002155305, 0.00288133, 0.004762273, -0.00315044, 0.000874343, 0.000788593, -0.003267953, 0.004678701, 0.00416028, -0.001871165, -0.001419193, 0.001846539, 0.00302519, -0.00177639, -0.003800438, 0.003782798, 0.001496537, -0.001303208, -0.004373099]

entry_phrases = [re.sub(r"[{}]+".format(punctuation), '', line.strip('\n').lower()) for line in open(Path('data/Train_Set_Entry_Terms.txt'), 'r', encoding='UTF-8').readlines()]
MeSH_phrases = [re.sub(r"[{}]+".format(punctuation), '', line.strip('\n').lower()) for line in open(Path('data/Train_Set_MeSH_Terms.txt'), 'r', encoding='UTF-8').readlines()]

phrase_vectors = p2v.get_phrase_vectors_from_raw_phrases(word_vectors, entry_phrases, MeSH_phrases)
experiments.e1.cosine.run(phrase_vectors, entry_phrases, MeSH_phrases)
experiments.e1.euclidean.run(phrase_vectors, entry_phrases, MeSH_phrases)
experiments.e1.sklearn.run(phrase_vectors, entry_phrases, MeSH_phrases)

word_weights = p2v.get_word_weights()
weighted_phrase_vectors = p2v.get_weighted_phrase_vectors_from_raw_phrases(word_vectors, word_weights, entry_phrases, MeSH_phrases)
experiments.e2.cosine.run(weighted_phrase_vectors, entry_phrases, MeSH_phrases)
experiments.e2.euclidean.run(weighted_phrase_vectors, entry_phrases, MeSH_phrases)
experiments.e2.sklearn.run(weighted_phrase_vectors, entry_phrases, MeSH_phrases)

# use this to create the data fro experiment 3
combined_phrases = entry_phrases + MeSH_phrases
#p2v.train_phrase_model_to_xlsx('data\\1.Raw Input.txt', 'data\\experiment3_vectors.xlsx', combined_phrases)

token_vectors = p2v.get_token_vectors_from_xlsx(Path('data/experiment3_vectors.xlsx'))
# something is getting caught in get_token_vectors_from_xlsx.
# oh no null doesn't work for some reason
token_vectors['null'] = [-0.004709332, 0.001775584, -0.004567288, -0.00246457, 0.001083848, -0.003429412, 0.000862219, -0.002232632, -0.00304723, -0.004944438, 0.004596454, -0.004404955, -0.003000282, -0.003165435, -0.002200568, 0.004271225, -8.48173E-05, -0.004283414, -0.003202339, 0.004270182, -0.000168136, -0.003539313, -0.000361949, 0.001319244, 0.002541783, -0.000720611, -0.00047383, 0.004293929, -0.00308211, -0.003346572, -0.004855514, 0.001246628, -0.001289282, 0.000895427, 0.002952904, -0.00164363, 0.002982932, 0.00347599, -0.002241338, -0.001309626, -0.001313913, 0.000976774, 0.001683609, 0.001426026, 0.00135383, 0.00436144, 0.00487035, -0.004040735, -0.001417252, 0.004737437, -0.004147124, -0.004218988, 0.002033669, -0.001542457, 0.001045715, -0.004644167, 0.00161695, 0.000614658, -0.001403821, 0.004211721, 0.003194033, 0.000245315, -0.000795979, -0.002494662, 0.002430575, 0.003901142, -0.004116238, -0.000143165, -0.003787928, -0.003964372, 0.004634351, -0.004172778, -0.001846442, 0.001292997, -0.00304682, 0.003727357, -0.002919678, 0.003675942, -0.002565662, 0.000781305, 0.003711185, 0.000342864, 0.004459185, 0.001390404, -0.001399887, 0.003102353, -0.004141059, -0.000152576, -0.004862174, -0.001977625, -0.00172297, -0.001812106, -0.004838639, 0.003739215, -0.000815019, 0.000964606, -0.002474158, -9.43384E-05, -0.003024496, -0.002677346]

# this is a hack to get around the corpus not containing all the Entry/MeSH phrases
phrase_tokens_map = {p2v.phrase_to_token(phrase): phrase for phrase in combined_phrases}
phrase_vectors = {k: token_vectors[k] for k in phrase_tokens_map.keys() if k in token_vectors}
phrases_that_are_in_the_corpus = [v for k, v in phrase_tokens_map.items() if k in phrase_vectors]
entry_MeSH_map = dict(zip(entry_phrases, MeSH_phrases))
corpus_entry_MeSH_map = [(e, m) for e, m in entry_MeSH_map.items() if e in phrases_that_are_in_the_corpus and m in phrases_that_are_in_the_corpus]
corpus_entry_phrases, corpus_MeSH_phrases = zip(*corpus_entry_MeSH_map)

experiments.e3.cosine.run(token_vectors, corpus_entry_phrases, corpus_MeSH_phrases)
experiments.e3.euclidean.run(token_vectors, corpus_entry_phrases, corpus_MeSH_phrases)
experiments.e3.sklearn.run(token_vectors, corpus_entry_phrases, corpus_MeSH_phrases)


# don't forget to change all the annoy closest to not be 10
