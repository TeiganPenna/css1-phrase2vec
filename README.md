# 41078 Computing Science Studio 1
This repository concerns the experiments done as part of the 41078 course

Three experiments are done:
1. Phrase vectors are calculated as a mean of the vectors of their component words
2. Phrase vectors are calculated as a mean of the vectors of their component words weighted by word frequency
3. Phrase vectors are calculated by replacing the phrases in the corpus with single n-gram tokens and treating them as words during training

Tests are done using two lists of synonymous phrases.

Each experiment has the following heuristics:
1. True Precision of closeness - Using [AnnoyIndex](https://github.com/spotify/annoy) angular and euclidean (% of phrases who's closest neighbour is its synonym)
2. Mean distance - Using [AnnoyIndex](https://githup.com/spotify/annoy)
3. Sklearn distance

