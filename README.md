# 41078 Computing Science Studio 1
This repository concerns the experiments done as part of the 41078 course

Three experiments are done:
1. Phrase vectors are calculated as a mean of the vectors of their component words
2. Phrase vectors are calculated as a mean of the vectors of their component words weighted by word frequency
3. Phrase vectors are calculated by replacing the phrases in the corpus with single n-gram tokens and treating them as words during training

Tests are done using two lists of synonymous phrases.

Each experiment has the following heuristics:
1. True Precision of closeness - Using [AnnoyIndex](https://github.com/spotify/annoy) angular and euclidean (% of phrases who's closest neighbour is its synonym)
2. Mean distance - Using [AnnoyIndex](https://githup.com/spotify/annoy) angular and euclidean
3. Sklearn distance

Experiment results are as follows:
```
Results for Experiment 1 (Cosine)
  True Precision: 10.666176168269804%
  Average Distance: 0.7543520572303769

Results for Experiment 1 (Euclidean)
  True Precision: 10.656438332449715%
  Average Distance: 0.014899424478962172

Results for Experiment 1 (Sklearn)
  Average Distance: 0.42651033318496334

Results for Experiment 2 (Cosine)
  True Precision: 8.083485712430887%
  Average Distance: 0.7337878031360708

Results for Experiment 2 (Euclidean)
  True Precision: 10.475747378899191%
  Average Distance: 6.491201046280378e-06

Results for Experiment 2 (Sklearn)
  Average Distance: 0.4758142731101864

Results for Experiment 3 (Cosine)
  True Precision: 25.885257065032345%
  Average Distance: 0.645577253067675

Results for Experiment 3 (Euclidean)
  True Precision: 25.885257065032345%
  Average Distance: 0.01862989795185907

Results for Experiment 3 (Sklearn)
  Average Distance: 0.45738971238246573
```
