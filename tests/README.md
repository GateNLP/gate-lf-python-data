# Test data

Created from two corpora:
* the sentence polarity dataset v1.0 (http://www.cs.cornell.edu/people/pabo/movie-review-data/)
  converted to GATE and used by the LF to create a training meta and data file.
  (instance classification using one NGRAM feature)
  Meta file copied to class-ngram-sp1.meta.json
  Data file shuffled and reduced to a random subset of 1000 lines to class-ngram-sp1.data.json
* the English corpus from the Universal dependencies (https://github.com/UniversalDependencies/UD_English)
  converted to GATE and used by the LF to create a training met and data file.
  (instance classification using a window around the instance)
  Meta file copied to class-window-pos1.meta.json
  Data file shuffled and reduced to a random subset of 1000 lines to class-window-pos1.data.json
* Same corpus Universal dependencies,
  converted to GATE and used by the LF to create a training meta and data file
  (element classification within a sequence, each element having several features)
  Meta file copied to seq-pos1.meta.json
  Data file shuffled and reduced to a random subset of 1000 lines to  seq-pos1.data.json
* Ionosphere dataset converted to GATE and used by LF to export the training set.
  (instance classification from several numeric features)
  Meta file copied to class-iris.meta.json
  Data file copied to class-iris.data.json
