# File Formats

The library expects the data in two files, located in the same directory and
their names only differing in the extension:
* Meta file, with extension ".meta": a JSON format file that contains information
  about the data, the attributes/features defined in the LearningFramework,
  and contains statistics about the distribution of values for each feature.
* Data file, with extension ".data": a file where each line is a JSON object
  representing a single instance, the exact format of the JSON object
  depends on the learning task.

## Meta File

The meta file contains the JSON representation of a (nested) map/dictionary.
The tope level entries in the map are:
* `featureNames`:
* `isSequence`:
* `featureStats`:
* `features`:
* `savedOn`:
* `sequLengths.min`:
* `sequLengths.max`:
* `sequLengths.mean`:
* `sequLengths.variance`:
* `targetStats`:
* `dataFile`:
* `linesWritten`:
* `featureInfo`:

## Data File

Each line in the data file is a JSON object representing an instance.
Currently an instance always consists of two parts: the independent data and
the target data. The independent data is either a list of features if `isSequence`
is false, or it is a list of sequence elements, where each element is a list
of features, if `isSequence` is true.

The target data is either a numeric or nominal value if `isSequence` is false,
or a list of nominal values, if `isSeqience` is true.

Example instance when `isSequence` is true and there is just one feature per
sequence element. Here each element of the sequence contains only one nominal/string
feature (the token text):

```
[[["EU"],["rejects"],["German"],["call"],["to"],["boycott"],["British"],["lamb"],["."]],["NNP","VBZ","JJ","NN","TO","VB","JJ","NN","."]]
```

Example instance when `isSequence` is false. Here an instance has a XXX features,
which may come (according to the LearningFramework features definition) from the
token to be classified or from preceding or following tokens:

```
[["of","adding","a","Patent","Pending","message","VERB","DET","a","a","a","Aa","Aa","a","","ng","","nt","ng","ge","","ing","","ent","ing","age"],"NOUN"]
```
