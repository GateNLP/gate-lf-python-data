#!/usr/bin/env python

# test loading a file
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from gatelfdata import Dataset

if len(sys.argv) != 2:
   raise Exception("Need one parameter: meta file")

file = sys.argv[1]

ds = Dataset(file)
print("Dataset loaded, isSequence=%s, nFeatures=%s, nAttrs=%s, nInstances=%s" % (ds.isSequence, ds.nFeatures, ds.nAttrs, ds.nInstances))
# Now list all the vocabs
vocabs = ds.vocabs
for name, vocab in vocabs.vocabs.items():
    print("Name=%s, vocab=%r" % (name, vocab, ))

# test what the encoding for some specific values would be
instance = [[["EU"], ["rejects"], ["German"], ["call"], ["to"], ["boycott"], ["British"], ["lamb"], ["."]], ["I-ORG", "O", "I-MISC", "O", "O", "O", "I-MISC", "O", "O"]]
print("indices for indep=%s" % ds.convert_indep(instance[0]))
print("indices for dep=%s" % ds.convert_dep(instance[1]))

# test how "splitting" with an existing validation file works
# ds.split(convert=True, validation_file="conll-en-ner-trainsubset.data.json")
# valset = ds.validation_set_converted(as_batch=True)
# print("Valset is ", valset)

