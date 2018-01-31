from __future__ import print_function
import sys
from . vocabs import Vocabs


# This represents a simple nominal (string) attribute that should
# get encoded as a dense embedding vector

class FeatureNominalEmbs(object):


    def __init__(self, fname, attrinfo, featurestats):
        """Create the instance from the given meta info of an input feature"""
        print("DEBUG: creating FeatureNgram instance for fname=", fname, "attrinfo=", attrinfo, file=sys.stderr)
        self.datatype = datatype
        self.attrinfo = attrinfo
        self.featurestats = featurestats
        self.vocab = Vocabs.addOrReturnVocab(attrinfo, featurestats)


    def __call__(self, value):
        """Convert a value of the expected type for this feature to a value that can be
        fed into the corresponding input unit of the network"""
        return self.vocab.string2idx(value)

    def __str__(self):
        return "FeatureNgram(name="+self.fname+")"
