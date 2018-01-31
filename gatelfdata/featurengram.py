from __future__ import print_function
import sys
from . vocabs import Vocabs

class FeatureNgram(object):
    """Represents an ngram attribute. The value of such an attribute is a list/sequence of
    things that can be represented by embeddings, """
    def __init__(self, fname, attrinfo, featurestats):
        """Create the instance from the given meta info of an input feature"""
        print("DEBUG: creating FeatureNgram instance for fname=", fname, "attrinfo=", attrinfo, file=sys.stderr)
        self.fname = fname
        self.attrinfo = attrinfo
        self.featurestats = featurestats
        self.vocab = Vocabs.addOrReturnVocab(attrinfo, featurestats)

    def __call__(self, value):
        """Convert a value of the expected type for this feature to a value that can be
        fed into the corresponding input unit of the network"""

        # ok, for an ngram we expect the value to be a list, in which case we
        # create a new list with the string idices of the values
        # otherwise, we report an error
        if isinstance(value, list):
            ret = [self.vocab.string2idx(v) for v in value]
            return ret
        else:
            raise Exception("Value for converting FeatureNgram not a list but {} of type {}".format(value,type(value)))

    def __str__(self):
        return "FeatureNgram(name="+self.fname+")"
