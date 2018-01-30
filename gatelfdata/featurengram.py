from __future__ import print_function
import sys
from . featurenominalembs import FeatureNominalEmbs

class FeatureNgram(object):
    """Represents an ngram attribute. The value of such an attribute is a list/sequence of
    things that can be represented by embeddings, """
    def __init__(self, fname, attrinfo, featurestats):
        """Create the instance from the given meta info of an input feature"""
        print("DEBUG: creating FeatureNgram instance for fname=", fname, "attrinfo=", attrinfo, file=sys.stderr)
        self.embs = FeatureNominalEmbs.addEmbeddings(attrinfo)

    def __call__(self, value):
        """Convert a value of the expected type for this feature to a value that can be
        fed into the corresponding input unit of the network"""
        raise Exception("NOT YET IMPLEMENTED, cannot convert value {}".format(value))
        #return value

    def __str__(self):
        return "FeatureNgram(name="+self.fname+")"
