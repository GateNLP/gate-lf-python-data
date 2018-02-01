from __future__ import print_function
import sys
from . vocabs import Vocabs
import logging

# This represents a simple nominal (string) attribute that should
# get encoded as a dense embedding vector

class FeatureNominalEmbs(object):


    def __init__(self, fname, attrinfo, featurestats):
        """Create the instance from the given meta info of an input feature"""
        logger = logging.getLogger(__name__)
        logger.debug("Creating FeatureNgram instance for fname/attrinfo=%r/%r", fname, attrinfo)
        self.fname = fname
        self.attrinfo = attrinfo
        self.featurestats = featurestats
        self.vocab = Vocabs.get_vocab(attrinfo)


    def __call__(self, value):
        """Convert a value of the expected type for this feature to a value that can be
        fed into the corresponding input unit of the network"""
        return self.vocab.string2idx(value)

    def __str__(self):
        return "FeatureNominalEmbs(name=%s" % self.fname

    def __repr__(self):
        return "FeatureNominalEmbs(name=%r" % self.fname
