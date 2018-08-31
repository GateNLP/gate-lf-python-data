"""Module for the FeatureNominalEmbs class"""

import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
streamhandler = logging.StreamHandler(stream=sys.stderr)
formatter = logging.Formatter(
                '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
streamhandler.setFormatter(formatter)
logger.addHandler(streamhandler)


# This represents a simple nominal (string) attribute that should
# get encoded as a dense embedding vector


class FeatureNominalEmbs(object):

    def __init__(self, fname, attrinfo, featurestats, vocab):
        """Create the instance from the given meta info of an input feature"""
        logger.debug("Creating FeatureNgram instance for fname/attrinfo=%r/%r", fname, attrinfo)
        self.fname = fname
        self.attrinfo = attrinfo
        self.featurestats = featurestats
        self.vocab = vocab

    def type_original(self):
        return "nominal"

    def type_converted(self):
        return "index"

    def __call__(self, value, normalize=None):
        """Convert a value of the expected type for this feature to a value that can be
        fed into the corresponding input unit of the network"""
        if normalize:
            raise Exception("Normalization does not make sense for ngram features")
        return self.vocab.string2idx(value)

    def __str__(self):
        return "FeatureNominalEmbs(name=%s" % self.fname

    def __repr__(self):
        return "FeatureNominalEmbs(name=%r" % self.fname
