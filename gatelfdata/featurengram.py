"""Module for the FeatureNGram class"""

import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
streamhandler = logging.StreamHandler(stream=sys.stderr)
formatter = logging.Formatter(
                '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
streamhandler.setFormatter(formatter)
logger.addHandler(streamhandler)


class FeatureNgram(object):
    """Represents an ngram attribute. The value of such an attribute is a list/sequence of
    things that can be represented by embeddings, """
    def __init__(self, fname, attrinfo, featurestats, vocab):
        """Create the instance from the given meta info of an input feature"""
        logger.debug("Creating FeatureNgram instance for fname/attrinfo=%r/%r", fname, attrinfo)
        self.fname = fname
        self.attrinfo = attrinfo
        self.featurestats = featurestats
        self.vocab = vocab

    def type_converted(self):
        """Return the name of the type of information of the feature, once it is converted to
        internal format."""
        return "indexlist"

    def type_original(self):
        """Return the name of the type of information of the original feature."""
        return "ngram"

    def __call__(self, value, normalize=None):
        """Convert a value of the expected type for this feature to a value that can be
        fed into the corresponding input unit of the network"""
        if normalize:
            raise Exception("Normalization does not make sense for ngram features")
        # ok, for an ngram we expect the value to be a list, in which case we
        # create a new list with the string idices of the values
        # otherwise, we report an error
        if isinstance(value, list):
            ret = [self.vocab.string2idx(v) for v in value]
            return ret
        else:
            raise Exception("Value for converting FeatureNgram not a list but {} of type {}".format(value, type(value)))

    def __str__(self):
        return "FeatureNgram(name=%s)" % self.fname

    def __repr__(self):
        return "FeatureNgram(name=%r)" % self.fname
