from . vocabs import Vocabs
import logging

# This represents a simple nominal (string) attribute that should
# get encoded as a one-hot vector of values


class FeatureNominal1ofk(object):

    def __init__(self, fname, attrinfo, featurestats):
        """Create the instance from the given meta info of an input feature"""
        logger = logging.getLogger(__name__)
        logger.debug("Creating FeatureNominal1ofk instance for fname/attrinfo=%r/%r", fname, attrinfo)
        self.fname = fname
        self.attrinfo = attrinfo
        self.featurestats = featurestats
        self.vocab = Vocabs.get_vocab(attrinfo)

    def type_converted(self):
        return "index"

    def type_original(self):
        return "nominal"

    def __call__(self, value, normalize=None):
        """Convert a value of the expected type for this feature to a value that can be
        fed into the corresponding input unit of the network
        TODO: this does not yet work properly and needs more thinking: we probably want
        to be able to also use indices for onehot features in some cases and defer
        conversion to one-hot vectors to the network?
        For now, we only use this for the target where it is ok and good to have
        one-hot vectors from the start.
        """
        if normalize:
            raise Exception("Normalization does not make sense for onehot nominal features")
        return self.vocab.string2onehot(value)

    def __str__(self):
        return "FeatureNominal1ofk(name=%s)" % self.fname

    def __repr__(self):
        return "FeatureNominal1ofk(name=%r)" % self.fname
