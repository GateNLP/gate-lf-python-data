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


def shorten(thelist, maxlen, shorten):
    """
    If thelist has more elements than maxlen, remove elements to make it of size maxlen.
    The parameter shorten is a string which can be one of left, right, both or middle
    and specifies where to remove elements.
    :param thelist: the list to shorten
    :param maxlen: maximum length of the list
    :param shorten: one of left, right, both, middle, where to remove the elements
    :return: the shortened list
    """
    if len(thelist <= maxlen):
        return thelist
    if shorten == "right":
        return thelist[0:maxlen]
    elif shorten == "left":
        return thelist[-maxlen-1:-1]
    elif shorten == "both":
        excess = len(thelist) - maxlen;
        left = int(excess/2)
        right = excess - left
        return thelist[left:-right]
    elif shorten == "middle":
        excess = len(thelist) - maxlen;
        left = int(excess/2)
        right = excess - left
        return thelist[0:left]+thelist[-right:]
    else:
        raise Exception("Not a valid value for the shorten setting: {}".format(shorten))


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
        self.shorten = attrinfo.get("shorten", "")
        # NOTE: the maxlen/shorten info gets update by any config specs in the dataset
        if not self.shorten:
            self.shorten = "right"
        self.maxlen = attrinfo.get("maxlen", 0)

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
            if self.maxlen > 0:
                value = shorten(value, self.maxlen, self.shorten)
            ret = [self.vocab.string2idx(v) for v in value]
            return ret
        else:
            raise Exception("Value for converting FeatureNgram not a list but {} of type {}".format(value, type(value)))

    def __str__(self):
        return "FeatureNgram(name=%s)" % self.fname

    def __repr__(self):
        return "FeatureNgram(name=%r)" % self.fname
