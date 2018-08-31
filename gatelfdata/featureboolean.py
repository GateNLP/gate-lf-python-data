"""Module for the FeatureBoolean class"""

import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
streamhandler = logging.StreamHandler(stream=sys.stderr)
formatter = logging.Formatter(
                '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
streamhandler.setFormatter(formatter)
logger.addHandler(streamhandler)


class FeatureBoolean(object):

    def __init__(self, fname, attrinfo, featurestats):
        """For now, we do not do anything fancy for numeric features."""
        logger.debug("Creating a FeatureBoolean from fname/attrinfo=%r/%r", fname, attrinfo)
        self.fname = fname
        self.attrinfo = attrinfo
        self.featurestats = featurestats

    def type_converted(self):
        return "float"

    def type_original(self):
        return "boolean"

    @staticmethod
    def bool2float(val):
        if val:
            return float(1.0)
        return float(0.0)

    def __call__(self, valueorlist, normalize=None):
        """Converts True to float(1.0) and False to float(0.0)"""
        if normalize:
            raise Exception("Normalization not supported for boolean features")
        if isinstance(valueorlist, list):
            return [FeatureBoolean.bool2float(x) for x in valueorlist]
        return valueorlist

    def __str__(self):
        return "FeatureBoolean(name=%s" % self.fname

    def __repr__(self):
        return "FeatureBoolean(name=%r" % self.fname
