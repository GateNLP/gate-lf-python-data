import logging
# This represents a simple boolean attribute


class FeatureBoolean(object):

    def __init__(self, fname, attrinfo, featurestats):
        """For now, we do not do anything fancy for numeric features."""
        logger = logging.getLogger(__name__)
        logger.debug("Creating a FeatureBoolean from fname/attrinfo=%r/%r", fname, attrinfo)
        self.fname = fname
        self.attrinfo = attrinfo
        self.featurestats = featurestats

    @staticmethod
    def bool2float(val):
        if val:
            return float(1.0)
        else:
            return float(0.0)

    def __call__(self,valueorlist):
        """Converts True to float(1.0) and False to float(0.0)"""
        if type(valueorlist) == list:
            return [FeatureBoolean.bool2float(x) for x in valueorlist]
        else:
            return valueorlist

    def __str__(self):
        return "FeatureBoolean(name=%s" % self.fname

    def __repr__(self):
        return "FeatureBoolean(name=%r" % self.fname
