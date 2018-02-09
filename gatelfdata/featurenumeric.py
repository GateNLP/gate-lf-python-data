# This represents a simple numeric attribute

# TODO: implement (optional?) scaling!
# NOTE: automatic scaling is currently done in the Dataset class, should move here!!

class FeatureNumeric(object):

    def __init__(self, fname, attrinfo, featurestats):
        """For now, we do not do anything fancy for numeric features."""
        self.fname = fname
        self.attrinfo = attrinfo
        self.featurestats = featurestats

    def __call__(self, valueorlist):
        """Currently this just passes through the original value or
        list of values. The value should be a float!"""
        return valueorlist

    def type(self):
        return "numeric"

    def __str__(self):
        return "FeatureNumeric(name=%s)" % self.fname

    def __repr__(self):
        return "FeatureNumeric(name=%r)" % self.fname
