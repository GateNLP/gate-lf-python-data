# This represents a simple numeric attribute

# TODO: implement (optional?) scaling!

class FeatureNumeric(object):

    def __init__(self, attrinfo, featurestats):
        """For now, we do not do anything fancy for numeric features."""
        self.attrinfo = attrinfo
        self.featurestats = featurestats

    def __call__(self,valueorlist):
        """Currently this just passes through the original value or
        list of values. The value should be a float!"""
        return valueorlist
