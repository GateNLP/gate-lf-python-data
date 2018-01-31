from __future__ import print_function
import sys
# This represents a simple numeric attribute

# TODO: implement (optional?) scaling!

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

    def __str__(self):
        return "FeatureNumeric(name=%s)" % self.fname

    def __repr__(self):
        return "FeatureNumeric(name=%r)" % self.fname
