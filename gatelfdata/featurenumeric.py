"""Module for the FeatureNumeric class"""

import sys
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
streamhandler = logging.StreamHandler(stream=sys.stderr)
formatter = logging.Formatter(
                '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
streamhandler.setFormatter(formatter)
logger.addHandler(streamhandler)


class FeatureNumeric(object):

    def __init__(self, fname, attrinfo, featurestats):
        """For now, we do not do anything fancy for numeric features."""
        self.fname = fname
        self.attrinfo = attrinfo
        self.featurestats = featurestats
        # store the values we need for normalization and decide if normalization should
        # be done by default
        self.mean = self.featurestats["mean"]
        self.var = self.featurestats["variance"]
        self.min = self.featurestats["min"]
        self.max = self.featurestats["max"]
        self.range = (self.max - self.min)
        self.n = self.featurestats["max"]
        # normalizer is either none or a function f(value) that returns the normalized value
        # For now we use mean/variance normalization by default: this can change once we implement
        # easier parametrization of this from the LF
        self.normalizer = self.normalize_meanvar

    def set_normalization(self, normalize=None):
        """Either one of 'meanvar', 'minmax' or a function that takes and returns a float or
        one of the normalization methods of this class."""
        if isinstance(normalize, str):
            self.normalizer = self._normalizer4str(normalize)
        else:
            self.normalizer = normalize

    def normalize_meanvar(self, value):
        if self.var > 0.0:
            return (value-self.mean)/self.var
        else:
            return value

    def normalize_minmax(self, value):
        if self.range > 0.0:
            return (value-self.min)/self.range
        else:
            return value

    def _normalizer4str(self, name):
        if name == 'minmax':
            return self.normalize_minmax
        elif name == 'meanvar':
            return self.normalize_meanvar
        else:
            raise Exception("Not a known normalization method: %s" % name)

    def normalize(self, value, normalize=None):
        """This normalizes the value using the currently set normalization if normalize is None,
        explicitly no normalization if normalize is false (overriding what is the default), or
        whatever the string or function set indicates.
        """
        if normalize:
            if isinstance(normalize, str):
                method = self._normalizer4str(normalize)
                return method(value)
            else:
                return normalize(value)
        else:
            if normalize is None:
                if self.normalizer:
                    return self.normalizer(value)
            else:  # must be False or other non-None false
                return value

    def __call__(self, valueorlist, normalize=None):
        """Currently this optionally normalizes the value or list of values,
        then passes through the original or normalized value or
        list of values. The value should be a float!"""
        if normalize:
            if isinstance(valueorlist, list):
                return [self.normalize(val, normalize=normalize) for val in valueorlist]
            else:
                return self.normalize(valueorlist, normalize=normalize)
        else:
            return valueorlist

    def type_converted(self):
        return "float"

    def type_original(self):
        return "numeric"

    def __str__(self):
        return "FeatureNumeric(name=%s)" % self.fname

    def __repr__(self):
        return "FeatureNumeric(name=%r)" % self.fname
