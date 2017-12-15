from feature import Feature

## This represents a simple attribute, of type number, boolean or string.
## the call method should convert a value of that type to the corresponding
## value(s) that needs to get fed into the network and ideally check the
## type against what is expected

class FeatureAttribute(Feature):

    def __init__(self,metainfo):
        """Create the instance from the meta info of an input feature"""


    def __call__(self):
        """Convert a value of the expected type for this feature to a value that can be
        fed into the corresponding input unit of the network"""
        pass