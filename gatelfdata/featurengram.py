class FeatureNgram(object):

    def __init__(self, datatype, attrinfo, featurestats):
        """Create the instance from the given meta info of an input feature"""


    def __call__(self,value):
        """Convert a value of the expected type for this feature to a value that can be
        fed into the corresponding input unit of the network"""
        #raise Exception("NOT YET IMPLEMENTED")
        return value

    def __str__(self):
        return "FeatureNgram(name="+self.fname+")"
