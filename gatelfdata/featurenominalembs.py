


# This represents a simple nominal (string) attribute that should
# get encoded as a dense embedding vector

# NOTE: all embedding data for all FeatureNominalEmbs features is stored
# in this class and shared via class members in the instances, if necessary!

class FeatureNominalEmbs(object):

    # a map from embedding ids to EmbeddingsData
    embeddings = {}

    def __init__(self, datatype, attrinfo, featurestats):
        """Create the instance from the given meta info of an input feature"""


    def __call__(self,value):
        """Convert a value of the expected type for this feature to a value that can be
        fed into the corresponding input unit of the network"""
        #raise Exception("NOT YET IMPLEMENTED")
        return value

    def __str__(self):
        return "FeatureNgram(name="+self.fname+")"
