from __future__ import print_function
from . featurenominal1ofk import FeatureNominal1ofk
from . featurenumeric import FeatureNumeric
from . featurenominalembs import FeatureNominalEmbs
from . featureboolean import FeatureBoolean
from . featurengram import FeatureNgram
import sys

class Feature(object):
    """Base class of all features. All information shared between some of the features is
    stored as class members of this base class."""
    def __init__(self, *args):
        raise Exception("Feature cannot be used for instances, use a type-specific class")


    @classmethod
    def make(cls, fname, datatype, attribute, featurestats):
        """Return the proper feature instance for the datatype, attribute
        and feature statistics."""
        print("DEBUG: making feature for name/type/attr:",
              fname, datatype, attribute, file=sys.stderr)
        kind = attribute["featureCode"]
        if kind == "N":
            # create an ngram feature, based on a simple feature of type nominal
            return FeatureNgram(fname, attribute, featurestats)
            pass
        else:
            # create a simple feature of the correct type
            if datatype == "nominal":
                # create a nominal feature of the correct kind for either
                # embedding or one-hot coding
                # This is decided by the setting of the corresponding
                # embedding definition.
                emb_train = attribute["emb_train"]
                emb_id = attribute["emb_id"]
                if emb_train == "onehot":
                    return FeatureNominal1ofk(fname, attribute, featurestats)
                else:
                    return FeatureNominalEmbs(fname, attribute, featurestats)
            elif datatype == "numeric":
                # simple numeric feature
                return FeatureNumeric(fname, attribute, featurestats)
            elif datatype == "boolean":
                # simple boolean feature
                return FeatureBoolean(fname, attribute, featurestats)
            else:
                raise Exception("Odd datatype: ", datatype)