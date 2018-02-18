from . featurenominal1ofk import FeatureNominal1ofk
from . featurenumeric import FeatureNumeric
from . featurenominalembs import FeatureNominalEmbs
from . featureboolean import FeatureBoolean
from . featurengram import FeatureNgram
import logging


class Feature(object):
    """Base class of all features. All information shared between some of the features is
    stored as class members of this base class."""
    def __init__(self, *args):
        raise Exception("Feature cannot be used for instances, use a type-specific class")

    @classmethod
    def make(cls, fname, datatype, attribute, featurestats):
        """Return the proper feature instance for the datatype, attribute
        and feature statistics."""
        kind = attribute["featureCode"]
        logger = logging.getLogger(__name__)
        logger.debug("Making feature for kind/name/type/attr: %r/%r/%r/%r", kind, fname, datatype, attribute)
        if kind == "N":
            # create an ngram feature, based on a simple feature of type nominal
            ret = FeatureNgram(fname, attribute, featurestats)
        else:
            # create a simple feature of the correct type
            if datatype == "nominal":
                # create a nominal feature of the correct kind for either
                # embedding or one-hot coding
                # This is decided by the setting of the corresponding
                # embedding definition.
                emb_train = attribute["codeas"]
                if emb_train == "onehot":
                    ret = FeatureNominal1ofk(fname, attribute, featurestats)
                else:
                    ret = FeatureNominalEmbs(fname, attribute, featurestats)
            elif datatype == "numeric":
                # simple numeric feature
                ret = FeatureNumeric(fname, attribute, featurestats)
            elif datatype == "boolean":
                # simple boolean feature
                ret = FeatureBoolean(fname, attribute, featurestats)
            else:
                raise Exception("Odd datatype: ", datatype)
        logger.debug("Returning: %r", ret)
        return ret
