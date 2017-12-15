

class Feature(object):
    """Base class of all features. All information shared between some of the features is
    stored as class members of this base class."""
    def __init__(self, *args):
        raise Exception("Feature cannot be used directly, use a subclass")

    def __call__(self, valuelist):
        """All features need to implement this method. It will map the original feature value
        to a numpy array."""
        raise Exception("Feature cannot be used directly, use a subclass")