import sys
from . targetnominal import TargetNominal

class Target(object):
    def __init__(self, *args):
        raise Exception("Target cannot be used directly, use a subclass")

    def __call__(self, valuelist):
        raise Exception("Target cannot be used directly, use a subclass")

    @classmethod
    def make(cls, meta):
        return TargetNominal()
