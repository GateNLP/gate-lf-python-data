import sys
from . targetnominal import TargetNominal

class Target(object):
    def __init__(self, *args):
        raise Exception("Target cannot be used directly, use a subclass")

    def __call__(self, valuelist):
        raise Exception("Target cannot be used directly, use a subclass")

    @classmethod
    def make(cls, meta):
        targetStats = meta["targetStats"]
        stringCounts = targetStats["stringCounts"]
        if len(stringCounts) == 0:
            raise Exception("Only nominal targets supported for now")
        return TargetNominal(targetStats)
