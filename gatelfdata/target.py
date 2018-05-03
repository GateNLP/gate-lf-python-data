from . targetnominal import TargetNominal


class Target(object):
    def __init__(self, *args):
        raise Exception("Target cannot be used directly, use a subclass")

    def __call__(self, valuelist):
        raise Exception("Target cannot be used directly, use a subclass")

    @classmethod
    def make(cls, meta, targets_need_padding=True):
        targetstats = meta["targetStats"]
        stringcounts = targetstats["stringCounts"]
        if len(stringcounts) == 0:
            raise Exception("Only nominal targets supported for now")
        return TargetNominal(meta, targets_need_padding=targets_need_padding)
