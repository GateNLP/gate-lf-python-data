"""Module for the Target class"""
import sys
import logging
from gatelfdata.targetnominal import TargetNominal

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
streamhandler = logging.StreamHandler(stream=sys.stderr)
formatter = logging.Formatter(
                '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
streamhandler.setFormatter(formatter)
logger.addHandler(streamhandler)


class Target(object):
    def __init__(self, *args):
        raise Exception("Target cannot be used directly, use a subclass")

    def __call__(self, valuelist):
        raise Exception("Target cannot be used directly, use a subclass")

    @classmethod
    def make(cls, meta, vocabs, targets_need_padding=True):
        targetstats = meta["targetStats"]
        stringcounts = targetstats["stringCounts"]
        if len(stringcounts) == 0:
            raise Exception("Only nominal targets supported for now")
        return TargetNominal(meta, vocabs, targets_need_padding=targets_need_padding)
