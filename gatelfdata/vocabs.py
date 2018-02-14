from . vocab import Vocab
import logging
import sys


class Vocabs(object):
    """A class for managing all the vocab instances that are needed by features"""

    # map from embedding id to vocab instance
    vocabs = {}

    @classmethod
    def setup_vocab(cls, attrinfo, featurestats):
        """Create or update the temporary Vocab instances so that the counts from different attributes get merged"""
        logger = logging.getLogger(__name__)
        logger.debug("Pre-initialising vocab for %r", attrinfo)
        counts = featurestats.get("stringCounts")
        if counts:
            emb_id = attrinfo.get("emb_id")
            if emb_id in cls.vocabs:
                vocab = cls.vocabs.get(emb_id)
                vocab.add_counts(counts)
            else:
                emb_train = attrinfo.get("emb_train")
                vocab = Vocab(featurestats["stringCounts"], emb_id=emb_id, emb_train=emb_train)
                cls.vocabs[emb_id] = vocab

    @classmethod
    def finish(cls):
        """Once all the counts have been gathered, create the final instances"""
        for id, vocab in cls.vocabs.items():
            vocab.finish()

    @classmethod
    def get_vocab(cls, attrinfo_or_embid):
        if isinstance(attrinfo_or_embid, dict):
            emb_id = attrinfo_or_embid.get("emb_id")
        else:
            emb_id = attrinfo_or_embid
        if emb_id in cls.vocabs:
            return cls.vocabs.get(emb_id)
        else:
            raise Exception("No vocab for emb_id: %s got %s" % (emb_id, cls.vocabs.keys()))

    @classmethod
    def init(cls):
        cls.vocabs = {}

    def __str__(self):
        return "Vocabs()"

    def __repr__(self):
        return "Vocabs()"
