"""Module for the Voabs class"""

import logging
from collections import defaultdict
from gatelfdata.vocab import Vocab
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
streamhandler = logging.StreamHandler(stream=sys.stderr)
formatter = logging.Formatter(
                '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
streamhandler.setFormatter(formatter)
logger.addHandler(streamhandler)


class Vocabs(object):
    """A class for managing all the vocab instances that are needed by features"""

    def __init__(self, remove_counts=True, remove_embs=True):
        """
        Create a vocabs instance and set the default behaviour when finishing each
        vocab.
        :param remove_counts: remove the counts per word data when finishing
        :param remove_embs: remove the embs per word data when finishing
        """
        # map from embedding id to vocab instance
        self.remove_counts = remove_counts
        self.remove_embs = remove_embs
        self.vocabs = defaultdict()

    def setup_vocab(self, attrinfo, featurestats):
        """Create or update the temporary Vocab instances so that the counts from different attributes get merged"""
        logger.debug("Pre-initialising vocab for %r", attrinfo)
        counts = featurestats.get("stringCounts")
        if counts:
            emb_id = attrinfo.get("emb_id")
            if emb_id in self.vocabs:
                vocab = self.vocabs.get(emb_id)
                vocab.add_counts(counts)
            else:
                emb_train = attrinfo.get("emb_train")
                emb_file = attrinfo.get("emb_file")
                emb_dims = attrinfo.get("emb_dims")
                emb_minfreq = attrinfo.get("emb_minfreq")
                vocab = Vocab(featurestats["stringCounts"],
                              emb_id=emb_id, emb_train=emb_train, emb_file=emb_file, emb_dims=emb_dims,
                              emb_minfreq=emb_minfreq)
                self.vocabs[emb_id] = vocab

    def finish(self):
        """Once all the counts have been gathered, create the final instances"""
        for _, vocab in self.vocabs.items():
            vocab.finish(remove_counts=self.remove_counts, remove_embs=self.remove_embs)
        logger.debug("Finished vocabs: %s" % (self.vocabs,))

    def get_vocab(self, attrinfo_or_embid):
        """Return a vocab instance for the given attribute name or embedding id."""
        if isinstance(attrinfo_or_embid, dict):
            emb_id = attrinfo_or_embid.get("emb_id")
        else:
            emb_id = attrinfo_or_embid
        if emb_id in self.vocabs:
            return self.vocabs.get(emb_id)
        else:
            raise Exception("No vocab for emb_id: %s got %s" % (emb_id, self.vocabs.keys()))

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "Vocabs(vocabs=%r)" % (self.vocabs.keys())
