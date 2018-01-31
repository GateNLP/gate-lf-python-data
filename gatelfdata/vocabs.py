from __future__ import print_function
import sys
from . vocab import Vocab


class Vocabs(object):
    """A class for managing all the vocab instances that are needed by features"""

    # map from embedding id to vocab instance
    vocabs = {}

    @classmethod
    def addOrReturnVocab(cls, attrinfo, featurestats):
        # print("DEBUG: adding/returning vocab for attr=", attrinfo, " stats=", featurestats, file=sys.stderr)
        print("DEBUG: adding/returning vocab for attr=", attrinfo, file=sys.stderr)
        emb_id = attrinfo.get("embd_id")
        if emb_id in cls.vocabs:
            print("DEBUG: returning existing vocab for", emb_id, file=sys.stderr)
            return cls.vocabs.get(emb_id)
        else:
            emb_file = attrinfo.get("emb_file")
            emb_train = attrinfo.get("emb_train")
            emb_dims = attrinfo.get("emb_dims")
            stringCounts = featurestats.get("stringCounts")
            # TODO: store the embedding infor with the vocab
            # TODO: if the embeddings should be loaded from a file, do that and also
            # check/update the embedding dimensions setting (which should get ignored in that case)
            vocab = Vocab(stringCounts)
            print("DEBUG: storing new vocab for", emb_id, file=sys.stderr)
            cls.vocabs[emb_id] = vocab
            return vocab