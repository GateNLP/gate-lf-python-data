"""Module for the TargetNominal class"""


from collections import Counter
from gatelfdata.vocab import Vocab
import sys
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
streamhandler = logging.StreamHandler(stream=sys.stderr)
formatter = logging.Formatter(
                '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
streamhandler.setFormatter(formatter)
logger.addHandler(streamhandler)


class TargetNominal(object):


    def __init__(self, meta, vocabs, targets_need_padding=False):
        self.meta = meta
        self.isSequence = meta["isSequence"]
        if self.isSequence:
            self.seq_max = meta["sequLengths.max"]
            self.seq_avg = meta["sequLengths.mean"]
        targetstats = meta["targetStats"]
        self.stringCounts = targetstats["stringCounts"]
        self.nrTargets = len(self.stringCounts)
        self.freqs = Counter(self.stringCounts)
        # so if we need to include a padding character for the targets, we set pad_index_only to True, if not,
        # we set no_special_indices True
        nspi = False
        pio = False
        if targets_need_padding:
            pio = True
        else:
            nspi = True
        self.vocab = Vocab(self.freqs, emb_id="<<TARGET>>", no_special_indices=nspi, pad_index_only=pio, emb_train="no")
        self.vocab.finish()
        vocabs.vocabs["<<TARGET>>"] = self.vocab
        # print("DEBUG!!!! Created vocab for target, itos is ", self.vocab.itos,  "pad_index_only is", self.vocab.pad_index_only, file=sys.stderr)
        # influences if the conversion will return the index or
        # the onehot vector
        self.as_onehot = False

    def set_as_onehot(self, flag=False):
        """Influence hot the original class label is converted. If
        the flag is False, then the string is converted to the corresponding
        string index, otherwise, to the corresponding onehot vector."""
        self.as_onehot = flag

    def zero_onehotvec(self):
        """Returns a zero vector with as many 0 as the one-hot representation would have."""
        return self.vocab.zero_onehotvec()

    def __call__(self, value, as_onehot=False):
        as_onehot = self.as_onehot or as_onehot
        if self.isSequence:
            if as_onehot:
                ret = [self.vocab.string2onehot(v) for v in value]
            else:
                ret = [self.vocab.string2idx(v) for v in value]
        else:
            if as_onehot:
                ret = self.vocab.string2onehot(value)
            else:
                ret = self.vocab.string2idx(value)
        # print("DEBUG looking up index for", value,"as_onehot=",as_onehot,"returning",ret,file=sys.stderr)
        return ret

    def idx2label(self, idx):
        return self.vocab.idx2string(idx)

    def __str__(self):
        return "TargetNominal()"

    def __repr__(self):
        return "TargetNominal()"
