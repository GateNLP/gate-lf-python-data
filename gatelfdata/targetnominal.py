from collections import Counter
from . vocab import Vocab

class TargetNominal(object):

    def __init__(self,meta):
        self.meta = meta
        self.isSequence = meta["isSequence"]
        if self.isSequence:
            self.seq_max = meta["sequLengths.max"]
            self.seq_avg = meta["sequLengths.mean"]
        targetStats = meta["targetStats"]
        self.stringCounts = targetStats["stringCounts"]
        self.nrTargets = len(self.stringCounts)
        self.freqs = Counter(self.stringCounts)
        self.vocab = Vocab(self.freqs, emb_id="<<TARGET>>", no_pad=True)
        self.vocab.finish()

    def __call__(self, value):
        if self.isSequence:
            ret = [self.vocab.string2onehot(v) for v in value]
            return ret
        else:
            return self.vocab.string2onehot(value)

    def __str__(self):
        return "TargetNominal()"

    def __repr__(self):
        return "TargetNominal()"
