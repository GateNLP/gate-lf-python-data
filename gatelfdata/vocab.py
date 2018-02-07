from builtins import *
from collections import Counter, OrderedDict, defaultdict


# OK, the protocol for using this is this:
# * create a preliminary instance using "Vocab(...)"
#   This can be a completely empty instance or already contain an initial set of counts using counts=
#   and/or initial embeddings-information stored with the Vocab
# * n times, add to the counts using v.add_counts() and also set embeddings-information using v.set_xxx()
#   and other info (e.g. additional symbols use) using v.set_xxx()
# * Once all the counts have been set, finish the datastructure and prepare for use using v.finish()
# * only after f.finish() is run the methods for mapping from and to indices, frequencies, or vectors
#   are usable, before finish() these methods will abort with an error!

# IMPORTANT:

class Vocab(object):
    """From the counter object, create string to id and id to string
    mappings."""
    def __init__(self, counts=None, max_size=None, min_freq=1, add_symbols=[], emb_id=None, emb_file=None, emb_train=None, emb_dim=0, pad_symbol="<PAD>", no_pad=False):
        """Create a vocabulary instance from the counts. If max_size is
        given sorts by frequency and only retains the max_size most frequent
        ones. Removes everything less the min_freq. Finally adds the
        strings in add_symbols. Raises an error if any of these symbols
        is already there. NOTE: if a padding symbol is added, it must
        be the first in the list! If no padding symbol is added, some
        arbitrary symbol will get assigned index 0!!"""
        if counts:
            self.freqs = Counter(counts)
        else:
            self.frequs = Counter()
        self.min_freq = min_freq
        self.add_symbols = add_symbols
        self.max_size = max_size
        self.emb_dim = emb_dim
        self.emb_id = emb_id
        self.emb_file = emb_file
        self.emb_train = emb_train
        self.itos = None
        self.stoi = None
        self.n = 0
        if no_pad:
            self.pad_symbol = None
        else:
            self.pad_symbol = pad_symbol
        self.no_pad = no_pad
        self.finished = False

    def add_counts(self, counts):
        self.freqs.update(counts)

    def set_pad_symbol(self, symbol="<PAD>"):
        self.pad_symbol = symbol

    def add_symbols(self, add_symbols=[]):
        for s in add_symbols:
            if s in self.freqs:
                raise Exception("Additional symbol already in counts:",s)
        if isinstance(add_symbols,list):
            for s in add_symbols:
                if s in self.add_symbols:
                    raise Exception("Symbol already added:", s)
                else:
                    self.add_symbols.append(s)
        else:
            if add_symbols in self.add_symbols:
                raise Exception("Symbol already added:", add_symbols)
            else:
                self.add_symbols.append(add_symbols)

    def set_min_freq(self, min_freq=1):
        self.min_freq = min_freq

    def set_max_size(self, max_size=None):
        self.max_size=max_size

    def set_emb_id(self, id):
        self.emb_id = id

    def set_emb_file(self, file):
        self.emb_file = file

    def set_emb_dim(self, dim):
        self.emb_dim = dim

    def set_no_pad(self, flag):
        """Set if pad should be used or not"""
        self.no_pad = flag

    def finish(self):
        # got through the entries and put all the keys into a list
        self.itos = [s for s in self.freqs if (self.freqs[s] >= self.min_freq)]
        # sort the keys by frequency in reverse order
        self.itos = sorted(self.itos)
        self.itos = sorted(self.itos, reverse=True, key=lambda x: self.freqs[x])
        # add the additional symbols at the beginning, first and always at index 0, the pad symbol, except
        # when no_pad is True
        if self.no_pad:
            self.itos = self.add_symbols + self.itos
        else:
            self.itos = [self.pad_symbol] + self.add_symbols + self.itos
        if self.max_size and len(self.itos) > self.max_size:
            self.itos = self.itos[:self.max_size]
        # now create the reverse map
        self.stoi = defaultdict(int)
        for i, s in enumerate(self.itos):
            self.stoi[s] = i
        self.n = len(self.itos)
        self.finished = True

    def idx2string(self, idx):
        if not self.finished:
            raise Exception("Vocab %r has not been finished!" % self)
        # TODO: we may need to handle a couple of additional special cases here!
        if idx >= len(self.itos):
            return "<PAD>" # TODO: we need a more organized way of handling special symbols!!
        else:
            return self.itos[idx]

    def string2idx(self, string):
        if not self.finished:
            raise Exception("Vocab has not been finished!")
        if string in self.stoi:
            return self.stoi.get(string)
        else:
            return 0

    def string2onehot(self, thestring):
        """return a one-hot vector for the string"""
        if not self.finished:
            raise Exception("Vocab %r has not been finished!" % self)
        vec = [0.0] * len(self.itos)
        if thestring in self.stoi:
            vec[self.stoi[thestring]] = 1.0
        return vec

    def onehot2string(self, vec):
        if not self.finished:
            raise Exception("Vocab has not been finished!")
        # check if there is really just one 1.0 in the vector!
        # TODO
        idx = vec.index(1.0)  ## TODO: this raises an exceptio if there is no 1.0
        return self.itos[idx]

    def count(self, str):
        c = self.freqs.get(str)
        if c:
            return c
        else:
            return 0

    def __str__(self):
        return "Vocab()"

    def __repr__(self):
        return "Vocab(emb_id=%r)" % self.emb_id
