from builtins import *
from collections import Counter, OrderedDict, defaultdict
import logging
import gzip
import re
import numpy as np
import sys

# OK, the protocol for using this is this:
# * create a preliminary instance using "Vocab(...)"
#   This can be a completely empty instance or already contain an initial set of counts using counts=
#   and/or initial embeddings-information stored with the Vocab
#   NOTE: if an embedding file should be used, it should be specified at creation time
#   NOTE: emb_train can be "yes", "mapping", "no" or "onehot":
#   * yes: we only need to eventually store the embeddings for the words taken from our training set
#        and optionally calculate a OOV vector from the words that have low frequency.
#        Since we want to train, we should also train the OOV vector, so we should have training examples,
#        so we should remove the low-frequency words from the vocab.
#        embs: keep if in vocab and not used for OOV vector
#        vocab: keep if in embs, remove if used for OOV
#   * no: we remember all embeddings. The OOV vector could be random or calculated by averaging over the
#     known embeddings for low-frequency training set words.
#     Words in our own training set which are not in the embeddings vocabulary can be
#     removed and will get treated as OOV
#     embs: keep all except those used to calculate OOV
#     vocab: keep if in emb, except remove if used for oov
#   * mapping: keep all embeddings. OOV vector is average over low frequency training set words, again optionally
#     with replacement of the embedding for those with the OOV vector
#     embs: keep all
#     vocab: this is more complex - we could still create an OOV vector and remove some words found in the training
#       set to train a mapping for it. However at test time, we may want to go back and use the original embeddings
#       for those words. So for this we may eventually need to have .train() and .eval() functions on vocab.
#       Also words in vocab not in the embeddings should get a random vector assigned.
#       For now the strategy is this:
#       - all embeddings are kept except the ones used for OOV
#       - if a vocab is not in emb, it gets a random vector
#       - if a vocab is in emb and is low-freq, it is used to calculate the OOV vector and removed from vocab (and emb)
#   * onehot: ignore the file or throw an error/warning
#   * None: throws an error when finishing
#   NOTE: so this means:
#   * if an embedding is used to calculate OOV, word is removed from emb and removed from vocab
# * n times, add to the counts using v.add_counts() and also set embeddings-information using v.set_xxx()
#   and other info (e.g. additional symbols use) using v.set_xxx()
# * Once all the counts have been set, finish the datastructure and prepare for use using v.finish()
# * only after f.finish() is run the methods for mapping from and to indices, frequencies, or vectors
#   are usable, before finish() these methods will abort with an error!

# Configuration and how it affects lookup:
# * padding symbol / padding index : the string that will get mapped to a zero vector.
#   by default this is index=0 and symbol=the empty string
#
# * OOV vector / OOV index: the index returned for a word which is not in the vocabulary and the embedding vector
#   returned for it. By default this is 1
# * Other special symbols / indices: indices 2-9 are reserved for additional symbols like "<START>" but not
#   used by default.


class Vocab(object):
    """From the counter object, create string to id and id to string
    mappings."""
    def __init__(self, counts=None, max_size=None, min_freq=1, add_symbols=[], emb_id=None, emb_file=None,
                 no_special_indices=False,
                 emb_train=None, emb_dims=0, pad_string="", oov_string="<<oov>>",
                 oov_vec_from="random", oov_vec_maxfreq=1):
        """Create a vocabulary instance from the counts. If max_size is
        given sorts by frequency and only retains the max_size most frequent
        ones. Removes everything less the min_freq.
        Adds the symbols listed in add_symbols to index positions 2 and after.
        Raises an error if any of these symbols is already there.
        The padding index is always 0, however the string for which the padding index is returned can be set.
        The OOV index is always 1, however, it can be configured how to create the OOV vector using oov_vec_from
        which can be "random" (create a random vector), "maxfreqavg" (average all embeddings with at most
        oov_vec_maxfreq). If maxfreqavg is specified, then the words matching will always be removed from the
        embeddings and our own vocabulary, resulting in the OOV index and vector to get returned.
        If no_special_indices=True, then only the words from the original counts are added and no padding or oov
        or other special indices are added. In that case, trying to look up a symbol not in the vocabulary
        results in an exception.
        """
        logger = logging.getLogger(__name__)
        if counts:
            self.freqs = Counter(counts)
        else:
            self.freqs = Counter()
        self.no_special_indices = no_special_indices
        self.min_freq = min_freq
        self.add_symbols = add_symbols
        self.max_size = max_size
        self.emb_dims = emb_dims
        self.emb_id = emb_id
        self.emb_file = emb_file
        self.emb_train = emb_train
        self.itos = None
        self.stoi = None
        self.stoe = {}
        self.n = 0
        self.pad_string = pad_string
        self.oov_string = oov_string
        self.finished = False
        self.oov_vec_from = oov_vec_from
        self.oov_vec_maxfreq = oov_vec_maxfreq
        self.embeddings_loaded = False
        self.oov_emb = None   # get set in finish()
        if oov_vec_from != "random" and oov_vec_from != "maxfreqavg":
            raise Exception("Vocab parameter oov_vec_from must be one of: random, maxfreqavg")
        if emb_train == "onehot" and emb_file:
            logger.warning("Vocab parameter emb_file is ignored if emb_train is onehot")
        if emb_file:
            self.load_embeddings(emb_file)
        if not self.embeddings_loaded and (self.emb_train == "no" or self.emb_train == "mapping"):
            raise Exception("Vocab emb_train 'no' or 'mapping' not usable without loaded embeddings, "
                            "got emb_train=%s and emb_file=%s" % (self.emb_train, self.emb_file))


    def load_embeddings(self, emb_file):
        """Load pre-calculated embeddings from the given file. This will update embd_dim as needed!
        Currently only supports text format, compressed text format or a two file format where
        the file with extension ".vocab" has one word per line and the file with extension ".npy"
        is a matrix with as many rows as there are words and as many columns as there are dimensions.
        The format is identified by the presence of one of the extensions ".txt", ".txt.gz",
        or ".vocab" and ".npy" in the emb_file given.
        The text formats may or may not have a first line that indicates the number of words and
        number of dimensions.
        """
        if emb_file.endswith(".txt") or emb_file.endswith(".txt.gz"):
            if emb_file.endswith(".txt.gz"):
                reader = gzip.open
            else:
                reader = open
            with reader(emb_file, 'r', encoding="utf-8") as infile:
                n_lines = 0
                for line in infile:
                    if n_lines==0 and re.match(r'^\s*[0-9]+\s+[0-9]+\s*$', line):
                        continue
                    line = line.rstrip()
                    fields = re.split(r' +', line)
                    word = fields[0]
                    embstr = fields[1:]
                    embs = [float(e) for e in embstr]
                    self.stoe[word] = embs
        elif emb_file.endswith(".vocab") or emb_file.endswith(".npy"):
            raise Exception("TODO: format .vocab/.npy not yet implemented!")
        else:
            raise Exception("Embeddings file must have one of the extensions: .txt, .txt.gz, .vocab, .npy")
        self.embeddings_loaded = True

    @staticmethod
    def rnd_vec(dims=100, strng=None, as_numpy=True):
        """Returns a random vector of the given dimensions where each dimension is in [0.0..1.0).
        If str is None, the vector is dependent on the current numpy random state. If a string is given,
        then the random state is seeded with a number derived from the string first, so the random vector
        will always be the same for that string and number of dimensions."""
        if str:
            np.random.seed(hash(strng) % (2**32-1))
        vec = np.random.rand(dims)
        if as_numpy:
            return vec
        else:
            return list(vec)

    def zero_vec(self, as_numpy=True):
        if as_numpy:
            return np.zeros(self.emb_dims)
        else:
            return list(np.zeros(self.emb_dims))


    def add_counts(self, counts):
        """Incrementally add additional counts to the vocabulary. This can be done only before the finish
        method is called"""
        if self.finished:
            print("ERROR: Vocab method add_counts() cannot be called after finish()", file=sys.stderr)
            raise Exception("Vocab method add_counts() cannot be called after finish()")
        self.freqs.update(counts)

    def add_symbols(self, add_symbols=[]):
        """Incrementally add additional special symbols. By default, the vectors for these symbols will be random."""
        if self.finished:
            raise Exception("Vocab method add_symbols() cannot be called after finish()")
        if self.no_special_indices:
            return
        for s in add_symbols:
            if s in self.freqs:
                raise Exception("Additional symbol already in counts:", s)
        if isinstance(add_symbols, list):
            for s in add_symbols:
                if s in self.add_symbols:
                    raise Exception("Symbol already added:", s)
                else:
                    emb = Vocab.rnd_vec(self.emb_dims, s)
                    self.sto2[s] = emb
                    self.add_symbols.append(s)
        else:
            if add_symbols in self.add_symbols:
                raise Exception("Symbol already added:", add_symbols)
            else:
                emb = Vocab.rnd_vec(self.emb_dims, add_symbols)
                self.sto2[add_symbols] = emb
                self.add_symbols.append(add_symbols)

    def set_min_freq(self, min_freq=1):
        self.min_freq = min_freq

    def set_max_size(self, max_size=None):
        self.max_size=max_size

    def set_emb_id(self, id):
        self.emb_id = id

    def set_emb_file(self, file):
        self.emb_file = file

    def set_emb_dims(self, dim):
        self.emb_dims = dim

    def finish(self):
        """Build the actual vocab instance, it can only be used properly to look-up things after calling
        this method, but no parameters can be changed nor counts added after this."""
        if not self.emb_train:
            raise Exception("Vocab emb_train parameter never set")
        # NOTE: if we have embeddings loaded and we need the rare word embeddings for building the OOV vector,
        # calculate the average vector from the rare words before we remove anything!
        if self.embeddings_loaded and self.oov_vec_from == "maxfreqavg":  # we successfully loaded embeddings
            # go through all the entries in our vocabulary and check the frequency
            # if it is lower than oov_vec_maxfreq, try to get the embedding from the embedding file
            # if we got an embedding, add it to the sum and count, after going through all, calculate the mean
            sum = self.zero_vec()
            n = 0
            todelete = set()
            for s, f in self.freqs.items():
                if f <= self.oov_vec_maxfreq:
                    emb = self.stoe.get(s)
                    if emb:
                        # remember for removal
                        todelete.add(s)
                        sum += emb
                        n += 1
            self.oov_emb = sum / n
            # remove the words from both our vocab and the embeddings
            for s in todelete:
                del self.freqs[s]
                del self.stoe[s]
        elif self.oov_vec_from == "random":
            self.oov_emb = Vocab.rnd_vec(self.emb_dims, self.oov_string)

        # if embeddings have been loaded and we have train=yes|no remove the voc entry if not in the embeddings
        if self.embeddings_loaded and (self.emb_train == "yes" or self.emb_train == "no"):
            todelete = set()
            for s in self.freq:
                if s not in self.stoe:
                    todelete.add(s)
            for s in todelete:
                del self.freq[s]

        # for this, remove the emb if it is not in voc
        if self.embeddings_loaded and self.emb_train == "yes":
            todelete = set()
            for s in self.stoe:
                if s not in self.freq:
                    todelete.add(s)
            for s in todelete:
                del self.stoe[s]

        # got through the entries and put all the keys satisfying the min_freq limit into a list
        self.itos = [s for s in self.freqs if (self.freqs[s] >= self.min_freq)]
        # sort the keys by frequency, then alphabetically in reverse order
        self.itos = sorted(self.itos)
        self.itos = sorted(self.itos, reverse=True, key=lambda x: self.freqs[x])
        # add the additional symbols at the beginning, first and always at index 0, the pad symbol, except
        # when no_pad is True
        if not self.no_special_indices:
            self.itos = [self.pad_string] + [self.oov_string] + self.add_symbols + self.itos
        if self.max_size and len(self.itos) > self.max_size:
            self.itos = self.itos[:self.max_size]
        # now create the reverse map
        self.stoi = defaultdict(int)
        for i, s in enumerate(self.itos):
            self.stoi[s] = i
        self.n = len(self.itos)

        if not self.no_special_indices:
            self.stoe[self.oov_string] = self.oov_emb
            # make sure we have embeddings and idx for the padding as well
            self.stoe[self.pad_string] = self.zero_vec()
            self.stoi[self.pad_string] = 0

        # If we do not have embeddings loaded and train is yes, we just create random embeddings for all
        # the remaining words in the vocab
        # NOTE: we can much more easily directly use the random embeddings in the net, so we do not do this here!
        #if not self.embeddings_loaded and self.emb_train == "yes":
        #    for s in self.stoi:
        #        self.stoe[s] = Vocab.rnd_vec(dims=self.emb_dims)

        self.finished = True

    def idx2string(self, idx):
        """Return the string for this index"""
        if not self.finished:
            raise Exception("Vocab %r has not been finished!" % self)
        if idx >= len(self.itos):
            raise Exception("Vocab: index larger than vocabulary size")
        else:
            return self.itos[idx]

    def string2idx(self, string):
        if not self.finished:
            raise Exception("Vocab has not been finished!")
        if string in self.stoi:
            return self.stoi[string]  # NOTE: the pad string is in there!
        else:
            return 1  # the index of the OOV if not found!

    def string2emb(self, string):
        if not self.finished:
            raise Exception("Vocab has not been finished!")
        if string in self.stoe:
            return self.stoe[string]  # pad string is in there!
        else:
            return self.oov_emb

    def idx2emb(self, idx):
        return self.string2emb(self.idx2string(idx))

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

    def count(self, strng):
        """Return the count/frequency for the given word. NOTE: after finish() this will return 0 for any words
        that have been removed because of one of the filter criteria!!"""
        c = self.freqs.get(strng)
        if c:
            return c
        else:
            return 0

    def __str__(self):
        return "Vocab()"

    def __repr__(self):
        return "Vocab(emb_id=%r)" % self.emb_id
