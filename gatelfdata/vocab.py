from builtins import *
from collections import Counter, defaultdict
import logging
import gzip
import re
import numpy as np
import sys
import math

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
    def __init__(self, counts=None, max_size=None, emb_minfreq=1, add_symbols=None, emb_id=None, emb_file=None,
                 no_special_indices=False,
                 pad_index_only=False,
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
        The parameter pad_index_only creates a vocab where 0 is reserved for the pad index.
        NOTE: currently this is what we use for ALL nominal targets if they get represented as an index
        (and not as onehot).
        """
        if not add_symbols:
            add_symbols = []
        logger = logging.getLogger(__name__)
        if counts:
            self.freqs = Counter(counts)
        else:
            self.freqs = Counter()
        self.no_special_indices = no_special_indices
        self.pad_index_only = pad_index_only
        self.min_freq = emb_minfreq or 1
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
        self.embeddings = None
        self.oov_emb = None
        if oov_vec_from != "random" and oov_vec_from != "maxfreqavg":
            raise Exception("Vocab parameter oov_vec_from must be one of: random, maxfreqavg")
        if emb_train == "onehot" and emb_file:
            logger.warning("Vocab parameter emb_file is ignored if emb_train is onehot")
        if not self.emb_file and self.emb_train == "mapping":
            raise Exception("Vocab emb_train 'mapping' not usable without embeddings file, "
                            "got emb_train=%s and emb_file=%s" % (self.emb_train, self.emb_file))
        if self.emb_file and self.emb_train == "onehot":
            raise Exception("Vocab emb_train 'onehot' not usable with embeddings file, "
                            "got emb_train=%s and emb_file=%s" % (self.emb_train, self.emb_file))
        self.have_oov = True
        if no_special_indices or pad_index_only:
            self.have_oov = False
        print("DEBUGING initialized vocab", self.emb_id, "pad_index_only is", self.pad_index_only, file=sys.stderr)

    # TODO: encapsulate the self.stoe access: once we use loading the memory mapped numpy array
    # as an alternative loading method, we will have two possible ways of how to get the embedding,
    # either through self.stoe directly or though self.stoeidx to get the index and then the matrix
    # to get the embedding. Instead implement a method _get_emb(str) which will handle this
    # correctly. We also need to modify the code in finish() for removing embeddings to use
    # a method _del_emb(str) to remove from self.stoe or self.stoidx and finally "finish" the
    # embeddings to create the final packed matrix in all situations, for caching and fast
    # re-loading!

    def load_embeddings(self, emb_file, only4vocab=False):
        """Load pre-calculated embeddings from the given file. This will update embd_dim as needed!
        Currently only supports text format, compressed text format or a two file format where
        the file with extension ".vocab" has one word per line and the file with extension ".npy"
        is a matrix with as many rows as there are words and as many columns as there are dimensions.
        The format is identified by the presence of one of the extensions ".txt", ".txt.gz",
        or ".vocab" and ".npy" in the emb_file given.
        The text formats may or may not have a first line that indicates the number of words and
        number of dimensions.
        If only4vocab is True, the embeddings for words not in our own vocabulary will be ignored.
        NOTE: this will not check if the case conventions or other conventions (e.g. hyphens) for the tokens
        in our vocabulary are compatible with the conventions used for the embeddings.
        """
        if emb_file.endswith(".txt") or emb_file.endswith(".txt.gz"):
            if emb_file.endswith(".txt.gz"):
                reader = gzip.open
            else:
                reader = open
            with reader(emb_file, 'rt', encoding="utf-8") as infile:
                n_lines = 0
                for line in infile:
                    if n_lines == 0 and re.match(r'^\s*[0-9]+\s+[0-9]+\s*$', line):
                        continue
                    line = line.rstrip()
                    fields = re.split(r' +', line)
                    word = fields[0]
                    embstr = fields[1:]
                    embs = [float(e) for e in embstr]
                    if not only4vocab:
                        self.stoe[word] = embs
                    else:
                        if word in self.stoi:
                            self.stoe[word] = embs
                # update the emb_dims setting
                if embs and len(self.stoe) > 0:
                    self.emb_dims = len(embs)
        elif emb_file.endswith(".vocab") or emb_file.endswith(".npy"):
            raise Exception("TODO: format .vocab/.npy not yet implemented!")
        else:
            raise Exception("Embeddings file must have one of the extensions: .txt, .txt.gz, .vocab, .npy")
        self.embeddings_loaded = True

    def get_embeddings(self):
        """Return a numpy matrix of the embeddings in the order of the indices. If no embeddings have been
        loaded this returns None."""
        # NOTE: this simply returns the array that was created in the finish method!
        return self.embeddings

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

    def add_symbols(self, add_symbols=None):
        """Incrementally add additional special symbols. By default, the vectors for these symbols will be random."""
        if not add_symbols:
            add_symbols = []
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
                    self.stoe[s] = emb
                    self.add_symbols.append(s)
        else:
            if add_symbols in self.add_symbols:
                raise Exception("Symbol already added:", add_symbols)
            else:
                emb = Vocab.rnd_vec(self.emb_dims, add_symbols)
                self.stoe[add_symbols] = emb
                self.add_symbols.append(add_symbols)

    def set_min_freq(self, min_freq=1):
        self.min_freq = min_freq

    def set_max_size(self, max_size=None):
        self.max_size = max_size

    def set_emb_id(self, embid):
        self.emb_id = embid

    def set_emb_file(self, file):
        self.emb_file = file

    def set_emb_dims(self, dim):
        self.emb_dims = dim

    def finish(self):
        """Build the actual vocab instance, it can only be used properly to look-up things after calling
        this method, but no parameters can be changed nor counts added after this."""
        print("DEBUGING finishing vocab", self.emb_id, "pad_index_only is", self.pad_index_only, file=sys.stderr)

        # if the emb_train parameter was never set, try to come up with a sensible default here:
        # - if a file is specified, use the setting "no" for now,
        # - otherwise use "yes"
        if not self.emb_train:
            # we set our own default here: if a file is specified, then emb_train is no, otherwise
            # it is yes.
            if self.emb_file:
                self.emb_train = "no"
            else:
                self.emb_train = "yes"

        # print("DEBUG: finishing vocab for ", self.emb_id, file=sys.stderr)

        # Course of action:
        # 1) If words get removed from our own vocab because of frequency or max size, this
        # has to be done first.
        # 2) At this point we can load the embeddings and optionally limit to just what we have now
        # in the vocabulary (all cases except mapping)
        # 3) now if we need to calculate the OOV vector from rare words, do this and remove those
        #  words from the vocab and the embeddings.
        # 4) if we have mapping, add all embedding words left which are not in our vocab to our vocab
        # 5) we now know how big the matrix for the embeddings needs to be, create it and set the rows
        # 6) remove the dictionary stoe, we can do this using matrix[stoi] instead

        # got through the entries and put all the keys satisfying the min_freq limit into a list
        self.itos = [s for s in self.freqs if (self.freqs[s] >= self.min_freq)]
        # sort the keys by frequency, then alphabetically in reverse order
        self.itos = sorted(self.itos)
        self.itos = sorted(self.itos, reverse=True, key=lambda x: self.freqs[x])
        # add the additional symbols at the beginning, first and always at index 0, the pad symbol, except
        # when no_pad is True
        if self.no_special_indices:
            pass # do nothing what we have is all we need
            print("DEBUGING in finishing vocab, onehot", self.emb_id, "pad_index_only is", self.pad_index_only, "itos is",
                  self.itos, file=sys.stderr)
        elif self.pad_index_only:
            self.itos = [self.pad_string] + self.itos
            print("DEBUGING in finishing vocab", self.emb_id, "pad_index_only is", self.pad_index_only, "itos is",
                  self.itos, file=sys.stderr)
        else:
            self.itos = [self.pad_string] + [self.oov_string] + self.add_symbols + self.itos
        if self.max_size and len(self.itos) > self.max_size:
            self.itos = self.itos[:self.max_size]
        # now create the reverse map
        self.stoi = defaultdict(int)
        for i, s in enumerate(self.itos):
            self.stoi[s] = i
        self.n = len(self.itos)

        if not self.emb_file and not self.emb_dims:
            # caclulate some embeddings dimensions automatically from the number of words
            # for only a few words, we essentially want as many dimensions as there are words and
            # for a huge number we want somewhere in the 100s.
            # TODO: figure out something reasonable, for now implement something simple
            # this is 3 for 10, 10 for 100, 31 for 1000, 100 for 10k and 316 for 100k
            self.emb_dims = int(math.sqrt(self.n+2))


        # figure out if we need all embeddings, otherwise only load the ones corresponding to the words we have
        if self.emb_train == "mapping":
            only4vocab = False
        else:
            only4vocab = True
        if self.emb_file:
            self.load_embeddings(self.emb_file, only4vocab=only4vocab)

        if not self.no_special_indices:
            self.stoi[self.pad_string] = 0
            self.stoi[self.oov_string] = 1
            self.stoe[self.pad_string] = self.zero_vec()

        # if we need to calculate the OOV vector from the rare words, do this and then remove those
        # words from both our own list and the embeddings.
        todelete = set()
        if self.embeddings_loaded and self.oov_vec_from == "maxfreqavg":
            # go through all the entries in our vocabulary and check the frequency
            # if it is lower than oov_vec_maxfreq, try to get the embedding from the embedding file
            # if we got an embedding, add it to the sum and count, after going through all, calculate the mean
            embsum = self.zero_vec()
            n = 0
            for s, f in self.freqs.items():
                if f <= self.oov_vec_maxfreq:
                    emb = self.stoe.get(s)
                    if emb:
                        # remember for removal
                        todelete.add(s)
                        embsum += emb
                        n += 1
            self.oov_emb = embsum / n
            self.stoe[self.oov_string] = self.oov_emb
        elif self.oov_vec_from == "random":
            self.oov_emb = self.rnd_vec(dims=self.emb_dims)
            self.stoe[self.oov_string] = self.oov_emb
        # NOTE: if oov_vec_from is "random", we will simply use whatever the Embedding layer has assigned to
        # the our index 1
        # NOTE: currently if we do not have loaded embeddings but oov_vec_from is maxfreqavg, we throw
        # and error. We could calculate our own random vectors first and then do the above etc. but for now
        # it is not worth the effort
        if (not self.embeddings_loaded) and self.oov_vec_from == "maxfreqavg":
            raise Exception("Vocab: oov_vec_from='maxfreqavg' cannot be used without an embeddings file ")

        # remember if something needed to get deleted
        have_deleted = False

        # remove the words we used for OOV from freqs and stoe
        # print("DEBUG: todelete1=", todelete, file=sys.stderr)
        if len(todelete) > 0:
            have_deleted = True
        for s in todelete:
            del self.freqs[s]
            del self.stoe[s]

        # if we use loaded embeddings and train is no or yes, also remove all our own words if they are
        # not in the embeddings,
        todelete = set()
        if self.embeddings_loaded and (self.emb_train == "yes" or self.emb_train == "no"):
            for s in self.freqs:
                if s not in self.stoe:
                    todelete.add(s)
            if len(todelete) > 0:
                have_deleted = True
            # print("DEBUG: todelete2=", todelete, file=sys.stderr)
            for s in todelete:
                del self.freqs[s]

        # TODO: !!!somewhere around here, if we have mapping, then:
        # * add random embedding vectors to stoe for words only in our vocab: DONE
        # * extend our own vocab by the embeddings words not already in there
        # The latter requires re-building our datastructures
        # (NOTE: for mapping we always expect embeddings to be loaded, this has been checked earlier)
        if self.emb_train == "mapping":
            # add random vectors to all vocab entries not in the embeddings
            for s in self.itos:
                if s not in self.stoe:
                    self.stoe[s] = self.rnd_vec(dims=self.emb_dims)

        # now if we deleted words, first rebuild the itos and then the stoi also update n
        if have_deleted:
            new_itos = [self.pad_string] + [self.oov_string] + [s for s in self.itos if s in self.freqs]
            self.itos = new_itos
            self.stoi = defaultdict(int)
            for i, s in enumerate(self.itos):
                self.stoi[s] = i
            self.n = len(self.itos)

        # print("DEBUG: itos new=", self.itos, file=sys.stderr)
        # print("DEBUG: stoi new=", self.stoi, file=sys.stderr)
        # print("DEBUG: stoe new=", self.stoe, file=sys.stderr)

        # if we have embeddings, create the numpy matrix and fill it
        if self.embeddings_loaded:
            self.embeddings = np.zeros((self.n, self.emb_dims))
            for i in range(self.n):
                w = self.itos[i]
                # we should not get a key error here since we should have reduced our own vocab to what is in stoe
                emb = self.stoe[w]
                # print("DEBUG: w=", w, "emb=", emb, file=sys.stderr)
                # print("DEBUG: np=", np.array(emb), file=sys.stderr)
                self.embeddings[i, :] = np.array(emb)

        # at this point, we could remove the stoe and freq datastructures to save some memory
        # for now we keep the freqs
        self.stoe = None
        # self.freqs = None

        self.finished = True
        # print("DEBUG: just created vocab: ", self, file=sys.stderr)

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
            if self.have_oov:
                return self.stoi[self.oov_string]
            else:
                # not a proper word no oov character, for now throw an exception, this should probablly never happen
                raise Exception("String not found in vocab and do not have OOV symbol either: %s" % string)

    def string2emb(self, string):
        """Return the embedding for the string or OOV if not found or the zero vector if string is the padding symbol"""
        if not self.finished:
            raise Exception("Vocab has not been finished!")
        return self.embeddings[self.string2idx(string)]

    def idx2emb(self, idx):
        return self.embeddings[idx]

    def string2onehot(self, thestring):
        """return a one-hot vector for the string"""
        # TODO: if the string is not found this should instead return the onehot vector of the OOV, if
        # we do have an OOV symbol
        if not self.finished:
            raise Exception("Vocab %r has not been finished!" % self)
        vec = [0.0] * len(self.itos)
        if thestring in self.stoi:
            vec[self.stoi[thestring]] = 1.0
        elif self.have_oov:
            vec[self.stoi[self.oov_string]] = 1.0
        return vec

    def zero_onehotvec(self):
        return [0.0] * len(self.itos)

    def onehot2string(self, vec):
        if not self.finished:
            raise Exception("Vocab has not been finished!")
        # check if there is really just one 1.0 in the vector!
        # TODO
        idx = vec.index(1.0)   # TODO: this raises an exceptio if there is no 1.0
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
        return self.__repr__()+":nentries=%d" % len(self.stoi)

    def __repr__(self):
        return "Vocab(emb_id=%r,emb_train=%r,emb_file=%r,emb_dims=%d)" % (self.emb_id, self.emb_train, self.emb_file, self.emb_dims)
